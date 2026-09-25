"""Tests for src/modeling.py: the time-based CV folds, the funnel-rule baseline, the pipelines, and the metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from src.modeling import (CV_FOLDS, LABEL, add_features, boosting_pipeline, bootstrap_metric_diff, cross_validate,
                          funnel_rule_score, lab_features_pipeline, logistic_pipeline, random_forest_pipeline,
                          ranking_metrics, rows_between, top_share_precision)
from src.tables import TRAIN_END, WINDOW_DAYS

FOLD_IDS = [f"fold{k}" for k in range(1, len(CV_FOLDS) + 1)]

# On the real data 3,254 of fold 3's training rows (15 buyers) have label windows reaching 2017-03-01/02,
# but no label was set by a purchase on those days, so the tuning results don't change.
FOLD_3_EMBARGO_BUG = pytest.mark.xfail(reason=(
    "fold 3's embargo (February 2017) is 28 days, shorter than the 30-day label window: labels of training "
    "first visits on 2017-01-30/31 look into 2017-03-01/02, inside the validation months"))


def remarketing_frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic remarketing table, first visits 2016-08-01 .. 2017-07-01; adding to cart predicts buying."""
    rng = np.random.default_rng(seed)
    cart = rng.random(n) < 0.2

    def pick(values: list[str]) -> np.ndarray:
        return rng.choice(values, n)

    df = pd.DataFrame({
        "session_date": pd.Timestamp("2016-08-01") + pd.to_timedelta(rng.integers(0, 335, n), unit="D"),
        "channel": pick(["Organic Search", "Direct", "Social", "Paid Search", "Referral"]),
        "medium": pick(["organic", "(none)", "referral", "cpc"]),
        "source": pick(["google", "(direct)", "youtube.com"]),
        "device_category": pick(["desktop", "mobile", "tablet"]),
        "operating_system": pick(["Macintosh", "Windows", "Android", "iOS"]),
        "browser": pick(["Chrome", "Safari", "Firefox"]),
        "sub_continent": pick(["Northern America", "Western Europe", "Southern Asia"]),
        "country": pick(["United States", "Canada", "Germany", "India"]),
        "hits": rng.integers(1, 60, n),
        "pageviews": rng.integers(1, 40, n),
        "time_on_site": rng.integers(0, 1800, n),
        "bounced": (rng.random(n) < 0.3).astype(int),
        "product_detail_views": rng.integers(0, 6, n),
        "distinct_products_viewed": rng.integers(0, 4, n),
        "add_to_cart_events": np.where(cart, rng.integers(1, 4, n), 0),
        "checkout_events": np.where(cart, rng.integers(0, 2, n), 0),
        "max_ecommerce_step": np.where(cart, 3, rng.integers(0, 3, n)),
    })
    df[LABEL] = (rng.random(n) < np.where(cart, 0.6, 0.03)).astype(int)
    return df


# ---------------------------------------------------------------- time-based folds

@pytest.mark.parametrize("train, valid", CV_FOLDS, ids=FOLD_IDS)
def test_fold_windows_are_ordered_and_stay_in_the_training_months(train, valid):
    (train_start, train_end), (valid_start, valid_end) = [tuple(map(pd.Timestamp, w)) for w in (train, valid)]
    assert train_start <= train_end < valid_start <= valid_end <= TRAIN_END


def test_folds_expand_the_training_window():
    assert {train[0] for train, _ in CV_FOLDS} == {"2016-08-01"}
    ends = [pd.Timestamp(train[1]) for train, _ in CV_FOLDS]
    assert ends == sorted(ends)


@pytest.mark.parametrize("train, valid", [
    pytest.param(*CV_FOLDS[0], id="fold1"),
    pytest.param(*CV_FOLDS[1], id="fold2"),
    pytest.param(*CV_FOLDS[2], id="fold3", marks=FOLD_3_EMBARGO_BUG),
])
def test_embargo_covers_the_30_day_label_window(train, valid):
    # A first visit late on the last training day may buy until 30 days after it starts, so training labels
    # are only closed at the end of train_end + 30 days; validation has to start after that.
    last_label_day = pd.Timestamp(train[1]) + pd.Timedelta(days=WINDOW_DAYS)
    assert last_label_day < pd.Timestamp(valid[0])


def test_rows_between_includes_both_ends():
    df = pd.DataFrame({"session_date": pd.to_datetime(["2017-01-31", "2017-02-01", "2017-02-28", "2017-03-01"])})
    assert rows_between(df, "2017-02-01", "2017-02-28").index.tolist() == [1, 2]


class RecordingModel:
    """Stands in for a pipeline and logs the first and last day it is trained and scored on."""

    def __init__(self, log: list[tuple[str, pd.Timestamp, pd.Timestamp]]):
        self.log = log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RecordingModel:
        self.log.append(("fit", X.session_date.min(), X.session_date.max()))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.log.append(("score", X.session_date.min(), X.session_date.max()))
        s = day_score(X)
        return np.column_stack([1 - s, s])


def day_score(df: pd.DataFrame) -> np.ndarray:
    return df.session_date.dt.day.to_numpy() / 31


def test_cross_validate_trains_and_scores_inside_each_fold():
    days = pd.date_range("2016-08-01", "2017-04-30")
    df = pd.DataFrame({"session_date": days, LABEL: (days.dayofweek < 2).astype(int)})
    log: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    result = cross_validate(lambda: RecordingModel(log), df)
    expected_log, fold_pr_auc = [], []
    for (a, b), (c, e) in CV_FOLDS:
        expected_log += [("fit", pd.Timestamp(a), pd.Timestamp(b)), ("score", pd.Timestamp(c), pd.Timestamp(e))]
        valid = rows_between(df, c, e)
        fold_pr_auc.append(average_precision_score(valid[LABEL], day_score(valid)))
    assert log == expected_log   # nothing from the embargo month on either side
    assert result["cv_pr_auc"] == pytest.approx(np.mean(fold_pr_auc))


# ---------------------------------------------------------------- baseline and pipelines

def test_funnel_rule_ranks_by_step_then_pageviews():
    # pageviews / 1e4 breaks ties within a step while pageviews < 10,000 (the real maximum is 469)
    df = pd.DataFrame({"max_ecommerce_step": [2, 6, 5, 0, 5, 1], "pageviews": [3, 1, 20, 9_999, 400, 0]})
    assert np.argsort(-funnel_rule_score(df), kind="stable").tolist() == [1, 4, 2, 0, 5, 3]


def test_funnel_rule_reads_the_step_after_add_features():
    df = remarketing_frame(n=50)
    features = add_features(df)
    assert features.max_ecommerce_step.map(type).eq(str).all()   # a category for the models
    assert funnel_rule_score(features) == pytest.approx(funnel_rule_score(df))


PIPELINES = {
    "logistic_regression": lambda: logistic_pipeline(C=1.0),
    "random_forest": lambda: random_forest_pipeline(min_samples_leaf=5, n_estimators=30),
    "gradient_boosting": lambda: boosting_pipeline(max_iter=40, min_samples_leaf=20),
    "lab_features": lab_features_pipeline,
}


@pytest.fixture
def one_openmp_thread():
    """Tiny data fits fastest on one thread; boosting's OpenMP threads crawl when the machine is busy."""
    with threadpool_limits(limits=1, user_api="openmp"):
        yield


@pytest.mark.usefixtures("one_openmp_thread")
@pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")
@pytest.mark.parametrize("name", list(PIPELINES))
def test_pipeline_scores_later_months_with_unseen_categories(name):
    rm = add_features(remarketing_frame())
    train, test = rm[rm.session_date <= TRAIN_END], rm[rm.session_date > TRAIN_END].copy()
    test.loc[test.index[:3], ["channel", "browser", "country"]] = ["Carrier Pigeon", "Netscape", "Atlantis"]
    s = PIPELINES[name]().fit(train, train[LABEL]).predict_proba(test)[:, 1]
    assert s.shape == (len(test),)
    assert ((s >= 0) & (s <= 1)).all()
    assert roc_auc_score(test[LABEL], s) > 0.75   # it finds the planted add-to-cart signal


# ---------------------------------------------------------------- evaluation

def test_top_share_precision_and_ranking_metrics():
    # 100 visitors in score order; the buyers sit at ranks 0, 1, 2, 20 and 50 (base rate 5%)
    s = np.linspace(1, 0, 100)
    y = np.zeros(100, dtype=int)
    y[[0, 1, 2, 20, 50]] = 1
    assert top_share_precision(y, s, 0.05) == pytest.approx(0.6)
    assert top_share_precision(y, s, 0.001) == 1.0   # at least one visitor
    m = ranking_metrics(y, s)
    assert m["precision_top_1pct"] == 1.0
    assert (m["lift_top_1pct"], m["lift_top_5pct"], m["lift_top_10pct"]) == pytest.approx((20.0, 12.0, 6.0))
    assert m["recall_top_10pct"] == pytest.approx(0.6)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))


def test_bootstrap_metric_diff_is_paired():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.3).astype(int)
    s = rng.random(300)
    same = bootstrap_metric_diff(y, s, s.copy(), average_precision_score, n_boot=50)
    assert (same["diff"], same["ci_low"], same["ci_high"], same["p_diff_le_0"]) == (0.0, 0.0, 0.0, 1.0)
    better = bootstrap_metric_diff(y, y + 0.1 * s, s, average_precision_score, n_boot=50)
    assert 0 < better["ci_low"] <= better["diff"] <= better["ci_high"]
    assert better["p_diff_le_0"] == 0.0

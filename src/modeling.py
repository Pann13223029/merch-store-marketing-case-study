"""Remarketing model (D2): features, time-based validation, baselines, and evaluation.

Population and label come from src/tables.py:remarketing_table(): external first-visit
non-buyers; label = purchase on a later visit within 30 days of the first visit.

Validation respects time. Tuning uses expanding-window folds inside the training months,
each followed by an embargo of at least 30 days so that no training label (which looks 30 days
ahead) overlaps the validation months. The test months (2017-05-01 .. 2017-07-01) are touched once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

LABEL = "buy_within_30d"

# (train days, validation days) with at least 30 days between them left out as an embargo. February has
# only 28 days, so fold 3's training ends on Jan 29 rather than Jan 31.
CV_FOLDS = [
    (("2016-08-01", "2016-10-31"), ("2016-12-01", "2017-01-31")),
    (("2016-08-01", "2016-12-31"), ("2017-02-01", "2017-03-31")),
    (("2016-08-01", "2017-01-29"), ("2017-03-01", "2017-04-30")),
]

COUNT_FEATURES = ["hits", "pageviews", "time_on_site", "product_detail_views",
                  "distinct_products_viewed", "add_to_cart_events", "checkout_events"]
CATEGORICAL_FEATURES = ["channel", "device_category", "operating_system", "browser",
                        "sub_continent", "country", "max_ecommerce_step"]
LAB_NUMERIC = ["max_ecommerce_step", "bounced", "time_on_site", "pageviews"]
LAB_CATEGORICAL = ["source", "medium", "channel", "device_category", "country"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns shared by all models (no information from after the first visit)."""
    out = df.copy()
    out["viewed_product"] = (out.product_detail_views > 0).astype(int)
    out["added_to_cart"] = (out.add_to_cart_events > 0).astype(int)
    out["reached_checkout"] = (out.checkout_events > 0).astype(int)
    out["max_ecommerce_step"] = out.max_ecommerce_step.astype(int).astype(str)
    return out


def rows_between(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df.session_date >= pd.Timestamp(start)) & (df.session_date <= pd.Timestamp(end))]


# ---------------------------------------------------------------- model pipelines

def _one_hot(min_frequency: int = 200) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                         sparse_output=True)


def logistic_pipeline(C: float = 1.0) -> Pipeline:
    prep = ColumnTransformer([
        ("counts", Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]),
         COUNT_FEATURES),
        ("flags", "passthrough", ["bounced", "viewed_product", "added_to_cart", "reached_checkout"]),
        ("cats", _one_hot(), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=C, max_iter=5000))])


def random_forest_pipeline(min_samples_leaf: int = 50, max_features: str | float = "sqrt",
                           n_estimators: int = 300, seed: int = 42) -> Pipeline:
    prep = ColumnTransformer([
        ("counts", "passthrough", COUNT_FEATURES + ["bounced"]),
        ("cats", _one_hot(), CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                   max_features=max_features, n_jobs=-1, random_state=seed)
    return Pipeline([("prep", prep), ("model", model)])


def boosting_pipeline(learning_rate: float = 0.05, max_leaf_nodes: int = 15,
                      min_samples_leaf: int = 200, l2_regularization: float = 1.0,
                      max_iter: int = 300, seed: int = 42) -> Pipeline:
    """Gradient boosting with native categorical splits (no one-hot).

    Categories seen fewer than 200 times in training are grouped as infrequent; unseen
    categories become missing values, which the model routes like any missing value.
    """
    prep = ColumnTransformer([
        ("counts", "passthrough", COUNT_FEATURES + ["bounced"]),
        ("cats", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                encoded_missing_value=np.nan, min_frequency=200), CATEGORICAL_FEATURES),
    ], sparse_threshold=0)
    n_counts = len(COUNT_FEATURES) + 1
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization, max_iter=max_iter, early_stopping=False,
        categorical_features=list(range(n_counts, n_counts + len(CATEGORICAL_FEATURES))), random_state=seed)
    return Pipeline([("prep", prep), ("model", model)])


def lab_features_pipeline() -> Pipeline:
    """Baseline: the GSP229 lab's feature set and model type, refit on our corrected table and label."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), LAB_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), LAB_CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=np.inf, max_iter=5000))])


def funnel_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Baseline: rank by furthest funnel step reached, ties broken by pageviews (no model)."""
    return df.max_ecommerce_step.astype(int).to_numpy() + df.pageviews.to_numpy() / 1e4


def funnel_geo_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Baseline: North American first visits first, then the funnel rule within each group (no model).

    The funnel-rule score stays below 10 (step <= 6, pageviews / 1e4 < 1), so the region always comes first.
    """
    return 10.0 * (df.sub_continent == "Northern America").to_numpy() + funnel_rule_score(df)


# ---------------------------------------------------------------- evaluation
# Every "top share" cut takes the k = round(n * frac) highest scores (at least 1); tied scores keep row order.

def _top_k(s: np.ndarray, frac: float) -> np.ndarray:
    return np.argsort(-s, kind="stable")[:max(1, int(round(len(s) * frac)))]


def top_share_precision(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    """Share of positives among the top `frac` of scores."""
    return float(np.mean(y[_top_k(s, frac)]))


def top_share_count(y: np.ndarray, s: np.ndarray, frac: float) -> int:
    """Number of positives ranked in the top `frac` of scores."""
    return int(np.sum(y[_top_k(s, frac)]))


def top_share_recall(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    """Share of all positives ranked in the top `frac` of scores."""
    return top_share_count(y, s, frac) / float(np.sum(y))


def bottom_share_count(y: np.ndarray, s: np.ndarray, frac: float) -> int:
    """Number of positives ranked in the bottom `frac` of scores: everything below the top round(n * (1 - frac))."""
    return int(np.sum(y[np.argsort(-s, kind="stable")[int(round(len(s) * (1 - frac))):]]))


def expected_top_share_count(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    """top_share_count() if tied scores were ordered at random instead of by row order (expected value).

    Rules such as funnel_rule_score() tie thousands of visitors, so a cut can fall inside a block of ties.
    """
    k = max(1, int(round(len(s) * frac)))
    groups = pd.DataFrame({"s": s, "y": y}).groupby("s", sort=True).y.agg(["size", "sum"]).iloc[::-1]
    above = groups["size"].cumsum() - groups["size"]
    taken = np.clip(k - above, 0, groups["size"])
    return float((groups["sum"] * taken / groups["size"]).sum())


def ranking_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    base = float(np.mean(y))
    return {
        "pr_auc": average_precision_score(y, s),
        "roc_auc": roc_auc_score(y, s),
        "precision_top_1pct": top_share_precision(y, s, 0.01),
        "lift_top_1pct": top_share_precision(y, s, 0.01) / base,
        "lift_top_5pct": top_share_precision(y, s, 0.05) / base,
        "lift_top_10pct": top_share_precision(y, s, 0.10) / base,
        "recall_top_10pct": top_share_recall(y, s, 0.10),
    }


def cross_validate(make_pipeline, df: pd.DataFrame, folds=CV_FOLDS) -> dict:
    """Mean PR-AUC and ROC-AUC over the time-based folds (plus per-fold PR-AUC)."""
    prs, rocs = [], []
    for (tr_start, tr_end), (va_start, va_end) in folds:
        tr, va = rows_between(df, tr_start, tr_end), rows_between(df, va_start, va_end)
        s = make_pipeline().fit(tr, tr[LABEL]).predict_proba(va)[:, 1]
        prs.append(average_precision_score(va[LABEL], s))
        rocs.append(roc_auc_score(va[LABEL], s))
    return {"cv_pr_auc": float(np.mean(prs)), "cv_roc_auc": float(np.mean(rocs)),
            "fold_pr_auc": [round(p, 4) for p in prs]}


def bootstrap_metric_diff(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, metric,
                          n_boot: int = 2000, seed: int = 42) -> dict:
    """metric(A) - metric(B) on the same rows, with a paired bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    point = metric(y, s_a) - metric(y, s_b)
    n, diffs = len(y), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = metric(y[idx], s_a[idx]) - metric(y[idx], s_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "ci_low": lo, "ci_high": hi, "p_diff_le_0": float(np.mean(diffs <= 0))}


def bootstrap_metric_ci(y: np.ndarray, s: np.ndarray, metric, n_boot: int = 2000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = [metric(y[idx], s[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return tuple(np.percentile(vals, [2.5, 97.5]))


# ---------------------------------------------------------------- tuning

TUNING_GRID = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [{"min_samples_leaf": leaf, "max_features": mf}
                      for leaf in (20, 100) for mf in ("sqrt", 0.3)],
    "gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": nodes, "min_samples_leaf": leaf}
                          for lr in (0.05, 0.1) for nodes in (15, 31) for leaf in (100, 400)],
}
# Second pass: both tree models peaked at the edge of the first grid, so extend it in that direction
TUNING_GRID_EXTENSION = {
    "random_forest": [{"min_samples_leaf": leaf, "max_features": "sqrt"} for leaf in (5, 10)],
    "gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": nodes, "min_samples_leaf": leaf}
                          for lr, nodes, leaf in [(0.05, 7, 400), (0.05, 15, 1000), (0.05, 7, 1000),
                                                  (0.02, 15, 400), (0.02, 7, 1000)]],
}
# Third pass: boosting still peaked at the most regularised edge of the second pass
TUNING_GRID_EXTENSION_2 = {
    "gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": nodes, "min_samples_leaf": leaf}
                          for lr, nodes, leaf in [(0.02, 4, 1000), (0.02, 7, 2000), (0.01, 7, 1000),
                                                  (0.02, 2, 1000), (0.02, 3, 1000)]],
}
PIPELINES = {"logistic_regression": logistic_pipeline, "random_forest": random_forest_pipeline,
             "gradient_boosting": boosting_pipeline}


def tune(train: pd.DataFrame, grid_spec: dict | None = None, log=print) -> pd.DataFrame:
    """Time-based CV over a grid (default TUNING_GRID); one row per (model, params)."""
    rows = []
    for name, grid in (grid_spec or TUNING_GRID).items():
        for params in grid:
            r = cross_validate(lambda: PIPELINES[name](**params), train)
            rows.append({"model": name, "params": params, **r})
            log(f"{name:20s} {params}  cv PR-AUC {r['cv_pr_auc']:.4f}  ROC-AUC {r['cv_roc_auc']:.4f}  folds {r['fold_pr_auc']}")
    return pd.DataFrame(rows)

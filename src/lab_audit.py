"""Replication and audit of Google's GSP229 lab models (BigQuery ML logistic regression).

The lab's models are re-fit with scikit-learn using the same preprocessing BigQuery ML applies
automatically: one-hot encoding for string features and standardization for numeric features.
BigQuery ML's logistic_reg default has no regularization, so none is used here either (C = inf).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LABEL = "will_buy_on_return_visit"
MODEL_1 = {"numeric": ["bounces", "time_on_site"], "categorical": []}
MODEL_2 = {
    "numeric": ["latest_ecommerce_progress", "bounces", "time_on_site", "pageviews"],
    "categorical": ["source", "medium", "channel", "device_category", "country"],
}
PUBLISHED_ROC_AUC = {"model_1": 0.72, "model_2": 0.91}


def lab_pipeline(spec: dict) -> Pipeline:
    steps = [("num", StandardScaler(), spec["numeric"])]
    if spec["categorical"]:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), spec["categorical"]))
    return Pipeline([
        ("prep", ColumnTransformer(steps)),
        ("model", LogisticRegression(C=np.inf, max_iter=5000)),
    ])


def fit(train: pd.DataFrame, spec: dict) -> Pipeline:
    X = train[spec["numeric"] + spec["categorical"]]
    return lab_pipeline(spec).fit(X, train[LABEL])


def scores(model: Pipeline, df: pd.DataFrame, spec: dict) -> np.ndarray:
    return model.predict_proba(df[spec["numeric"] + spec["categorical"]])[:, 1]


def top_k_precision(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    """Share of positives among the top `frac` of rows ranked by score."""
    k = max(1, int(round(len(s) * frac)))
    top = np.argsort(-s, kind="stable")[:k]
    return float(np.mean(y[top]))


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    base = float(np.mean(y))
    p_top1 = top_k_precision(y, s, 0.01)
    p_top10 = top_k_precision(y, s, 0.10)
    return {
        "rows": len(y),
        "positives": int(np.sum(y)),
        "base_rate": base,
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "precision_top_1pct": p_top1,
        "lift_top_1pct": p_top1 / base,
        "lift_top_10pct": p_top10 / base,
    }


def paired_bootstrap_auc_gap(
    y: np.ndarray, s: np.ndarray, subset: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict:
    """AUC on all rows minus AUC on a subset (e.g. external visitors), with a paired bootstrap CI.

    Each bootstrap resample draws rows with replacement from the full set, then computes both AUCs
    on that same resample, so the two estimates share their sampling noise.
    """
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, s) - roc_auc_score(y[subset], s[subset])
    n = len(y)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb, mb = y[idx], s[idx], subset[idx]
        gaps[b] = roc_auc_score(yb, sb) - roc_auc_score(yb[mb], sb[mb])
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"auc_gap": point, "ci_low": lo, "ci_high": hi, "p_gap_le_0": float(np.mean(gaps <= 0))}


def paired_bootstrap_auc_diff(
    y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict:
    """ROC-AUC of score A minus ROC-AUC of score B on the same rows, with a paired bootstrap CI."""
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"auc_diff": point, "ci_low": lo, "ci_high": hi, "p_diff_le_0": float(np.mean(diffs <= 0))}

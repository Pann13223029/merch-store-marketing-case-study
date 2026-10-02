"""Tests for the paired bootstrap helpers in src/lab_audit.py, and the H4 headline on the committed predictions.

data/raw/a13_bqml_predictions.parquet (row-level scores from the three BigQuery ML models on the lab's evaluation
set) is committed, so the headline check runs in CI too; it skips if the file is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.lab_audit import paired_bootstrap_auc_diff, paired_bootstrap_auc_gap

PREDICTIONS = Path(__file__).resolve().parents[1] / "data" / "raw" / "a13_bqml_predictions.parquet"


def noisy_scores(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, a fairly good score, a weaker score) for n rows, about a quarter positive."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    return y, y + rng.normal(0, 0.8, n), y + rng.normal(0, 2.0, n)


def test_auc_gap_is_zero_when_the_subset_is_every_row():
    y, s, _ = noisy_scores()
    gap = paired_bootstrap_auc_gap(y, s, np.ones(len(y), dtype=bool), n_boot=50)
    assert gap["auc_gap"] == 0.0
    assert gap["ci_low"] == 0.0 and gap["ci_high"] == 0.0
    assert gap["p_gap_le_0"] == 1.0


def test_auc_gap_point_and_interval():
    y, s, _ = noisy_scores()
    subset = np.random.default_rng(1).random(len(y)) < 0.6
    gap = paired_bootstrap_auc_gap(y, s, subset, n_boot=200, seed=3)
    assert gap["auc_gap"] == pytest.approx(roc_auc_score(y, s) - roc_auc_score(y[subset], s[subset]))
    assert gap["ci_low"] <= gap["auc_gap"] <= gap["ci_high"]
    assert gap == paired_bootstrap_auc_gap(y, s, subset, n_boot=200, seed=3)   # reproducible


def test_auc_diff_point_interval_and_sign():
    y, good, weak = noisy_scores()
    same = paired_bootstrap_auc_diff(y, good, good, n_boot=50)
    assert same["auc_diff"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0
    diff = paired_bootstrap_auc_diff(y, good, weak, n_boot=200, seed=3)
    assert diff["auc_diff"] == pytest.approx(roc_auc_score(y, good) - roc_auc_score(y, weak))
    assert 0 < diff["ci_low"] <= diff["auc_diff"] <= diff["ci_high"]
    assert diff["p_diff_le_0"] == 0.0


@pytest.mark.skipif(not PREDICTIONS.exists(), reason="no data/raw/a13_bqml_predictions.parquet")
def test_published_lab_model_auc_gap_from_employees():
    # H4: the lab model's ROC-AUC on all first visits minus on external visitors only is about 0.047 (0.910 vs 0.863)
    pred = pd.read_parquet(PREDICTIONS)
    y = pred.will_buy_on_return_visit.astype(int).to_numpy()
    s = pred.score_model_2.to_numpy()
    external = ~pred.is_internal.astype(bool).to_numpy()
    assert len(pred) == 102_025
    assert roc_auc_score(y, s) == pytest.approx(0.910, abs=5e-4)
    assert roc_auc_score(y[external], s[external]) == pytest.approx(0.863, abs=5e-4)
    gap = paired_bootstrap_auc_gap(y, s, external, n_boot=40)
    assert gap["auc_gap"] == pytest.approx(0.0468, abs=5e-4)
    assert 0.03 < gap["ci_low"] < gap["auc_gap"] < gap["ci_high"] < 0.065

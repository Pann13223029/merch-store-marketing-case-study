"""Checks on the real journey table, skipped when data/processed/journeys.parquet isn't cached locally.

Notebook 02 builds the cache from BigQuery and it is gitignored, so CI runs without these checks. The
expected totals are the reported ones (reports/03_process.md): 5,074 purchases, $683,533.14 capped revenue.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.attribution import MODELS, Attribution, credit_vs_presence

JOURNEYS = Path(__file__).resolve().parents[1] / "data" / "processed" / "journeys.parquet"
PURCHASES = 5_074
CAPPED_REVENUE_USD = 683_533.14

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not JOURNEYS.exists(), reason="no local data/processed/journeys.parquet (notebook 02 builds it)"),
]


@pytest.fixture(scope="module")
def journeys() -> pd.DataFrame:
    return pd.read_parquet(JOURNEYS)


def test_purchase_totals_match_the_reports(journeys):
    conv = journeys[journeys.converted]
    assert len(conv) == PURCHASES
    assert conv.revenue_capped_usd.sum() == pytest.approx(CAPPED_REVENUE_USD, abs=0.005)


# The headline chain (arrival paths, order 3), the orders compared when choosing it, and the conservative relabel
@pytest.mark.parametrize("path_col, order", [
    ("arrival_path", 1), ("arrival_path", 2), ("arrival_path", 3), ("conservative_path", 3),
])
def test_every_model_conserves_credit(journeys, path_col, order):
    table = Attribution(journeys, path_col, order=order).credit_table()
    for model in MODELS:
        assert table[(model, "conversions")].sum() == pytest.approx(PURCHASES), model
        assert table[(model, "revenue")].sum() == pytest.approx(CAPPED_REVENUE_USD, abs=0.01), model


def test_presence_check_flags_markov_over_credit(journeys):
    # The third-order chain credits Affiliates with 16.2 purchases from 4 purchasing journeys, and Social
    # with 89 from 75.
    check = credit_vs_presence(Attribution(journeys, "arrival_path", order=3))
    assert check.index[check.exceeds_presence].tolist() == ["Affiliates", "Social"]
    assert check.loc["Affiliates", "credited"] == pytest.approx(16.23, abs=0.005)
    assert check.loc["Affiliates", "presence"] == 4

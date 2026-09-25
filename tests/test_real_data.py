"""Checks on the real journey table, skipped when data/processed/journeys.parquet isn't cached locally.

Notebook 02 builds the cache from BigQuery and it is gitignored, so CI runs without these checks. The key account
(src/segments.py) is dropped on load, as notebook 05 does, so the checks hold whether the cache was built before
or after src/journeys.py began leaving key accounts out. Expected totals, without the key account: 5,058
purchases and $664,824.30 of revenue capped per purchase session at the unchanged Process cap, $1,605.91.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.attribution import MODELS, Attribution, credit_vs_presence
from src.segments import is_key_account

JOURNEYS = Path(__file__).resolve().parents[1] / "data" / "processed" / "journeys.parquet"
PURCHASES = 5_058
CAPPED_REVENUE_USD = 664_824.30
REVENUE_CAP_USD = 1_605.908

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not JOURNEYS.exists(), reason="no local data/processed/journeys.parquet (notebook 02 builds it)"),
]


@pytest.fixture(scope="module")
def journeys() -> pd.DataFrame:
    j = pd.read_parquet(JOURNEYS)
    return j[~is_key_account(j.full_visitor_id)].reset_index(drop=True)


def test_purchase_totals_match_the_reports(journeys):
    conv = journeys[journeys.converted]
    assert len(conv) == PURCHASES
    assert conv.revenue_capped_usd.sum() == pytest.approx(CAPPED_REVENUE_USD, abs=0.005)


def test_revenue_is_capped_at_the_process_cap(journeys):
    # leaving the key account out doesn't move the cap (without it the 99th percentile would be about $1,506)
    assert journeys.revenue_capped_usd.max() == pytest.approx(REVENUE_CAP_USD, abs=0.0005)


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
    # The third-order chain credits Affiliates with 16.1 purchases from 4 purchasing journeys, and Social
    # with 88.6 from 75.
    check = credit_vs_presence(Attribution(journeys, "arrival_path", order=3))
    assert check.index[check.exceeds_presence].tolist() == ["Affiliates", "Social"]
    assert check.loc["Affiliates", "credited"] == pytest.approx(16.11, abs=0.005)
    assert check.loc["Affiliates", "presence"] == 4
    assert check.loc["Social", "credited"] == pytest.approx(88.60, abs=0.005)
    assert check.loc["Social", "presence"] == 75

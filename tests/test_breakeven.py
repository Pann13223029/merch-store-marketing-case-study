"""Tests for src/breakeven.py: the break-even arithmetic and the score bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.breakeven import (BANDS, CENTRAL, REVENUE_CAP_USD, breakeven_lift, max_affordable_cost, sensitivity,
                           value_by_band, value_of_targets)

BAND_LABELS = ["Top 1%", "1%–2%", "2%–5%", "5%–10%", "10%–20%", "20%–50%", "50%–100%"]


def ranked_visitors(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(rank, score) for n visitors in shuffled order: rank 0 has the highest score."""
    rank = np.random.default_rng(seed).permutation(n)
    return rank, 1.0 - rank / n


def test_central_assumptions_match_the_reports():
    # 10% lift and 50% margin in the central case; revenue capped at the $1,606 99th-percentile order (D-P2)
    assert CENTRAL == {"lift": 0.10, "margin": 0.50}
    assert REVENUE_CAP_USD == 1606


def test_max_affordable_cost_is_revenue_times_lift_times_margin():
    assert max_affordable_cost(14.06) == pytest.approx(14.06 * 0.10 * 0.50)
    assert max_affordable_cost(100.0, lift=0.08, margin=0.30) == pytest.approx(2.4)
    assert max_affordable_cost(pd.Series([0.0, 10.0, 40.0]), 0.2, 0.7) == pytest.approx([0.0, 1.4, 5.6])


def test_sensitivity_grid():
    bands = pd.DataFrame({"band": ["Top 1%", "1%–2%"], "revenue_per_visitor": [20.0, 4.0]})
    grid = sensitivity(bands)
    assert grid.shape == (2, 15)   # 3 margins x 5 lifts
    assert grid.index.tolist() == ["Top 1%", "1%–2%"]
    assert grid.loc["Top 1%", ("margin 30%", "lift 5%")] == pytest.approx(20.0 * 0.05 * 0.30)
    assert grid.loc["1%–2%", ("margin 70%", "lift 20%")] == pytest.approx(4.0 * 0.20 * 0.70)
    assert grid.loc["Top 1%", ("margin 50%", "lift 10%")] == pytest.approx(max_affordable_cost(20.0))


def test_value_by_band_boundaries():
    # 1,000 visitors: bands hold ranks 0-9, 10-19, 20-49, 50-99, 100-199, 200-499 and 500-999. With revenue
    # equal to rank, each band's revenue per visitor is the mean rank inside it, which pins its edges.
    rank, score = ranked_visitors(1000)
    y = (rank < 20).astype(int)   # the top 2% buy
    bands = value_by_band(y, rank.astype(float), score, n_boot=50)
    assert bands.band.tolist() == BAND_LABELS
    assert bands.visitors.tolist() == [10, 10, 30, 50, 100, 300, 500]
    assert bands.revenue_per_visitor.tolist() == [4.5, 14.5, 34.5, 74.5, 149.5, 349.5, 749.5]
    assert bands.buyers.tolist() == [10, 10, 0, 0, 0, 0, 0]
    assert bands.buy_rate.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("n", [250, 1001, 91_431])
def test_value_by_band_covers_every_visitor_once(n):
    rank, score = ranked_visitors(n)
    bands = value_by_band(np.zeros(n, dtype=int), rank.astype(float), score, n_boot=5)
    assert bands.visitors.sum() == n
    assert len(bands) == len(BANDS)


def test_value_by_band_caps_revenue_per_visitor():
    rank, score = ranked_visitors(100)
    revenue = np.where(rank == 0, 10_000.0, 0.0)   # one bulk buyer in the top 1%
    bands = value_by_band((rank == 0).astype(int), revenue, score, n_boot=50)
    assert bands.revenue_per_visitor[0] == REVENUE_CAP_USD
    assert bands.revenue_per_visitor[1:].eq(0).all()


def test_value_by_band_bootstrap_interval():
    rank, score = ranked_visitors(1000)
    revenue = np.random.default_rng(1).gamma(1.0, 50.0, 1000)
    bands = value_by_band(np.zeros(1000, dtype=int), revenue, score, n_boot=200, seed=3)
    assert (bands.rpv_ci_low <= bands.revenue_per_visitor).all()
    assert (bands.revenue_per_visitor <= bands.rpv_ci_high).all()
    pd.testing.assert_frame_equal(bands, value_by_band(np.zeros(1000, dtype=int), revenue, score, n_boot=200, seed=3))


def test_breakeven_lift_inverts_max_affordable_cost():
    assert breakeven_lift(0.10, 2.23) == pytest.approx(0.10 / (2.23 * 0.50))
    lift = breakeven_lift(0.15, 3.0, margin=0.30)
    assert max_affordable_cost(3.0, lift=lift, margin=0.30) == pytest.approx(0.15)


def test_value_of_targets_arithmetic():
    # revenue equal to rank: the top 1% of 1,000 visitors (ranks 0-9) spent 45, the top 10% (ranks 0-99) 4,950
    rank, score = ranked_visitors(1000)
    y = (rank < 20).astype(int)
    v = value_of_targets(y, rank.astype(float), score, months=2.0, targets=(0.01, 0.10), n_boot=50)
    assert v.target.tolist() == ["top 1%", "top 10%"]
    assert v.visitors_per_month.tolist() == [5.0, 50.0]
    assert v.share_of_later_buyers.tolist() == [0.5, 1.0]
    assert v.revenue_per_visitor.tolist() == pytest.approx([4.5, 49.5])
    assert v.gross_profit_per_month.tolist() == pytest.approx([45 * 0.05 / 2, 4950 * 0.05 / 2])
    # lift 5% x margin 30% and lift 20% x margin 70% scale the central 10% x 50% by 0.3 and 2.8
    assert v.range_low.tolist() == pytest.approx((v.gross_profit_per_month * 0.3).tolist())
    assert v.range_high.tolist() == pytest.approx((v.gross_profit_per_month * 2.8).tolist())
    assert (v.ci_low <= v.gross_profit_per_month).all() and (v.gross_profit_per_month <= v.ci_high).all()


def test_value_of_targets_default_targets_match_the_band_table():
    rank, score = ranked_visitors(2000)
    revenue = np.random.default_rng(2).gamma(0.5, 40.0, 2000)
    months = 2.0066
    v = value_of_targets(np.ones(2000, dtype=int), revenue, score, months, n_boot=10)
    bands = value_by_band(np.ones(2000, dtype=int), revenue, score, n_boot=10)
    assert v.target.tolist() == ["top 1%", "top 5%", "top 10%", "top 20%"]
    top1 = bands.revenue_per_visitor[0] * bands.visitors[0] * 0.05 / months
    assert v.gross_profit_per_month[0] == pytest.approx(top1)


def test_value_of_targets_zero_lift_visitors_cost_but_add_nothing():
    rank, score = ranked_visitors(100)
    revenue = np.where(rank == 1, 10_000.0, 10.0)   # rank 1 is a bulk buyer, capped at $1,606
    staff = rank == 0
    v = value_of_targets(np.ones(100, dtype=int), revenue, score, months=1.0, targets=(0.02,), zero_lift=staff,
                         n_boot=20)
    assert v.visitors_per_month[0] == 2                        # staff still count as remarketed visitors
    assert v.revenue_per_visitor[0] == pytest.approx((0.0 + REVENUE_CAP_USD) / 2)


def test_value_of_targets_is_reproducible():
    rank, score = ranked_visitors(500)
    revenue = np.random.default_rng(5).gamma(1.0, 20.0, 500)
    a = value_of_targets(np.ones(500, dtype=int), revenue, score, months=1.0, n_boot=100, seed=7)
    pd.testing.assert_frame_equal(a, value_of_targets(np.ones(500, dtype=int), revenue, score, months=1.0,
                                                      n_boot=100, seed=7))

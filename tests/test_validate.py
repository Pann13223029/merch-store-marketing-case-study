"""Tests for src/validate.py.

validate_sessions() reports each check as PASS or FAIL (it doesn't raise). The checks compare the extract with
totals measured on the raw BigQuery tables. A small synthetic table can't match those, so most tests swap RAW
for the synthetic table's own totals.

visitor_concentration() flags buyers who hold a large share of a channel's revenue.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src import validate
from src.validate import validate_sessions, visitor_concentration

N_CHECKS = 15
SYNTHETIC_TOTALS = {
    "sessions": 6,
    "purchase_sessions": 2,
    "revenue_usd": 150,
    "internal_visitors": 1,
    "internal_sessions": 2,
    "internal_purchases": 1,
    "internal_revenue_usd": 50,   # $50.50 observed: the check allows $1 of rounding
}


@pytest.fixture
def sessions(session) -> pd.DataFrame:
    """A consistent clean session table spanning 2016-08-01 .. 2017-08-01."""
    return pd.DataFrame([
        session("a", "2016-08-01 10:00"),
        session("a", "2016-09-01 10:00", visit_number=2, purchased=True, revenue=100.0),
        session("b", "2017-08-01 10:00", bounced=True, hits=1, pageviews=1, time_on_site=0),
        session("c", "2017-01-01 10:00", internal=True, is_internal_entry=True),
        session("c", "2017-01-02 10:00", visit_number=2, internal=True, purchased=True, revenue=50.5),
        session("d", "2017-03-01 10:00"),
    ])


@pytest.fixture
def synthetic_totals(monkeypatch) -> None:
    """Swap the raw totals for the synthetic table's own (restored after each test)."""
    for key, value in SYNTHETIC_TOTALS.items():
        monkeypatch.setitem(validate.RAW, key, value)


def failed(checks: pd.DataFrame) -> list[str]:
    return checks.loc[checks.status == "FAIL", "check"].tolist()


def test_passes_on_a_consistent_table(sessions, synthetic_totals):
    checks = validate_sessions(sessions)
    assert len(checks) == N_CHECKS
    assert failed(checks) == []


def test_raw_totals_are_compared_as_measured(sessions):
    # without the swap, the six-row table fails exactly the checks against the raw totals
    assert failed(validate_sessions(sessions)) == [
        "Row count matches raw", "Purchase sessions match raw", "Revenue matches raw",
        "Internal visitors match Prepare", "Internal sessions / purchases / revenue match Prepare",
    ]


def _set(row: int, column: str, value: object) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Mutation that sets one cell; `value` may be a function of the table."""
    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[row, column] = value(df) if callable(value) else value
        return df
    return mutate


# One way to break each check (the mutation may break others too)
BROKEN = {
    "Row count matches raw": lambda df: df.drop(index=5),
    "session_key is unique": _set(1, "session_key", lambda df: df.session_key[0]),
    "Purchase sessions match raw": _set(5, "purchased", True),
    "Revenue matches raw": _set(1, "revenue_usd", 120.0),
    "Internal visitors match Prepare": _set(5, "is_internal", True),
    "Internal sessions / purchases / revenue match Prepare": _set(4, "revenue_usd", 80.0),
    "Internal flag is constant within each visitor": _set(4, "is_internal", False),
    "No NULLs in engagement / outcome columns": _set(0, "pageviews", np.nan),
    "No negative values": _set(0, "time_on_site", -5),
    "Date range is 2016-08-01 to 2017-08-01": _set(2, "session_date", pd.Timestamp("2017-07-31")),
    "is_new_visit agrees with visit_number == 1": _set(1, "is_new_visit", True),
    "Bounced sessions have <= 1 pageview and 0 time on site (< 0.01% exceptions)": _set(2, "pageviews", 3),
    "Purchase sessions reach the purchase step in hit data (>95%)": _set(1, "max_ecommerce_step", 5),
    "Revenue > 0 only on purchase sessions": _set(0, "revenue_usd", 9.99),
}


@pytest.mark.parametrize("check", list(BROKEN))
def test_each_check_fails_on_a_broken_table(sessions, synthetic_totals, check):
    checks = validate_sessions(BROKEN[check](sessions.copy()))
    assert len(checks) == N_CHECKS   # every check still runs and reports
    assert check in failed(checks)


def test_purchases_without_revenue_must_stay_rare(sessions, synthetic_totals, session):
    free = [session(f"z{i}", "2017-02-01 10:00", purchased=True, revenue=0.0) for i in range(60)]
    checks = validate_sessions(pd.concat([sessions, pd.DataFrame(free)], ignore_index=True))
    assert "Purchase sessions without revenue stay negligible (< 60)" in failed(checks)


# ---------------------------------------------------------------- visitor concentration

def purchases_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Purchase sessions from (channel, visitor, revenue) rows."""
    return pd.DataFrame(rows, columns=["channel", "full_visitor_id", "revenue_usd"])


@pytest.fixture
def purchases() -> pd.DataFrame:
    return purchases_frame([
        ("Display", "big", 50.0), ("Display", "big", 10.0), ("Display", "y", 30.0), ("Display", "z", 10.0),
        *[("Organic Search", f"v{i}", 20.0) for i in range(5)],   # five buyers at exactly 20% each
        ("Affiliates", "w", 0.0),
    ])


def test_concentration_totals_and_top_buyer(purchases):
    c = visitor_concentration(purchases, "channel")
    assert c.index.tolist() == ["Display", "Organic Search", "Affiliates"]   # largest revenue first
    assert c.loc["Display", ["purchases", "buyers", "revenue", "top_visitor"]].tolist() == [4, 3, 100.0, "big"]
    assert c.loc["Display", "top_share"] == pytest.approx(0.6)   # a buyer's purchases add up
    assert c.loc["Organic Search", "top_share"] == pytest.approx(0.2)


def test_concentration_flags_every_buyer_above_the_threshold(purchases):
    c = visitor_concentration(purchases, "channel")
    assert c.loc["Display", "flagged"] == ["big", "y"]           # 60% and 30%, largest first
    assert c.loc["Organic Search", "flagged"] == []               # exactly 20% is not more than 20%
    assert visitor_concentration(purchases, "channel", threshold=0.5).loc["Display", "flagged"] == ["big"]


def test_concentration_top_n_share(purchases):
    c = visitor_concentration(purchases, "channel", top_n=2)
    assert c.loc["Display", "top_n_share"] == pytest.approx(0.9)
    assert c.loc["Organic Search", "top_n_share"] == pytest.approx(0.4)
    assert visitor_concentration(purchases, "channel").loc["Display", "top_n_share"] == pytest.approx(1.0)


def test_concentration_of_a_channel_without_revenue(purchases):
    c = visitor_concentration(purchases, "channel")
    assert np.isnan(c.loc["Affiliates", "top_share"]) and np.isnan(c.loc["Affiliates", "top_n_share"])
    assert c.loc["Affiliates", "flagged"] == []


def test_concentration_by_other_labels_and_revenue(purchases):
    # another label column (the Display purchases arrived direct) and revenue capped per purchase at $20
    p = purchases.assign(arrival_channel=purchases.channel.replace({"Display": "Direct"}),
                         capped=purchases.revenue_usd.clip(upper=20.0))
    c = visitor_concentration(p, "arrival_channel", "capped")
    assert c.loc["Direct", "revenue"] == pytest.approx(60.0)      # 20 + 10 + 20 + 10
    assert c.loc["Direct", "top_share"] == pytest.approx(0.5)     # "big": 20 + 10
    assert c.loc["Direct", "flagged"] == ["big", "y"]

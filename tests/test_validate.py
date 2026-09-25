"""Tests for src/validate.py:validate_sessions(), which reports each check as PASS or FAIL (it doesn't raise).

The checks compare the extract with totals measured on the raw BigQuery tables. A small synthetic table
can't match those, so most tests swap RAW for the synthetic table's own totals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src import validate
from src.validate import validate_sessions

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

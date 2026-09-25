"""Tests for src/journeys.py: who is in, journey splitting, the 30-day lookback, the analysis period, channel
labels, and the revenue cap."""

from __future__ import annotations

import pandas as pd
import pytest

from src.journeys import WINDOW_S, journeys, revenue_cap, touches
from src.segments import KEY_ACCOUNTS

DAY = 86400
T0 = int(pd.Timestamp("2017-01-10 12:00", tz="UTC").timestamp())
KEY = next(iter(KEY_ACCOUNTS))


def journey_ids(t: pd.DataFrame) -> list[int]:
    """Journey ids renumbered 0, 1, 2... in order of appearance, so tests can compare the grouping."""
    return pd.factorize(t.journey_id)[0].tolist()


def test_a_purchase_ends_the_journey(session):
    t = touches(pd.DataFrame([session("v", T0), session("v", T0 + DAY, purchased=True), session("v", T0 + 2 * DAY)]))
    assert journey_ids(t) == [0, 0, 1]
    assert t.converted.tolist() == [True, True, False]


def test_a_new_visitor_starts_a_new_journey(session):
    t = touches(pd.DataFrame([session("v1", T0), session("v2", T0 + 3600)]))
    assert journey_ids(t) == [0, 1]


def test_a_gap_of_more_than_30_days_starts_a_new_journey(session):
    # exactly 30 days apart: same journey; 30 days and 1 second: a new one
    t = touches(pd.DataFrame([session("v", T0), session("v", T0 + WINDOW_S), session("v", T0 + 2 * WINDOW_S + 1)]))
    assert journey_ids(t) == [0, 0, 1]


def test_converting_journeys_keep_only_the_30_days_before_the_purchase(session):
    buy = T0 + 60 * DAY
    t = touches(pd.DataFrame([
        session("v", buy - WINDOW_S - 1),   # one journey (no gap over 30 days), but outside the lookback
        session("v", buy - WINDOW_S),       # exactly 30 days before: kept
        session("v", buy - 10 * DAY),
        session("v", buy, purchased=True),
    ]))
    assert t.visit_start_time.tolist() == [buy - WINDOW_S, buy - 10 * DAY, buy]
    assert journey_ids(t) == [0, 0, 0]
    assert t.touch_index.tolist() == [1, 2, 3]   # renumbered after the trim


def test_non_converting_journeys_are_not_trimmed(session):
    t = touches(pd.DataFrame([session("v", T0), session("v", T0 + 25 * DAY), session("v", T0 + 50 * DAY)]))
    assert journey_ids(t) == [0, 0, 0]
    assert t.touch_index.tolist() == [1, 2, 3]


@pytest.mark.parametrize("last_day, kept", [
    ("2016-08-30", False), ("2016-08-31", True), ("2017-07-01", True), ("2017-07-02", False),
])
def test_analysis_period_is_set_by_the_journey_end(session, last_day, kept):
    # a journey that starts before the period still counts, with all its touches, if it ends inside it
    last = int(pd.Timestamp(f"{last_day} 12:00", tz="UTC").timestamp())
    t = touches(pd.DataFrame([session("v", last - 16 * DAY), session("v", last)]))
    assert len(t) == (2 if kept else 0)


def test_internal_visitors_are_excluded(session):
    t = touches(pd.DataFrame([session("staff", T0, internal=True), session("v", T0)]))
    assert t.full_visitor_id.tolist() == ["v"]


def test_key_accounts_are_excluded_like_employees(session):
    t = touches(pd.DataFrame([
        session(KEY, T0, channel="Display", source="dfa"),
        session(KEY, T0 + DAY, visit_number=2, channel="Display", source="dfa", is_true_direct=True, purchased=True,
                revenue=47_082.06),
        session("v", T0),
    ]))
    assert t.full_visitor_id.tolist() == ["v"]


def test_revenue_cap_is_set_on_all_outside_purchases_in_the_period(session):
    # Median for readable numbers. Key accounts count (the cap was set with them in), employees and purchases
    # outside the analysis period don't, and neither do sessions without a purchase.
    rows = [
        session("a", "2016-08-31 12:00", purchased=True, revenue=10.0),    # first day of the period
        session("b", "2017-07-01 12:00", purchased=True, revenue=20.0),    # last day of the period
        session(KEY, "2017-04-05 12:00", purchased=True, revenue=47_082.06),
        session("c", "2017-01-10 12:00"),                                  # no purchase
        session("staff", "2017-01-10 12:00", internal=True, purchased=True, revenue=5_000.0),
        session("d", "2016-08-30 12:00", purchased=True, revenue=3_000.0),    # before the period
        session("e", "2017-07-02 12:00", purchased=True, revenue=4_000.0),    # after it
    ]
    df = pd.DataFrame(rows)
    assert revenue_cap(df, quantile=0.5) == 20.0
    assert revenue_cap(df.iloc[[0, 1]], quantile=0.5) == 15.0   # without the key account the median would fall
    assert revenue_cap(df) == pytest.approx(20.0 + 0.98 * (47_082.06 - 20.0))   # the 99th percentile by default


def test_three_channel_labels(session):
    rows = [
        # fresh search click
        session("v", T0, channel="Organic Search", source="google"),
        # direct return that GA labels with the earlier campaign (source stays the campaign's)
        session("v", T0 + DAY, channel="Organic Search", source="google", is_true_direct=True),
        # direct return whose source is literally "(direct)", still labelled with the earlier campaign by GA
        session("v", T0 + 2 * DAY, channel="Organic Search", source="(direct)", is_true_direct=True),
        # "(direct)" source without isTrueDirect (117k real sessions): neither relabel touches it
        session("v", T0 + 3 * DAY, channel="Organic Search", source="(direct)"),
        # typed URL
        session("v", T0 + 4 * DAY, channel="Direct", source="(direct)", is_true_direct=True, purchased=True),
    ]
    t = touches(pd.DataFrame(rows))
    assert t.arrival_channel.tolist() == ["Organic Search", "Direct", "Direct", "Organic Search", "Direct"]
    assert t.conservative_channel.tolist() == ["Organic Search", "Organic Search", "Direct", "Organic Search",
                                               "Direct"]
    assert t.ga_channel.tolist() == ["Organic Search"] * 4 + ["Direct"]


def test_journeys_table(session):
    rows = [
        session("v", T0, channel="Paid Search", source="google"),
        session("v", T0 + DAY, channel="Paid Search", source="google", is_true_direct=True, purchased=True,
                revenue=2000.0),
        session("w", T0, channel="Social", source="youtube.com"),
        session("x", T0, channel="Referral", source="blog.example", purchased=True, revenue=40.0),
    ]
    j = journeys(touches(pd.DataFrame(rows)), revenue_cap_usd=1606.0).set_index("full_visitor_id")
    assert j.loc["v", "arrival_path"] == "Paid Search > Direct"
    assert j.loc["v", "conservative_path"] == "Paid Search > Paid Search"
    assert j.loc["v", "ga_path"] == "Paid Search > Paid Search"
    assert j.loc["v", "ga_last_channel"] == "Paid Search"
    assert j.n_touches.to_dict() == {"v": 2, "w": 1, "x": 1}
    assert j.converted.to_dict() == {"v": True, "w": False, "x": True}
    assert j.revenue_usd.to_dict() == {"v": 2000.0, "w": 0.0, "x": 40.0}
    assert j.revenue_capped_usd.to_dict() == {"v": 1606.0, "w": 0.0, "x": 40.0}   # capped per order
    assert (j.loc["v", "start_date"], j.loc["v", "end_date"]) == (pd.Timestamp("2017-01-10"),
                                                                   pd.Timestamp("2017-01-11"))

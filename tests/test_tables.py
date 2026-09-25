"""Tests for src/tables.py:remarketing_table(): the D2 population, the 30-day label, and the split."""

from __future__ import annotations

import pandas as pd
import pytest

from src.tables import WINDOW_DAYS, remarketing_table

WINDOW_S = WINDOW_DAYS * 86400
T0 = int(pd.Timestamp("2017-01-10 12:00", tz="UTC").timestamp())


@pytest.mark.parametrize("delay_s, label", [
    (WINDOW_S - 1, 1),   # 30 days minus 1 second after the first visit starts
    (WINDOW_S, 1),       # the window is inclusive
    (WINDOW_S + 1, 0),
])
def test_label_counts_a_later_purchase_within_30_days(session, delay_s, label):
    rm = remarketing_table(pd.DataFrame([session("v", T0), session("v", T0 + delay_s, visit_number=2, purchased=True)]))
    assert rm.buy_within_30d.tolist() == [label]
    assert rm.revenue_30d_usd.tolist() == [25.0 * label]
    assert rm.return_visits_30d.tolist() == [label]


def test_population_is_external_first_visit_non_buyers(session):
    rm = remarketing_table(pd.DataFrame([
        session("keep", T0),
        session("bought_first", T0, purchased=True),                    # bought on the first visit
        session("staff", T0, internal=True),                            # Google employee
        session("staff", T0 + 86400, visit_number=2, internal=True, purchased=True),
        session("returning", T0, visit_number=3),                       # first visit predates the data
    ]))
    assert rm.full_visitor_id.tolist() == ["keep"]


@pytest.mark.parametrize("first_visit, split", [
    ("2016-08-01 12:00", "train"),
    ("2017-04-30 12:00", "train"),
    ("2017-05-01 12:00", "test"),
    ("2017-07-01 12:00", "test"),    # the last first-visit day whose 30-day window fits in the data
    ("2017-07-02 12:00", None),      # excluded
])
def test_first_visit_date_sets_the_split_and_the_cutoff(session, first_visit, split):
    rm = remarketing_table(pd.DataFrame([session("v", first_visit)]))
    assert rm.split.tolist() == ([] if split is None else [split])


def test_midnight_split_first_visit_becomes_one_row(session):
    # GA splits a session that crosses midnight into two rows with the same visit_id
    before = int(pd.Timestamp("2017-01-20 23:50", tz="UTC").timestamp())
    after = int(pd.Timestamp("2017-01-21 00:00", tz="UTC").timestamp())
    rm = remarketing_table(pd.DataFrame([
        session("v", before, visit_id=7, hits=3, pageviews=2, time_on_site=500, bounced=True),
        session("v", after, visit_id=7, hits=4, pageviews=3, time_on_site=100),
        session("v", after + 4 * 86400, visit_number=2, purchased=True, revenue=10.0),
    ]))
    assert len(rm) == 1
    row = rm.iloc[0]
    assert row.first_start == before
    assert row.session_date == pd.Timestamp("2017-01-20")
    assert (row.hits, row.pageviews, row.time_on_site) == (7, 5, 600)   # summed over both halves
    assert row.bounced == 0   # a bounce only if both halves bounced
    assert (row.buy_within_30d, row.revenue_30d_usd, row.return_visits_30d) == (1, 10.0, 1)   # 2nd half isn't a return


def test_purchase_in_the_second_half_of_a_split_first_visit_excludes_the_visitor(session):
    before = int(pd.Timestamp("2017-01-20 23:50", tz="UTC").timestamp())
    rm = remarketing_table(pd.DataFrame([
        session("v", before, visit_id=7),
        session("v", before + 600, visit_id=7, purchased=True),
    ]))
    assert rm.empty


def test_two_first_visits_keep_the_earliest(session):
    # 608 real visitors have two visit_number = 1 visits; the later one then counts as a return visit
    rm = remarketing_table(pd.DataFrame([
        session("v", T0 + 2 * 86400, visit_id=2, purchased=True),
        session("v", T0, visit_id=1),
    ]))
    assert rm.visit_id.tolist() == [1]
    assert rm.buy_within_30d.tolist() == [1]


def test_outcome_counts_later_visits_in_the_window(session):
    rm = remarketing_table(pd.DataFrame([
        session("v", T0),
        session("v", T0 + 86400, visit_number=2),
        session("v", T0 + 2 * 86400, visit_number=3, purchased=True, revenue=30.0),
        session("v", T0 + 3 * 86400, visit_number=4, purchased=True, revenue=12.5),
        session("v", T0 + WINDOW_S + 60, visit_number=5, purchased=True, revenue=999.0),   # too late
    ]))
    assert (rm.buy_within_30d[0], rm.revenue_30d_usd[0], rm.return_visits_30d[0]) == (1, 42.5, 3)


# ---------------------------------------------------------------- internal_flag="first_visit" (no hindsight)

def test_default_table_has_no_internal_column(session):
    rm = remarketing_table(pd.DataFrame([session("v", T0)]))
    assert rm.equals(remarketing_table(pd.DataFrame([session("v", T0)]), internal_flag="visitor"))
    assert "is_internal" not in rm


def test_first_visit_flag_drops_only_an_internal_first_visit(session):
    # is_internal is the whole-year visitor flag; is_internal_entry marks the session that set it
    sessions = pd.DataFrame([
        session("keep", T0),
        session("staff_first", T0, internal=True, is_internal_entry=True),        # knowable at scoring time
        session("staff_first", T0 + 86400, visit_number=2, internal=True, purchased=True),
        session("staff_later", T0 + 60, internal=True),                           # known only in hindsight
        session("staff_later", T0 + 86400, visit_number=2, internal=True, is_internal_entry=True,
                purchased=True, revenue=40.0),
    ])
    assert remarketing_table(sessions).full_visitor_id.tolist() == ["keep"]
    rm = remarketing_table(sessions, internal_flag="first_visit")
    assert rm.full_visitor_id.tolist() == ["keep", "staff_later"]
    assert rm.is_internal.tolist() == [False, True]
    # the later internal-entry session counts toward the label, as it would in a live campaign
    assert (rm.buy_within_30d.tolist(), rm.revenue_30d_usd.tolist()) == ([0, 1], [0.0, 40.0])


def test_first_visit_table_is_the_default_table_plus_the_added_visitors(session):
    sessions = pd.DataFrame([
        session("a", T0 + 10),
        session("b", T0, pageviews=9),
        session("staff", T0, internal=True),              # starts in the same second as "b": placed after it
        session("staff", T0 + 3600, visit_number=2, internal=True, is_internal_entry=True),
        session("c", T0 + 5, visit_id=3),
        session("c", T0 + 7200, visit_number=2, purchased=True),
    ])
    default = remarketing_table(sessions)
    rm = remarketing_table(sessions, internal_flag="first_visit")
    assert default.full_visitor_id.tolist() == ["b", "c", "a"]
    assert rm.full_visitor_id.tolist() == ["b", "staff", "c", "a"]
    assert list(rm.columns) == list(default.columns) + ["is_internal"]
    shared = rm[~rm.is_internal].drop(columns="is_internal").reset_index(drop=True)
    pd.testing.assert_frame_equal(shared, default)


def test_unknown_internal_flag_is_an_error(session):
    with pytest.raises(ValueError, match="internal_flag"):
        remarketing_table(pd.DataFrame([session("v", T0)]), internal_flag="session")

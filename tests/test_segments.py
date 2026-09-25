"""Tests for src/segments.py: who counts as a key account, and the segment each session belongs to."""

from __future__ import annotations

import pandas as pd

from src import segments
from src.segments import EMPLOYEES, EXTERNAL, KEY_ACCOUNT, KEY_ACCOUNTS, is_key_account, segment

ACCOUNT = "1957458976293878100"


def test_the_key_account_found_by_the_concentration_check():
    assert KEY_ACCOUNTS == {ACCOUNT}


def test_is_key_account_compares_visitor_ids_exactly():
    ids = pd.Series([ACCOUNT, "1957458976293878101", "957458976293878100", ""])
    assert is_key_account(ids).tolist() == [True, False, False, False]


def test_each_session_gets_one_segment(session):
    df = pd.DataFrame([
        session("v", "2017-01-10 12:00"),
        session("staff", "2017-01-10 13:00", internal=True),
        session(ACCOUNT, "2017-01-10 14:00", purchased=True, revenue=17_859.50),
        session("v", "2017-01-11 12:00", visit_number=2),
    ])
    assert segment(df).tolist() == [EXTERNAL, EMPLOYEES, KEY_ACCOUNT, EXTERNAL]
    assert segment(df).index.equals(df.index)


def test_employees_take_precedence_over_key_accounts(session, monkeypatch):
    # an employee's id added to the key accounts by mistake stays employee traffic
    monkeypatch.setattr(segments, "KEY_ACCOUNTS", frozenset({"staff"}))
    df = pd.DataFrame([session("staff", "2017-01-10 13:00", internal=True)])
    assert segment(df).tolist() == [EMPLOYEES]

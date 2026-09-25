"""Shared builders for the synthetic test data: no BigQuery, no network, no data/ caches."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest


def _session_row(visitor: str, start: str | int, *, visit_number: int = 1, visit_id: int | None = None,
                 purchased: bool = False, revenue: float | None = None, internal: bool = False,
                 **overrides) -> dict:
    """One row of the clean session table (sql/process/01_sessions_clean.sql) with plausible defaults.

    `start` is a UTC timestamp string or Unix seconds, and session_date is its UTC calendar day. A purchase
    has $25 of revenue unless given and reaches the purchase step (6). Any column can be overridden.
    """
    start = int(pd.Timestamp(start, tz="UTC").timestamp()) if isinstance(start, str) else start
    visit_id = start if visit_id is None else visit_id
    row = {
        "session_key": f"{visitor}-{visit_id}-{start}",
        "full_visitor_id": visitor,
        "visit_id": visit_id,
        "visit_start_time": start,
        "session_start": pd.Timestamp(start, unit="s", tz="UTC"),
        "session_date": pd.Timestamp(start, unit="s").normalize(),
        "visit_number": visit_number,
        "is_new_visit": visit_number == 1,
        "channel": "Organic Search",
        "source": "google",
        "medium": "organic",
        "is_true_direct": False,
        "is_internal_entry": False,
        "device_category": "desktop",
        "operating_system": "Macintosh",
        "browser": "Chrome",
        "continent": "Americas",
        "sub_continent": "Northern America",
        "country": "United States",
        "hits": 6,
        "pageviews": 4,
        "time_on_site": 120,
        "bounced": False,
        "product_detail_views": 1,
        "distinct_products_viewed": 1,
        "add_to_cart_events": int(purchased),
        "checkout_events": int(purchased),
        "max_ecommerce_step": 6 if purchased else 2,
        "transactions": int(purchased),
        "revenue_usd": (25.0 if purchased else 0.0) if revenue is None else revenue,
        "purchased": purchased,
        "is_internal": internal,
    }
    row.update(overrides)
    return row


@pytest.fixture
def session() -> Callable[..., dict]:
    """Factory for session rows, e.g. session("v1", "2017-01-10 12:00", visit_number=2, purchased=True)."""
    return _session_row

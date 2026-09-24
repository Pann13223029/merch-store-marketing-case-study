"""Validation checks for the clean session table (sql/process/01_sessions_clean.sql).

Each check compares the cleaned extract against totals measured on the raw BigQuery
tables during Prepare (reports/02_prepare.md), so cleaning can't silently drop or
double-count sessions, purchases or revenue.
"""

from __future__ import annotations

import pandas as pd

# Totals measured directly on the raw tables in Prepare.
RAW = {
    "sessions": 903_653,
    "purchase_sessions": 11_552,
    "revenue_usd": 1_780_149,
    "internal_visitors": 37_332,
    "internal_sessions": 76_536,
    "internal_purchases": 5_408,
    "internal_revenue_usd": 730_377,
}

ENGAGEMENT_AND_OUTCOME = [
    "hits", "pageviews", "time_on_site", "product_detail_views",
    "add_to_cart_events", "checkout_events", "transactions", "revenue_usd",
]


def validate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Run all checks and return one row per check: status, check, observed."""
    rows = []

    def check(name: str, passed: bool, observed: object = "") -> None:
        rows.append({"status": "PASS" if passed else "FAIL", "check": name, "observed": str(observed)})

    purchases = df[df.purchased]
    internal = df[df.is_internal]
    internal_visitors = df.groupby("full_visitor_id").is_internal.first().sum()

    check("Row count matches raw", len(df) == RAW["sessions"], f"{len(df):,}")
    check("session_key is unique", df.session_key.is_unique, f"{df.session_key.nunique():,} unique")
    check("Purchase sessions match raw", len(purchases) == RAW["purchase_sessions"], f"{len(purchases):,}")
    check("Revenue matches raw", round(df.revenue_usd.sum()) == RAW["revenue_usd"], f"${df.revenue_usd.sum():,.0f}")
    check("Internal visitors match Prepare", internal_visitors == RAW["internal_visitors"], f"{internal_visitors:,}")
    # Revenue within $1: the exact value is $730,376.50, which BigQuery rounds half-up and
    # Python rounds half-to-even.
    check(
        "Internal sessions / purchases / revenue match Prepare",
        len(internal) == RAW["internal_sessions"]
        and internal.purchased.sum() == RAW["internal_purchases"]
        and abs(internal.revenue_usd.sum() - RAW["internal_revenue_usd"]) <= 1,
        f"{len(internal):,} / {internal.purchased.sum():,} / ${internal.revenue_usd.sum():,.2f}",
    )
    check(
        "Internal flag is constant within each visitor",
        df.groupby("full_visitor_id").is_internal.nunique().max() == 1,
    )
    check("No NULLs in engagement / outcome columns", df[ENGAGEMENT_AND_OUTCOME].isna().sum().sum() == 0)
    check("No negative values", (df[ENGAGEMENT_AND_OUTCOME] < 0).sum().sum() == 0)
    check(
        "Date range is 2016-08-01 to 2017-08-01",
        (df.session_date.min().date().isoformat(), df.session_date.max().date().isoformat())
        == ("2016-08-01", "2017-08-01"),
        f"{df.session_date.min().date()} to {df.session_date.max().date()}",
    )
    mismatched_new = (df.is_new_visit != (df.visit_number == 1)).sum()
    check("is_new_visit agrees with visit_number == 1", mismatched_new == 0, f"{mismatched_new:,} mismatches")
    # GA quirk: a pageview plus a non-interaction event is still a bounce but records a duration
    # (2 of ~450k bounces), so allow a negligible share of exceptions.
    bounced = df[df.bounced]
    odd_bounces = ((bounced.pageviews > 1) | (bounced.time_on_site > 0)).sum()
    check(
        "Bounced sessions have <= 1 pageview and 0 time on site (< 0.01% exceptions)",
        odd_bounces / len(bounced) < 0.0001,
        f"{odd_bounces} of {len(bounced):,} bounces",
    )
    reached = (purchases.max_ecommerce_step >= 6).mean()
    check("Purchase sessions reach the purchase step in hit data (>95%)", reached > 0.95, f"{reached:.1%}")
    check("Revenue > 0 only on purchase sessions", (df.loc[~df.purchased, "revenue_usd"] == 0).all())
    no_rev = (purchases.revenue_usd == 0).sum()
    check("Purchase sessions without revenue stay negligible (< 60)", no_rev < 60, f"{no_rev} sessions")

    return pd.DataFrame(rows)

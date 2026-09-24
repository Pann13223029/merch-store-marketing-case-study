"""Analysis tables built from the clean session table (data/raw/sessions_clean.parquet).

- remarketing_table():  corrected D2 table: external first-visit non-buyers, label = purchase on a
                        later visit within 30 days, only first visits whose 30-day window fits in the data.

The audit of Google's GSP229 lab runs on the lab's own table instead (sql/audit/a09-a13).
"""

from __future__ import annotations

import pandas as pd

WINDOW_DAYS = 30
TRAIN_END = pd.Timestamp("2017-04-30")   # lab: train on first visits 2016-08-01..2017-04-30
TEST_START = pd.Timestamp("2017-05-01")  # lab: evaluate on first visits 2017-05-01..2017-06-30
TEST_END = pd.Timestamp("2017-06-30")

FIRST_VISIT_FEATURES = [
    # acquisition
    "channel", "medium", "source",
    # technology and geography
    "device_category", "operating_system", "browser", "sub_continent", "country",
    # engagement
    "hits", "pageviews", "time_on_site", "bounced",
    # shopping behaviour
    "product_detail_views", "distinct_products_viewed", "add_to_cart_events",
    "checkout_events", "max_ecommerce_step",
]


def _split(dates: pd.Series) -> pd.Series:
    return pd.Series(
        pd.cut(
            dates,
            bins=[pd.Timestamp("2016-07-31"), TRAIN_END, TEST_END, pd.Timestamp("2017-08-02")],
            labels=["train", "eval", "predict"],
        ).astype(str),
        index=dates.index,
    )


def remarketing_table(sessions: pd.DataFrame) -> pd.DataFrame:
    """Corrected D2 table: one row per external visitor whose first visit is in the data.

    Population: first visit (visit_number = 1) did not purchase, and the first visit starts early
    enough (on or before 2017-07-01) that its full 30-day follow-up lies inside the data.
    Label: any purchase in a LATER visit starting within 30 days of the first visit's start.
    """
    ext = sessions[~sessions.is_internal].sort_values(["full_visitor_id", "visit_start_time"])

    # A first visit that crosses midnight appears as two rows with the same visit_id; merge them.
    first_rows = ext[ext.visit_number == 1]
    first = first_rows.groupby(["full_visitor_id", "visit_id"], as_index=False).agg(
        first_start=("visit_start_time", "min"),
        session_date=("session_date", "min"),
        channel=("channel", "first"),
        medium=("medium", "first"),
        source=("source", "first"),
        device_category=("device_category", "first"),
        operating_system=("operating_system", "first"),
        browser=("browser", "first"),
        sub_continent=("sub_continent", "first"),
        country=("country", "first"),
        hits=("hits", "sum"),
        pageviews=("pageviews", "sum"),
        time_on_site=("time_on_site", "sum"),
        bounced=("bounced", "min"),
        product_detail_views=("product_detail_views", "sum"),
        distinct_products_viewed=("distinct_products_viewed", "max"),
        add_to_cart_events=("add_to_cart_events", "sum"),
        checkout_events=("checkout_events", "sum"),
        max_ecommerce_step=("max_ecommerce_step", "max"),
        purchased_first_visit=("purchased", "max"),
    )
    # 608 external visitors (0.09%) have two distinct visit_number = 1 visits; keep the earliest.
    first = first.sort_values("first_start").drop_duplicates("full_visitor_id", keep="first")

    later = ext.merge(first[["full_visitor_id", "visit_id", "first_start"]], on="full_visitor_id",
                      suffixes=("", "_first"))
    later = later[
        (later.visit_id != later.visit_id_first)
        & (later.visit_start_time > later.first_start)
        & (later.visit_start_time <= later.first_start + WINDOW_DAYS * 86400)
    ]
    outcome = later.groupby("full_visitor_id").agg(
        buy_within_30d=("purchased", "max"),
        revenue_30d_usd=("revenue_usd", "sum"),
        return_visits_30d=("visit_id", "nunique"),
    )

    t = first.merge(outcome, on="full_visitor_id", how="left")
    t["buy_within_30d"] = t.buy_within_30d.fillna(False).astype(int)
    t["revenue_30d_usd"] = t.revenue_30d_usd.fillna(0.0)
    t["return_visits_30d"] = t.return_visits_30d.fillna(0).astype(int)

    t = t[~t.purchased_first_visit & (t.session_date <= pd.Timestamp("2017-07-01"))].copy()
    t["bounced"] = t.bounced.astype(int)
    t["split"] = _split(t.session_date).replace({"predict": "eval"})  # 2017-07-01 joins the test month
    t.loc[t.split == "eval", "split"] = "test"
    return t.drop(columns=["purchased_first_visit"]).reset_index(drop=True)

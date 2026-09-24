"""Customer journeys for multi-touch attribution (D1), built from external sessions.

Journey rules (reports/03_process.md):
  * A visitor's sessions form one journey until a purchase or a gap of more than 30 days of
    inactivity; the next session then starts a new journey.
  * Converting journeys keep only touches within the 30-day lookback before the purchase.
  * Analysis period: journeys ending 2016-08-31 .. 2017-07-01, so every converting journey has its
    full 30-day lookback inside the data and every non-converting journey has 30 days of observed
    silence after its last touch.

Three channel labels per touch:
  * arrival_channel       what actually brought the visitor: isTrueDirect visits become "Direct".
  * conservative_channel  only visits whose source is literally "(direct)" become "Direct" (sensitivity).
  * ga_channel            GA's channelGrouping, which credits direct returns to the previous campaign.
"""

from __future__ import annotations

import pandas as pd

WINDOW_S = 30 * 86400
PERIOD_START = pd.Timestamp("2016-08-31")
PERIOD_END = pd.Timestamp("2017-07-01")


def touches(sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per touch (session) in the analysis period, tagged with its journey."""
    t = (
        sessions.loc[~sessions.is_internal, [
            "full_visitor_id", "visit_start_time", "session_date", "channel", "source",
            "is_true_direct", "purchased", "revenue_usd",
        ]]
        .sort_values(["full_visitor_id", "visit_start_time"])
        .reset_index(drop=True)
    )
    t["arrival_channel"] = t.channel.where(~t.is_true_direct, "Direct")
    t["conservative_channel"] = t.channel.where(~(t.is_true_direct & (t.source == "(direct)")), "Direct")
    t = t.rename(columns={"channel": "ga_channel"})

    same_visitor = t.full_visitor_id.eq(t.full_visitor_id.shift())
    gap = t.visit_start_time - t.visit_start_time.shift()
    prev_purchased = t.purchased.shift(fill_value=False).astype(bool)
    new_journey = ~same_visitor | (gap > WINDOW_S) | prev_purchased
    t["journey_id"] = new_journey.cumsum()

    # Journey-level end facts, then the 30-day lookback trim on converting journeys.
    last = t.groupby("journey_id").agg(end_time=("visit_start_time", "max"), end_date=("session_date", "max"),
                                       converted=("purchased", "max"))
    t = t.join(last, on="journey_id")
    t = t[~t.converted | (t.end_time - t.visit_start_time <= WINDOW_S)]
    t = t[(t.end_date >= PERIOD_START) & (t.end_date <= PERIOD_END)]
    t["touch_index"] = t.groupby("journey_id").cumcount() + 1
    return t.drop(columns=["end_time"]).reset_index(drop=True)


def journeys(touch_df: pd.DataFrame, revenue_cap_usd: float) -> pd.DataFrame:
    """One row per journey with its three channel paths, outcome, and capped revenue."""
    g = touch_df.groupby("journey_id", sort=False)
    j = g.agg(
        full_visitor_id=("full_visitor_id", "first"),
        start_date=("session_date", "min"),
        end_date=("session_date", "max"),
        n_touches=("touch_index", "max"),
        converted=("converted", "first"),
        revenue_usd=("revenue_usd", "sum"),
        ga_last_channel=("ga_channel", "last"),
    )
    for col in ["arrival_channel", "conservative_channel", "ga_channel"]:
        j[col.replace("_channel", "_path")] = g[col].agg(" > ".join)
    j["revenue_capped_usd"] = j.revenue_usd.clip(upper=revenue_cap_usd)
    return j.reset_index()

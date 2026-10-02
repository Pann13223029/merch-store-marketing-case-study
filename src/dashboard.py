"""Export small, dashboard-ready CSV files for the Looker Studio dashboard (dashboard/data/).

Every file is an aggregate (no visitor-level rows), rebuilt from the cached pipeline outputs.
Marketing figures cover outside visitors only: Google employees and key accounts (src/segments.py) appear only
as their own segments in monthly_channel.csv.
Run notebooks 01-05 first, then:  python -m src.dashboard
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.attribution import MODELS, Attribution
from src.breakeven import months_spanned, value_by_band
from src.journeys import revenue_cap
from src.modeling import LABEL, PIPELINES, add_features
from src.segments import EXTERNAL, is_key_account, segment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "data"
MARKOV_ORDER = 3
MODEL_LABELS = {
    "ga_last_click": "GA report (last non-direct click)",
    "last_touch": "Last touch",
    "first_touch": "First touch",
    "linear": "Linear",
    "position_based": "Position-based",
    "markov": "Markov (data-driven)",
    "markov_conservative": "Markov, conservative relabel",
}


def monthly_channel(sessions: pd.DataFrame) -> pd.DataFrame:
    """Sessions, purchases and revenue by month, channel (GA's label) and segment.

    Segments: External, Internal (Google employees), and Key account. Capped revenue caps each purchase
    session at revenue_cap(sessions) (D-P2, $1,605.91), as in notebooks 02 and 05.
    """
    cap = revenue_cap(sessions)
    s = sessions.assign(
        month=sessions.session_date.dt.strftime("%Y-%m-01"),   # a full date, which Looker Studio types as Date
        segment=segment(sessions),
        revenue_capped_usd=sessions.revenue_usd.clip(upper=cap),
    )
    s = s[s.month <= "2017-07-01"]   # 2017-08 holds a single day
    return (s.groupby(["month", "channel", "segment"], as_index=False)
            .agg(sessions=("session_key", "size"), purchases=("purchased", "sum"),
                 revenue_usd=("revenue_usd", "sum"), revenue_capped_usd=("revenue_capped_usd", "sum"))
            .round(2))


def channel_profile(sessions: pd.DataFrame) -> pd.DataFrame:
    """External traffic by channel as GA labels it (the store's own channel report), without key accounts.

    Capped revenue caps each purchase session at revenue_cap(sessions) (D-P2), computed on the full table.
    """
    cap = revenue_cap(sessions)
    ext = sessions[segment(sessions) == EXTERNAL]
    p = (ext.assign(revenue_capped_usd=ext.revenue_usd.clip(upper=cap))
         .groupby("channel", as_index=False)
         .agg(sessions=("session_key", "size"), visitors=("full_visitor_id", "nunique"),
              purchases=("purchased", "sum"), revenue_usd=("revenue_usd", "sum"),
              revenue_capped_usd=("revenue_capped_usd", "sum")))
    p["conversion_rate"] = p.purchases / p.sessions
    return p.sort_values("purchases", ascending=False).round(4)


def attribution_tables(journeys: pd.DataFrame, touches: pd.DataFrame, n_boot: int = 1000,
                       seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(credit by channel and model with 95% CIs, value per click for paid channels).

    The intervals come from a bootstrap that resamples visitors, all of a visitor's journeys together.
    """
    att = Attribution(journeys, "arrival_path", order=MARKOV_ORDER)
    table = att.credit_table()
    shares = att.shares()
    boot = att.bootstrap_shares(n_boot=n_boot, seed=seed, clusters=journeys.full_visitor_id)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    rows = []
    for k, m in enumerate(MODELS):
        for i, c in enumerate(att.channels):
            rows.append({
                "channel": c, "model": MODEL_LABELS[m],
                "purchases_credited": table.loc[c, (m, "conversions")],
                "purchase_share": shares.loc[c, m],
                "purchase_share_ci_low": low[i, k], "purchase_share_ci_high": high[i, k],
                "revenue_credited_usd": table.loc[c, (m, "revenue")],
            })
    credit = pd.DataFrame(rows)
    credit["revenue_share"] = credit.revenue_credited_usd / credit.groupby("model").revenue_credited_usd.transform("sum")

    revenue = table.xs("revenue", axis=1, level="measure")[MODELS].copy()
    revenue["markov_conservative"] = Attribution(journeys, "conservative_path", order=MARKOV_ORDER) \
        .credit_table()[("markov", "revenue")]
    clicks = touches.arrival_channel.value_counts()
    paid = ["Paid Search", "Display", "Affiliates"]
    value = (revenue.loc[paid].div(clicks.loc[paid], axis=0).rename(columns=MODEL_LABELS)
             .rename_axis("channel").reset_index()
             .melt(id_vars="channel", var_name="model", value_name="value_per_click_usd"))
    value["clicks"] = value.channel.map(clicks)
    return credit.round(5), value.round(4)


def retargeting_bands(remarketing: pd.DataFrame, tuning: pd.DataFrame) -> pd.DataFrame:
    """Test-month value of first-time visitors by score band, from the model chosen in tuning."""
    rm = add_features(remarketing)
    train, test = rm[rm.split == "train"], rm[rm.split == "test"]
    best = tuning.loc[tuning.cv_pr_auc.idxmax()]
    model = PIPELINES[best.model](**best.params).fit(train, train[LABEL])
    scores = model.predict_proba(test)[:, 1]
    bands = value_by_band(test[LABEL].to_numpy(), test.revenue_30d_usd.to_numpy(), scores)
    months = months_spanned(test.session_date)
    bands.insert(1, "band_order", range(1, len(bands) + 1))
    bands["visitors_per_month"] = bands.visitors / months
    bands["model"] = best.model
    return bands.round(4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = pd.read_parquet(ROOT / "data/raw/sessions_clean.parquet")
    journeys = pd.read_parquet(ROOT / "data/processed/journeys.parquet")
    touches = pd.read_parquet(ROOT / "data/processed/touches.parquet")
    # src/journeys.py leaves key accounts out; dropping them here also covers tables built before that rule
    journeys = journeys[~is_key_account(journeys.full_visitor_id)].reset_index(drop=True)
    touches = touches[~is_key_account(touches.full_visitor_id)].reset_index(drop=True)
    remarketing = pd.read_parquet(ROOT / "data/processed/remarketing_table.parquet")
    tuning = pd.read_json(ROOT / "data/processed/tuning_results.json")

    outputs = {
        "monthly_channel.csv": monthly_channel(sessions),
        "channel_profile.csv": channel_profile(sessions),
        "retargeting_bands.csv": retargeting_bands(remarketing, tuning),
    }
    credit, value = attribution_tables(journeys, touches)
    outputs["attribution_credit.csv"] = credit
    outputs["value_per_click.csv"] = value
    for name, df in outputs.items():
        df.to_csv(OUT / name, index=False)
        print(f"{name:26s} {len(df):>4} rows")


if __name__ == "__main__":
    main()

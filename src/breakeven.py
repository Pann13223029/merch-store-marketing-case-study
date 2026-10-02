"""Break-even for remarketing (D2): what one remarketed first-time visitor is worth, by score band.

Remarketing a visitor pays off when

    revenue per visitor (next 30 days) x incremental lift x gross margin  >=  cost per visitor reached

Revenue per visitor is measured, not modeled: it is what visitors in each score band actually
spent in the 30 days after their first visit, in the out-of-time test months. Each visitor's 30-day
total is capped at D-P2's $1,606 per-purchase-session threshold; capping per visitor is stricter than
D-P2's per-session rule and lowers the top-10%/20% value by about 3-4%. Lift and margin are
assumptions with a sensitivity grid.

Lift evidence: a randomized display-retargeting campaign lifted purchases 10.5% (Johnson, Lewis &
Nubbemeyer 2017, "Ghost Ads", JMR); the median conversion lift (purchases, sign-ups or similar
advertiser-defined actions) was 8% across the 184 of 432 Google Display Network experiments that
measured conversions (Johnson, Lewis & Nubbemeyer, "The Online Display Ad Effectiveness Funnel &
Carryover", SSRN 2701578). Both lifts were measured on people who actually saw an ad (ghost-ad
experiments estimate the effect on exposed users).

Scope: value_of_targets applies that lift to every visitor in the target, so its monthly values are
upper bounds that assume the ads reach everyone in the audience. The break-even cost
(max_affordable_cost) is per visitor actually reached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = [(0.00, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.00)]
TARGETS = (0.01, 0.05, 0.10, 0.20)   # retarget the top X% of scored first visits
CENTRAL = {"lift": 0.10, "margin": 0.50}
LIFTS = (0.05, 0.08, 0.10, 0.15, 0.20)
MARGINS = (0.30, 0.50, 0.70)
# D-P2's per-purchase-session 99th percentile, applied here to each visitor's 30-day total (stricter)
REVENUE_CAP_USD = 1606


def months_spanned(dates) -> float:
    """Length in months (30.4 days each) of the window from the earliest to the latest date, both included.

    The count is inclusive: first visits from May 1 to Jul 1, 2017 cover 62 days, not the 61 that
    (max - min).days gives, so the divisor is ((max - min).days + 1) / 30.4.
    """
    dates = pd.to_datetime(pd.Series(dates))
    if dates.isna().all():
        raise ValueError("months_spanned needs at least one date")
    return ((dates.max().normalize() - dates.min().normalize()).days + 1) / 30.4


def _band_label(lo: float, hi: float) -> str:
    return f"Top {hi:.0%}" if lo == 0 else f"{lo:.0%}–{hi:.0%}"


def value_by_band(y: np.ndarray, revenue: np.ndarray, score: np.ndarray, bands=BANDS,
                  n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Observed 30-day buy rate and revenue per visitor for each score band (with bootstrap CIs)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(-score, kind="stable")
    n = len(score)
    rev = np.minimum(revenue, REVENUE_CAP_USD)
    rows = []
    for lo, hi in bands:
        idx = order[int(round(lo * n)):int(round(hi * n))]
        if len(idx) == 0:
            raise ValueError(f"band {_band_label(lo, hi)} is empty for n={n}; banding needs more visitors")
        r = rev[idx]
        boots = np.array([r[rng.integers(0, len(r), len(r))].mean() for _ in range(n_boot)])
        rows.append({
            "band": _band_label(lo, hi),
            "visitors": len(idx),
            "buyers": int(y[idx].sum()),
            "buy_rate": float(y[idx].mean()),
            "revenue_per_visitor": float(r.mean()),
            "rpv_ci_low": float(np.percentile(boots, 2.5)),
            "rpv_ci_high": float(np.percentile(boots, 97.5)),
        })
    return pd.DataFrame(rows)


def max_affordable_cost(revenue_per_visitor, lift: float = CENTRAL["lift"], margin: float = CENTRAL["margin"]):
    """Highest cost per visitor actually reached (30 days) at which the band still breaks even."""
    return np.asarray(revenue_per_visitor) * lift * margin


def breakeven_lift(cost_per_visitor, revenue_per_visitor, margin: float = CENTRAL["margin"]):
    """Smallest incremental lift at which remarketing a visitor pays for its cost (the inverse of max_affordable_cost)."""
    return np.asarray(cost_per_visitor) / (np.asarray(revenue_per_visitor) * margin)


def value_of_targets(y: np.ndarray, revenue: np.ndarray, score: np.ndarray, months: float, targets=TARGETS,
                     zero_lift: np.ndarray | None = None, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Extra gross profit per month, before ad cost, from remarketing the top share of visitors by score.

    An upper bound: the lift applies to every visitor in the target, as if the ads reach all of them.
    `months` is the length of the scored window, e.g. months_spanned(test.session_date).

    Value = capped 30-day revenue of the visitors in the target x lift x margin / months, in the central case
    (`gross_profit_per_month`), with a bootstrap 95% CI over the visitors in the target (revenue noise only) and
    the range across the LIFTS x MARGINS grid (`range_low`, `range_high`). Visitors in `zero_lift` (e.g. Google
    employees, whom an ad can't turn into customers) add no revenue but still count as remarketed visitors.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(-score, kind="stable")
    rev = np.minimum(revenue, REVENUE_CAP_USD)
    if zero_lift is not None:
        rev = np.where(zero_lift, 0.0, rev)
    if months <= 0:
        raise ValueError(f"months must be positive, not {months}")
    if np.sum(y) == 0:
        raise ValueError("y has no positives; share_of_later_buyers is undefined")
    central = CENTRAL["lift"] * CENTRAL["margin"]
    rows = []
    for frac in targets:
        idx = order[:int(round(frac * len(score)))]
        if len(idx) == 0:
            raise ValueError(f"target top {frac:.0%} is empty for n={len(score)}")
        r = rev[idx]
        boots = np.array([r[rng.integers(0, len(r), len(r))].sum() for _ in range(n_boot)])
        rows.append({
            "target": f"top {frac:.0%}",
            "visitors_per_month": len(idx) / months,
            "share_of_later_buyers": y[idx].sum() / y.sum(),
            "revenue_per_visitor": r.mean(),
            "gross_profit_per_month": r.sum() * central / months,
            "ci_low": np.percentile(boots, 2.5) * central / months,
            "ci_high": np.percentile(boots, 97.5) * central / months,
            "range_low": r.sum() * min(LIFTS) * min(MARGINS) / months,
            "range_high": r.sum() * max(LIFTS) * max(MARGINS) / months,
        })
    return pd.DataFrame(rows)


def sensitivity(bands: pd.DataFrame, lifts=LIFTS, margins=MARGINS) -> pd.DataFrame:
    """Max affordable cost per visitor for every band x (lift, margin) scenario."""
    cols = {}
    for m in margins:
        for l in lifts:
            cols[(f"margin {m:.0%}", f"lift {l:.0%}")] = max_affordable_cost(bands.revenue_per_visitor, l, m)
    out = pd.DataFrame(cols, index=bands.band)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out

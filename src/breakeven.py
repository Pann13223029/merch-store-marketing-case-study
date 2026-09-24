"""Break-even for remarketing (D2): what one remarketed first-time visitor is worth, by score band.

Remarketing a visitor pays off when

    revenue per visitor (next 30 days) x incremental lift x gross margin  >=  cost per visitor

Revenue per visitor is measured, not modelled: it is what visitors in each score band actually
spent in the 30 days after their first visit, in the out-of-time test months (capped per
visitor, as in D-P2). Lift and margin are assumptions with a sensitivity grid.

Lift evidence: a randomized display-retargeting campaign lifted purchases 10.5% (Johnson, Lewis &
Nubbemeyer 2017, "Ghost Ads", JMR); the median conversion lift across 432 Google Display Network
experiments was 8% (Johnson, Lewis & Nubbemeyer, "The Online Display Ad Effectiveness Funnel &
Carryover", SSRN 2701578).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = [(0.00, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.00)]
CENTRAL = {"lift": 0.10, "margin": 0.50}
LIFTS = (0.05, 0.08, 0.10, 0.15, 0.20)
MARGINS = (0.30, 0.50, 0.70)
REVENUE_CAP_USD = 1606  # per visitor, the 99th-percentile order cap from D-P2


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
    """Highest cost per remarketed visitor (30 days) at which the band still breaks even."""
    return np.asarray(revenue_per_visitor) * lift * margin


def sensitivity(bands: pd.DataFrame, lifts=LIFTS, margins=MARGINS) -> pd.DataFrame:
    """Max affordable cost per visitor for every band x (lift, margin) scenario."""
    cols = {}
    for m in margins:
        for l in lifts:
            cols[(f"margin {m:.0%}", f"lift {l:.0%}")] = max_affordable_cost(bands.revenue_per_visitor, l, m)
    out = pd.DataFrame(cols, index=bands.band)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out

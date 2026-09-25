"""Build kaggle/merch_store_marketing_case_study.ipynb from this repository.

The helper cells are the src/ modules, copied in unchanged, so the Kaggle notebook runs the same code
as the full project. Rebuild after changing src/:  python kaggle/build_notebook.py
"""
import re
import sys
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

def module_source(name: str) -> str:
    """Repo module text with its intra-repo imports removed (everything shares one namespace here)."""
    src = (ROOT / "src" / f"{name}.py").read_text()
    src = re.sub(r"^from src(\.\w+)? import .*\n", "", src, flags=re.M)
    src = src.replace("from __future__ import annotations\n", "")
    return f"# ---- src/{name}.py (copied unchanged from the repository, minus its imports of other repo modules)\n" + src

md("""# Where should the Google Merchandise Store spend its next marketing dollar?

A marketing-analytics case study on the **Google Analytics 360 export of the Google Merchandise Store** (BigQuery public dataset, Aug 2016 – Aug 2017). It's a condensed, runnable version of the full project: **[github.com/Pann13223029/merch-store-marketing-case-study](https://github.com/Pann13223029/merch-store-marketing-case-study)** (write-ups, 5 notebooks, SQL, charts, executive summary).

**The stakeholder:** the store's Head of Marketing.
- **D1:** which channels actually produce buyers?
- **D2:** which first-time visitors are worth paying to bring back?

**What this notebook shows:**
1. **About 41% of the store's revenue comes from Google employees**, arriving through a hidden internal referral. It has to be removed before any marketing analysis.
2. **A retargeting model** trained on outside visitors only: the top 10% of first-time visitors include 70% of later buyers, but the program is worth only about $700 a month.
3. **Attribution:** the store's GA report over-credits Organic Search by about 15 points of purchases, and hides that returning visitors bring about half of all purchases.

The repository also audits **Google's own BigQuery ML tutorial model** for this question. Its top 1% of "likely buyers" are 98.8% Google employees. That audit needs your own Google Cloud project, so it's summarized at the end rather than re-run here.""")

code('''import logging, warnings
import numpy as np
import pandas as pd
from google.cloud import bigquery

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
N_BOOT = 1000''')

md("""## Helper code

These cells are the repository's `src/` modules, copied unchanged (apart from imports between them) so every result here comes from the same code as the full project.""")
for name in ["validate", "tables", "journeys", "modeling", "breakeven", "attribution", "viz"]:
    code(module_source(name))

md("""## 1. One clean row per session (BigQuery SQL)

The query unnests each session's hits to count funnel steps (product views, add-to-cart, checkout). It flags employee traffic at the **visitor** level: anyone who ever arrived through the hidden internal referral (`Referral`, source `(direct)`, path `/`). It scans about 0.8 GB.""")
sql = (ROOT / "sql/process/01_sessions_clean.sql").read_text()
code(f'''SESSIONS_SQL = """
{sql}"""

client = bigquery.Client()
job = client.query(SESSIONS_SQL)
try:
    sessions = job.to_dataframe()                                 # BigQuery Storage API (fast)
except Exception:
    sessions = job.to_dataframe(create_bqstorage_client=False)   # plain REST download (slower)
sessions["session_date"] = pd.to_datetime(sessions.session_date)
checks = validate_sessions(sessions)
print(f"{{len(sessions):,}} sessions; {{(checks.status == 'PASS').sum()}} of {{len(checks)}} validation checks pass")
checks''')

md("""## 2. About 41% of revenue is Google employees

In the public data, their visits look like `Referral` traffic with the source hidden. Most come from Google office cities. Google's training copy of this data shows the real source, `mall.googleplex.com` (the employee store link), and against it this flag is 99.2% precise and catches 98.2% of employee purchases (see the repository). **Every analysis below uses outside visitors only.**""")
code('''seg = sessions.assign(segment=np.where(sessions.is_internal, "Internal (Google employees)", "External"))
summary = seg.groupby("segment").agg(visitors=("full_visitor_id", "nunique"), sessions=("session_key", "size"),
                                     purchase_sessions=("purchased", "sum"), revenue_usd=("revenue_usd", "sum"))
summary.assign(share_of_revenue=summary.revenue_usd / summary.revenue_usd.sum()).style.format(
    {"visitors": "{:,}", "sessions": "{:,}", "purchase_sessions": "{:,}", "revenue_usd": "${:,.0f}", "share_of_revenue": "{:.1%}"})''')

md("""## 3. D2: which first-time visitors are worth bringing back?

- **Population:** outside visitors whose first visit didn't end in a purchase.
- **Label:** a purchase on a later visit within 30 days.
- **Train/test split:** train on first visits Aug 2016 – Apr 2017, test once on May – Jul 2017.""")
code('''rm = add_features(remarketing_table(sessions))
train, test = rm[rm.split == "train"], rm[rm.split == "test"]
rm.groupby("split").agg(visitors=(LABEL, "size"), later_buyers=(LABEL, "sum"), rate=(LABEL, "mean")).style.format({"rate": "{:.3%}"})''')
code('''from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint
h1 = train[train.channel != "(Other)"]
ct = pd.crosstab(h1.channel, h1[LABEL])
chi2, p, dof, _ = chi2_contingency(ct)
g = pd.DataFrame({"n": ct.sum(axis=1), "k": ct[1]}); g["rate"] = g.k / g.n
g["lo"], g["hi"] = proportion_confint(g.k, g.n, method="wilson"); g = g.sort_values("rate", ascending=False)
rate_with_ci_chart(list(g.index), g.rate.tolist(), g.lo.tolist(), g.hi.tolist(),
    title="Where a first visit comes from predicts whether the visitor comes back to buy",
    subtitle="Share of outside first-time visitors (who didn't buy) that purchase on a later visit within 30 days, with 95% CI",
    note=f"First visits Aug 2016 – Apr 2017 (n = {g.n.sum():,}). Chi-square: χ² = {chi2:,.0f}, df = {dof}, p < 0.001. Google employees excluded.");''')

md("""**The model:** a random forest with the settings that won time-based tuning in the repository (28 configurations; minimum leaf size 10, √features). It's compared with a simple funnel rule and with the feature set of Google's own lab, refit on this corrected data.""")
code('''scores = {"Funnel rule": funnel_rule_score(test),
          "Lab's features, refit": lab_features_pipeline().fit(train, train[LABEL]).predict_proba(test)[:, 1],
          "Our model (random forest)": random_forest_pipeline(min_samples_leaf=10, max_features="sqrt")
                                           .fit(train, train[LABEL]).predict_proba(test)[:, 1]}
y = test[LABEL].to_numpy()
pd.DataFrame({k: ranking_metrics(y, s) for k, s in scores.items()}).T.style.format(
    {"pr_auc": "{:.3f}", "roc_auc": "{:.3f}", "precision_top_1pct": "{:.1%}", "lift_top_1pct": "{:.1f}x",
     "lift_top_5pct": "{:.1f}x", "lift_top_10pct": "{:.1f}x", "recall_top_10pct": "{:.0%}"})''')
code('''colors = {"Our model (random forest)": SERIES_1, "Lab's features, refit": "#eb6834", "Funnel rule": "#1baf7a"}
gains_chart({k: cumulative_gains(y, scores[k]) for k in colors}, colors,
    title="Retargeting the top 10% of first-time visitors reaches 70% of future buyers",
    subtitle="Share of later buyers captured by targeting the highest-scored first visits",
    note=f"Out-of-time test: first visits May 1 – Jul 1, 2017 ({len(test):,} outside visitors, {int(y.sum())} later buyers).");''')

md("""**Break-even:** is a visitor worth retargeting? Only if revenue per visitor over the next 30 days × incremental lift × gross margin is at least the cost per visitor. Revenue per visitor is measured on the test months. Lift is 10%, anchored on randomized experiments (8–10.5%), and margin is assumed at 50%.""")
code('''bands = value_by_band(y, test.revenue_30d_usd.to_numpy(), scores["Our model (random forest)"])
bands["max_affordable_cost_usd"] = max_affordable_cost(bands.revenue_per_visitor)
bands.style.format({"buy_rate": "{:.2%}", "revenue_per_visitor": "${:.2f}", "rpv_ci_low": "${:.2f}",
                    "rpv_ci_high": "${:.2f}", "max_affordable_cost_usd": "${:.3f}"})''')

md("""## 4. D1: which channels deserve the credit?

The GA report uses **last non-direct click**: when someone comes back by bookmark, GA credits their earlier campaign. Here journeys follow the channel each visit **actually arrived through**. Purchases are credited under five rules, including a **third-order Markov chain** (the order was chosen by held-out likelihood in the repository).""")
code('''t = touches(sessions)
cap = t.loc[t.purchased, "revenue_usd"].quantile(0.99)
j = journeys(t, revenue_cap_usd=cap)
att = Attribution(j, "arrival_path", order=3)
shares = att.shares()
boot = att.bootstrap_shares(n_boot=N_BOOT, seed=42)
order = ["Organic Search", "Direct", "Paid Search", "Referral", "Display", "Social", "Affiliates"]
shares.loc[order].style.format("{:.1%}")''')
code('''gi, mi = MODELS.index("ga_last_click"), MODELS.index("markov")
idx = {c: i for i, c in enumerate(att.channels)}
chart_order = ["Direct", "Social", "Affiliates", "Display", "Paid Search", "Referral", "Organic Search"]
vals, lo, hi = [], [], []
for c in chart_order:
    d = 100 * (boot[:, idx[c], mi] - boot[:, idx[c], gi])
    vals.append(100 * (shares.loc[c, "markov"] - shares.loc[c, "ga_last_click"]))
    lo.append(np.percentile(d, 2.5)); hi.append(np.percentile(d, 97.5))
diverging_ci_chart(chart_order, vals, lo, hi,
    title="GA's report gives Organic Search 15 points of credit that belong to return visits",
    subtitle="Share of purchases credited by the data-driven (Markov) model minus the share GA's report credits, in points",
    note=f"{int(j.converted.sum()):,} outside purchases, Aug 31, 2016 – Jul 1, 2017. Whiskers: paired bootstrap 95% CI.",
    pos_label="GA under-credits", neg_label="GA over-credits",
    value_fmt=lambda v: f"{v:+.1f} pts", tick_fmt=lambda t: f"{t:+.0f} pts");''')
code('''clicks = t.arrival_channel.value_counts()
revenue = att.credit_table().xs("revenue", axis=1, level="measure")[MODELS]
(revenue.loc[["Paid Search", "Display"]].div(clicks.loc[["Paid Search", "Display"]], axis=0)).T.style.format("${:.2f}")''')
md("""A Paid Search click is worth about **$1.56–$2.46** under every rule, so it breaks even below about $0.80–$1.20 per click at a 50% margin. Display's value depends heavily on how return visits carrying its campaign tag are credited ($2.85–$9.08). Those visits hold 8 of its 10 largest orders. **Test Display with a holdout before moving its budget.**""")

md("""## 5. Audit of Google's own model (summary)

Google's lab *Predict Visitor Purchases with a Classification Model in BigQuery ML* (GSP229) predicts `will_buy_on_return_visit` and reports ROC-AUC 0.91. The repository recreates it in BigQuery ML with the lab's own SQL (0.724 / 0.909 vs the published 0.72 / 0.91) and finds:
- **61%** of its training positives and **74%** of its evaluation positives are Google employees;
- its score is inflated by **0.047** (95% CI 0.035–0.059) compared with outside visitors, and the gap disappears without the traffic-source features;
- its top 1% of prospects are **98.8% employees**, and only 2 of those 1,020 are outside customers who later bought.

## 6. Recommendations

1. Filter employee traffic out of marketing reports.
2. Report channels with a multi-touch view next to GA's default.
3. Keep Paid Search and cap bids near $0.80–1.20 per click.
4. Run a Display holdout before changing its budget.
5. Retarget only top-scored first-time visitors, with a 10% holdout.
6. Invest in returning visitors and corporate buyers.
7. Stop paying for YouTube or affiliates to drive sales.

*Data: Google Analytics 360 sample (`bigquery-public-data.google_analytics_sample`), published by Google.*""")

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = cells
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kaggle/merch_store_marketing_case_study.ipynb"
nbf.write(nb, out)
print("written", len(cells), "cells")

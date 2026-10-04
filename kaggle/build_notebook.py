"""Build kaggle/merch_store_marketing_case_study.ipynb from this repository.

The helper cells are the src/ modules, copied in unchanged, so the Kaggle notebook runs the same code
as the full project. Rebuild after changing src/:  python kaggle/build_notebook.py [output path]
Importing this module has no side effects: build_cells() / build_notebook() assemble the notebook in memory,
and only running the script writes the file.
"""
import re
import sys
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "kaggle/merch_store_marketing_case_study.ipynb"
# The gradient-boosting settings that won time-based tuning (top gradient_boosting entry by cv_pr_auc in
# data/processed/tuning_results.json; tests/test_kaggle_notebook.py checks they still match).
BOOSTING_PARAMS = {"learning_rate": 0.02, "max_leaf_nodes": 4, "min_samples_leaf": 1000}


def module_source(name: str) -> str:
    """Repo module text with its intra-repo imports removed (everything shares one namespace here)."""
    src = (ROOT / "src" / f"{name}.py").read_text()
    src = re.sub(r"^from src(\.\w+)? import .*\n", "", src, flags=re.M)
    src = src.replace("from __future__ import annotations\n", "")
    return f"# ---- src/{name}.py (copied unchanged from the repository, minus its imports of other repo modules)\n" + src


def build_cells() -> list:
    """The notebook's cells, in order (Markdown and code), built from the current repository."""
    cells = []
    # No $ sign may reach the notebook's Markdown. Jupyter and Kaggle read text between two $ signs as math, and
    # GitHub's notebook viewer strips the backslash from an escaped \$ before its math renderer runs, so even
    # "\$700–\$760" turns into math there. Amounts in Markdown are written "USD 690–745" instead.
    def md(s):
        if "$" in s:
            raise ValueError(f"write amounts in Markdown as 'USD 690', not with a $ sign: {s[:60]!r}")
        cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))
    boosting_args = ", ".join(f"{k}={v!r}" for k, v in BOOSTING_PARAMS.items())

    md("""# Where should the Google Merchandise Store spend its next marketing dollar?

*Pann Phetra · [github.com/Pann13223029](https://github.com/Pann13223029) · Analysis designed and directed by me; code written with AI assistance.*

Taken at face value, Google's online merch store runs on referral traffic. It doesn't: about 41% of its revenue comes from Google's own employees, clicking through from an internal link that the public data hides as "Referral". In this notebook I find them, set them aside, and then answer the two questions the store's Head of Marketing actually faces:
- **D1:** which channels actually produce buyers?
- **D2:** which first-time visitors are worth paying to bring back?

It's a condensed, runnable version of my full project on the Google Analytics 360 export of the Google Merchandise Store (BigQuery public dataset, Aug 2016 – Aug 2017): **[github.com/Pann13223029/merch-store-marketing-case-study](https://github.com/Pann13223029/merch-store-marketing-case-study)** (write-ups, 6 notebooks, SQL, tests, charts and an executive summary). Kaggle publishes every public notebook under the Apache 2.0 license; the same code is also MIT-licensed in the GitHub repo. It's also my capstone for the Google Data Analytics certificate.

**What you'll see:**
1. **The employees, and one buyer who needs a segment of its own.** About 41% of the store's revenue comes from Google staff, and one outside corporate buyer (the key account) decides Display's numbers alone. I report both on their own before any marketing analysis.
2. **Retargeting:** scored the way a live campaign would score them, the model's top 10% of first-time visitors holds 71% of the outside customers who bought within the next 30 days, against 61% for a two-line rule (North America first, then how far the visit got). But the top 10% is worth only up to about USD 690–745 a month in extra gross profit, if the ads reach everyone in the audience (before ad costs).
3. **Attribution:** the store's GA report likely over-credits Organic Search by up to about 15 points of purchases. A Paid Search click's attributed value is a ceiling, so I'd split brand from non-brand search and test pausing brand ads (not yet sized). Without the key account, Display's value is stable.

I also audited **Google's BigQuery ML teaching lab** for this question, and its published 0.91 turned out to be inflated by employees. That audit needs your own Google Cloud project, so I summarize it at the end rather than re-run it here.""")

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

These cells are my repository's `src/` modules, copied unchanged (apart from imports between them), so every result here comes from the same code as the full project.""")
    for name in ["validate", "tables", "segments", "journeys", "modeling", "breakeven", "attribution", "viz"]:
        code(module_source(name))

    md("""## 1. One clean row per session (BigQuery SQL)

I start in BigQuery. The query unnests each session's hits to count funnel steps (product views, add-to-cart, checkout), and flags employee traffic at the **visitor** level: anyone who ever arrived through the hidden internal referral (`Referral`, source `(direct)`, path `/`). It scans about 0.8 GB.""")
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

    md("""## 2. About 41% of revenue is Google employees, and one buyer is a segment of its own

In the public data, employees' visits look like `Referral` traffic with the source hidden, and most come from Google office cities. Google's training copy of this data shows the real source, `mall.googleplex.com` (the employee store link). Checked against it, my flag is 99.2% precise and catches 98.2% of employee purchases (details in the repository).

I also report one outside visitor on its own (`KEY_ACCOUNTS` above): a corporate buyer, 278 visits on weekdays in office hours, that was already buying before its only Display click, and whose next 15 purchases GA's campaign carry-over then labeled Display. **From here on, every analysis uses outside visitors only, without the key account.**""")
    code('''summary = sessions.assign(segment=segment(sessions)).groupby("segment").agg(
    visitors=("full_visitor_id", "nunique"), sessions=("session_key", "size"),
    purchase_sessions=("purchased", "sum"), revenue_usd=("revenue_usd", "sum"))
summary.assign(share_of_revenue=summary.revenue_usd / summary.revenue_usd.sum()).style.format(
    {"visitors": "{:,}", "sessions": "{:,}", "purchase_sessions": "{:,}", "revenue_usd": "${:,.0f}", "share_of_revenue": "{:.1%}"})''')

    md("""## 3. D2: which first-time visitors are worth bringing back?

Here's how I set it up:

- **Population:** outside visitors whose first visit didn't end in a purchase.
- **Label:** a purchase on a later visit within 30 days.
- **Train/test split:** train on first visits Aug 2016 – Apr 2017, test on first visits May 1 – Jul 1, 2017, which were never used to choose features, settings or the model.

The chart of later-purchase rates by channel leaves out the `(Other)` channel (23 training first visits and no later buyers, as the repository's notebook 04 shows).""")
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

    md("""**The model:** I use gradient boosting with the settings that won time-based tuning in the repository (28 configurations; learning rate 0.02, 4-leaf trees, minimum leaf size 1,000), and compare it with two rules that need no model (the funnel rule, and a **two-line rule**: North American first visits first, then the funnel rule) and with the feature set of Google's own lab, refit on this corrected data.""")
    code(f'''model = boosting_pipeline({boosting_args}).fit(train, train[LABEL])
scores = {{"Funnel rule": funnel_rule_score(test),
          "Two-line rule": funnel_geo_rule_score(test),
          "Lab's features, refit": lab_features_pipeline().fit(train, train[LABEL]).predict_proba(test)[:, 1],
          "My model (gradient boosting)": model.predict_proba(test)[:, 1]}}
y = test[LABEL].to_numpy()
pd.DataFrame({{k: ranking_metrics(y, s) for k, s in scores.items()}}).T.style.format(
    {{"pr_auc": "{{:.3f}}", "roc_auc": "{{:.3f}}", "precision_top_1pct": "{{:.1%}}", "lift_top_1pct": "{{:.1f}}x",
     "lift_top_5pct": "{{:.1f}}x", "lift_top_10pct": "{{:.1f}}x", "recall_top_10pct": "{{:.1%}}"}})''')
    md("""**Scored as a live campaign would score them.** The table above leaves out every visitor whom any visit in the year marks as an employee. For a first-time visitor, that flag often comes from a later visit, which a live campaign can't see when it scores the first one. So I rebuild the table with only what's knowable at scoring time (`internal_flag="first_visit"`): a visitor is left out only if the first visit itself comes through the internal link. I refit the model on it with the same settings and months. **Staff**, the employees whom only a later visit reveals, stay in training and in the ranking, but their purchases don't count as later buyers, because an ad can't turn them into customers.""")
    code(f'''live = add_features(remarketing_table(sessions, internal_flag="first_visit"))
train_live, test_live = live[live.split == "train"], live[live.split == "test"]
staff = test_live.is_internal.to_numpy()        # flagged as employees only by a later visit: for evaluation only
y_live = test_live[LABEL].to_numpy()
y_real = y_live * ~staff                        # real later buyers: staff purchases don't count
m = "My model (gradient boosting)"
live_scores = {{m: boosting_pipeline({boosting_args})
                    .fit(train_live, train_live[LABEL]).predict_proba(test_live)[:, 1],
               "Two-line rule": funnel_geo_rule_score(test_live),
               "Funnel rule": funnel_rule_score(test_live),
               "My model trained without staff, as first reported": model.predict_proba(test_live)[:, 1]}}
n_real = int(y_real.sum())
reached = {{k: top_share_count(y_real, s, 0.10) for k, s in live_scores.items()}}
for k, count in reached.items():
    print(f"{{k}}: {{count}} of {{n_real}} real later buyers in the top 10% ({{count / n_real:.0%}})")
all_reached, staff_reached = (top_share_count(yy, live_scores[m], 0.10) for yy in (y_live, y_live * staff))
print(f"Staff are {{staff_reached}} of the {{all_reached}} later buyers the model's top 10% reaches ({{staff_reached / all_reached:.0%}})")

colors = {{m: SERIES_1, "Two-line rule": "#eda100", "Funnel rule": "#1baf7a"}}
gains_chart({{k: cumulative_gains(y_real, live_scores[k]) for k in colors}}, colors,
    title=f"The model's top 10% reaches {{reached[m] / n_real:.0%}} of later buyers; a two-line rule reaches "
          f"{{reached['Two-line rule'] / n_real:.0%}}",
    subtitle="Share of real later buyers (a purchase within 30 days, staff not counted) captured by the highest-scored first visits",
    note=f"Out-of-time test, scored as a live campaign would: first visits May 1 – Jul 1, 2017 ({{len(test_live):,}} first-time visitors, "
         f"{{n_real}} real\\nlater buyers). Staff stay in the ranking; their purchases don't count. Model refit with staff kept in training.");''')

    md("""The model clearly beats the funnel rule, and the two-line rule by about 10 points of real later buyers (95% CI 4.5–15 in the repository). What it mainly adds is a sharper top of the ranking: its PR-AUC edge over the rule isn't significant. Part of the 71% comes from keeping staff in training. Trained without them, as first reported, the model reaches 68% of the same buyers (the fourth line printed above). And staff are nearly 1 in 4 of the later buyers its top 10% reaches. Retargeting can't cause their purchases, so they count as reach, not value.

**Break-even:** is a visitor worth retargeting? Only if revenue per visitor over the next 30 days × incremental lift × gross margin is at least the cost per visitor the ads actually reach. I measure revenue per visitor on the test months, take a 10% lift anchored on randomized experiments (8–10.5%) that measured it on people who actually saw an ad, and assume a 50% margin.""")
    code('''model_score = scores["My model (gradient boosting)"]
bands = value_by_band(y, test.revenue_30d_usd.to_numpy(), model_score)
bands["max_affordable_cost_usd"] = max_affordable_cost(bands.revenue_per_visitor)
bands.style.format({"buy_rate": "{:.2%}", "revenue_per_visitor": "${:.2f}", "rpv_ci_low": "${:.2f}",
                    "rpv_ci_high": "${:.2f}", "max_affordable_cost_usd": "${:.3f}"})''')
    code('''# Extra gross profit per month, before ad costs, from retargeting the top X% (central case, 95% CI, range across lift x margin)
months = months_spanned(test.session_date)   # May 1 – Jul 1, 2017: 62 days, both ends included
value_of_targets(y, test.revenue_30d_usd.to_numpy(), model_score, months).style.format(
    {"visitors_per_month": "{:,.0f}", "share_of_later_buyers": "{:.0%}", "revenue_per_visitor": "${:.2f}",
     "gross_profit_per_month": "${:,.0f}", "ci_low": "${:,.0f}", "ci_high": "${:,.0f}", "range_low": "${:,.0f}",
     "range_high": "${:,.0f}"})''')
    md("""The top 10% (about 4,500 visitors a month) is worth up to about USD 690–745 a month, and the top 20% up to about USD 920–950. The lower figures are the central case above; the higher ones come from the repository, once employees in the audience are valued at zero (USD 744 and USD 948). These are upper bounds: extra gross profit before ad costs, if the ads reach everyone in the audience. So retargeting is a small, cheap program. Its real lift needs a randomized test: I sized it in the repository as a 50/50 holdout of the top 20% for 12 months, which detects a lift of about 14% or more.""")

    md("""## 4. D1: which channels deserve the credit?

GA's report uses **last non-direct click**: when someone comes back by bookmark, GA credits their earlier campaign. I rebuild the journeys instead, following the channel each visit **actually arrived through**, with employees and the key account left out. Purchases are credited under five path-based rules (GA's report, last touch, first touch, linear and position-based) and a **third-order Markov chain** (the order was chosen by held-out likelihood in the repository). Revenue is capped per purchase session at the cap set in the repository's Process phase (USD 1,606), and the bootstrap resamples visitors.""")
    code('''t = touches(sessions)                      # outside visitors, key account excluded
cap = revenue_cap(sessions)                # $1,606 per purchase session
j = journeys(t, revenue_cap_usd=cap)
att = Attribution(j, "arrival_path", order=3)
shares = att.shares()
boot = att.bootstrap_shares(n_boot=N_BOOT, seed=42, clusters=j.full_visitor_id)   # resamples visitors
order = ["Organic Search", "Direct", "Paid Search", "Referral", "Display", "Social", "Affiliates"]
print(f"{int(j.converted.sum()):,} purchases; revenue cap ${cap:,.2f} per purchase session")
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
    title="GA's report likely over-credits Organic Search by up to about 15 points of purchases",
    subtitle=("Share of purchases credited by a data-driven (Markov) model minus the share GA's channel report credits, in points,\\n"
              "if every direct return (isTrueDirect) was self-initiated: the top of the range"),
    note=(f"{int(j.converted.sum()):,} outside purchases, Aug 31, 2016 – Jul 1, 2017; Google employees and one key account excluded.\\n"
          "Whiskers: 95% CI from a paired bootstrap that resamples visitors."),
    pos_label="GA under-credits", neg_label="GA over-credits",
    value_fmt=lambda v: f"{v:+.1f} pts", tick_fmt=lambda t: f"{t:+.0f} pts");''')
    md("""The gap assumes every `isTrueDirect` visit was a self-initiated return. GA also sets that flag when two visits in a row carry the same campaign details, such as a repeat Google search, and the export can't tell the two apart. So in the repository I show the range: 15.5 points at the top, 1.8 on GA's own labels, and about 13–15 at a benchmark from how returning visitors behave. Hence **"likely up to about 15 points"**, with moderate confidence. With Direct split by visit number, visitors returning directly bring 34% of purchases; when they first came from a campaign, GA's report credits that campaign.

**What is a paid click worth?** Capped revenue each rule credits to a channel, divided by the visits it actually brought (ad clicks). The seventh rule, a conservative relabel, is the repository's last one.""")
    code('''clicks = t.arrival_channel.value_counts()
revenue = att.credit_table().xs("revenue", axis=1, level="measure")[MODELS].copy()
revenue["markov_conservative"] = Attribution(j, "conservative_path", order=3).credit_table()[("markov", "revenue")]
per_click = revenue.loc[["Paid Search", "Display"]].div(clicks.loc[["Paid Search", "Display"]], axis=0).T
per_click["Paid Search bid cap, 50% margin"] = per_click["Paid Search"] * 0.5
per_click.style.format("${:.2f}")''')
    md("""**Paid Search:** a click is credited with **USD 1.56–2.43** under the seven rules, so at a 50% margin the break-even bid is at most **USD 0.78–1.21**, and only if every attributed sale needed the ad. In the repository, all 65 Paid Search purchases with a readable keyword were on brand keywords (the store's or Google's name), the clicks least likely to need an ad, so I'd split brand from non-brand and test pausing brand ads (not yet sized) before raising bids.

**Display:** without the key account, a click is credited with **USD 2.84–3.87** under every rule (GA's report put it at USD 9.08 with the account, as the repository's notebook 05 shows). The value is stable, but it's attribution, not incrementality. A purchase-based holdout can't detect Display's effect at about 2 credited purchases a week, so I recommend a 12-week 50/50 holdout of Display's own audience, measured on site visits (it has enough power only if that audience makes at most about 500–1,000 visits a week).""")

    md("""## 5. Audit of Google's teaching lab (summary)

Google's lab *Predict Visitor Purchases with a Classification Model in BigQuery ML* (GSP229) teaches BigQuery ML with the label `will_buy_on_return_visit`, and reports ROC-AUC 0.91. I recreated it in BigQuery ML with the lab's own SQL (0.724 / 0.909 vs the published 0.72 / 0.91) and found:
- its 0.91 is inflated: **61%** of its training positives and **74%** of its evaluation positives are Google employees, and its top 1% of prospects are **98.8% employees**;
- on outside visitors it scores **0.863** (against 0.910 on all first visits, both recomputed from the row-level predictions), close to a version trained without employees (0.877).

Fine for teaching; remove internal traffic before targeting.

## 6. What I'd do

Four things can start now, two need a test first, and two are worth exploring. A third test, pausing brand ads, should follow the brand/non-brand split, but it isn't sized yet.

**Now**

1. Report Google employees and the key account as their own segments.
2. Show a multi-touch view beside GA's channel report, with the Organic Search range.
3. Split brand from non-brand search, and keep bids under the value ceiling (USD 0.78–1.21 a click at most).
4. Review any spend on YouTube promotion and affiliates.

**Test**

5. Retarget only the top-scored first-time visitors, and measure the lift with a 50/50 holdout of the top 20% for 12 months.
6. Hold Display's budget and run a 12-week 50/50 holdout of Display's own audience, measured on site visits (if that audience makes at most about 500–1,000 visits a week).

**Explore**

7. Retention: email and reminders for past visitors.
8. A direct sales path for corporate (bulk) buyers.

**What would change my mind:**
- **Retargeting:** if the test's purchase lift sits clearly above break-even (its 95% CI above cost per retargeted visitor ÷ (USD 2.10 × 50% margin)), scale it; if clearly below, stop. In between, I extend the test once, up to 24 months.
- **Display:** if held-out users make significantly fewer visits, Display adds traffic, and I'd value those visits against its cost. If the measured difference stays below the visits GA credits to Display, I'd budget it on that difference, not on GA's report.
- **Organic Search:** if the store's own data showed that most direct returns start with a fresh Google search, the over-credit shrinks toward the 1.8 points on GA's own labels.
- **Paid Search:** no rule yet. The brand-pause test needs sizing first.

## 7. What I learned

- **Question the data before the model.** The biggest findings came from asking who is in the data: employees and one corporate buyer changed almost every channel number.
- **Credit isn't causation.** Attribution shows the paths people took, not what each channel caused, so the recommendations end in tests.
- **Review early.** An AI-assisted red-team review changed six conclusions at the end; next time I'd review at the start and halfway too. The repository lists what changed.
- **Start from the decision.** Framing the work around the Head of Marketing's two decisions told me which analyses mattered.

*Data: Google Analytics 360 sample (`bigquery-public-data.google_analytics_sample`), published by Google.*""")

    return cells


def build_notebook() -> nbf.NotebookNode:
    """The full notebook (no outputs), ready to write."""
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = build_cells()
    return nb


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    nb = build_notebook()
    nbf.write(nb, out)
    print("written", len(nb.cells), "cells")

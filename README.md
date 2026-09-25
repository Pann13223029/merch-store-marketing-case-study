# Where should the Google Merchandise Store spend its next marketing dollar?

A marketing-analytics case study on Google's own online store: its Google Analytics 360 export in BigQuery (Aug 2016 – Aug 2017, 903,653 sessions). The project follows the Google Data Analytics case-study structure (**Ask → Prepare → Process → Analyze → Share → Act**), using the methods of the Advanced certificate: SQL on nested data, hypothesis tests, regression, tuned machine-learning models, and bootstrap confidence intervals.

**Stakeholder:** the store's Head of Marketing, who has two decisions to make:
- **D1: channel budget.** Which channels actually produce buyers, compared with what the GA report credits them for?
- **D2: retargeting.** Which first-time visitors are worth paying to bring back?

→ **One-page executive summary for the stakeholder:** [reports/00_executive_summary.md](reports/00_executive_summary.md)

---

## Key findings

**1. About 41% of the store's revenue is Google employees, not marketing.** In the public data, their visits show up as "Referral" with the source hidden. Google's own training copy of the data shows the referrer: `mall.googleplex.com`, the internal employee store link. That let us check our employee filter against the true source: 99.2% of flagged visitors really are employees, and it catches 98.2% of employee purchases.

**2. Google's official BigQuery ML tutorial model for this question mostly learned to recognize Google employees.** We rebuilt it from the lab's own SQL and got the published scores back exactly (0.724 / 0.909 vs 0.72 / 0.91). Employees make up 61% of the "future buyers" it trains on, and 98.8% of its top-ranked prospects.

![The lab model's top prospects are employees](reports/figures/lab_audit_top_ranked.png)

**3. A corrected retargeting model finds the buyers, but the payoff is small.** Trained only on outside visitors, with a 30-day window, and tested once on months it never saw:
- the top 10% of first-time visitors include **70% of the people who later buy**, and the bottom 40% include none;
- even so, the whole program is worth about **$700 a month** in extra gross profit before ad costs.

![Cumulative gains](reports/figures/d2_gains_chart.png)

**4. The store's channel report over-credits Organic Search by about 15 points of purchases.** GA's default report credits a visitor's earlier campaign when they come back by bookmark or typed URL. Traced through actual visitor journeys, **people returning on their own drive about half of purchases and 61% of revenue**, and the report hides that.

![Credit gap](reports/figures/d1_credit_gap.png)

**5. Paid Search pays its way, but Display can't be judged from this data.**
- **Paid Search:** a click is worth $1.56–$2.46 under every attribution model tested.
- **Display:** a click is worth $2.85–$9.08, depending on how 1,118 ambiguous return visits are credited. Those visits hold 8 of its 10 largest orders.

![Value per paid click](reports/figures/d1_value_per_click.png)

## Recommendations (Act)

| # | Recommendation | Evidence |
|---|---|---|
| 1 | Filter internal traffic out of marketing reports and track it as its own segment | 41% of revenue, 47% of purchase sessions |
| 2 | Report channels with a multi-touch view next to GA's default report | Organic Search over-credited by 15.4 pts; returning visitors under-credited by 16.9 |
| 3 | Keep Paid Search; set maximum bids from the value of a click (about $0.80–$1.20 at a 50% margin) | $1.56–$2.46 per click under every model |
| 4 | Run a Display holdout test before changing its budget | 91% of Display's reported revenue comes from ambiguous return visits |
| 5 | Retarget only high-scoring first-time visitors, with a 10% holdout to measure the real effect | Top 1% worth up to $0.70 per visitor; beyond the top 20%, under $0.01 |
| 6 | Invest in returning visitors and a path for corporate (bulk) buyers | 61% of revenue from returning visitors; 51 bulk orders = 29% of revenue |
| 7 | Don't pay for YouTube or affiliates to drive sales | YouTube: about 98,000 visits in Oct–Nov 2016 with 0 purchases; Affiliates: 9 sales all year |

---

## How the analysis was done

### Ask ([reports/01_ask.md](reports/01_ask.md))
The business task, stakeholders, 8 guiding questions, 4 hypotheses with pre-specified tests, and success criteria. The public dataset is heavily used, and Google's own lab already asks "will this visitor buy?", so the project is framed around **budget decisions** and around **auditing** that lab rather than repeating it.

### Prepare ([reports/02_prepare.md](reports/02_prepare.md) · [notebook 01](notebooks/01_prepare.ipynb))
- **Credibility and completeness:** a ROCCC credibility check and a field-by-field completeness review (city 56% redacted, campaign 97% unset, `timeOnSite` NULL means zero).
- **Employee traffic:** found as referral traffic with a hidden source, 66% of it from Google office cities.
- **Duplicates:** the 898 "duplicate" sessions turned out to be visits split at midnight.
- **Order values:** the top 1% of orders carry 29% of revenue, so revenue is **capped per order** and purchases count first.

### Process ([reports/03_process.md](reports/03_process.md) · [notebook 02](notebooks/02_process.ipynb))
- **One clean session table** from nested BigQuery data (`UNNEST` over hits for funnel steps), with 15 validation checks against raw totals, all passing.
- **GA credit shifting:** 29% of outside purchases happen on return visits that GA credits to an earlier campaign. So attribution uses the channel visitors **actually arrived through**, and GA's own labels are kept as the baseline.

### Analyze ([reports/04_analyze.md](reports/04_analyze.md))
| Part | Notebook | Methods | Headline |
|---|---|---|---|
| **A. Audit of Google's lab (GSP229)** | [03](notebooks/03_analyze_lab_audit.ipynb) | Recreated the lab in **BigQuery ML** with its own SQL; unredacted ground truth; paired bootstrap; test with features removed to find the cause | H4 supported: ROC-AUC is inflated by 0.047 (95% CI 0.035–0.059), and the traffic-source features are how the model recognizes employees |
| **B. Retargeting model (D2)** | [04](notebooks/04_analyze_remarketing.ipynb) | χ² test with Cramér's V, z-tests, logistic regression with odds ratios; logistic regression, random forest, and gradient boosting tuned with **time-based folds and an embargo**; one test on unseen months; calibration and permutation importance; break-even analysis | PR-AUC 0.061 vs 0.028 for a simple rule; top 10% of visitors = 70% of later buyers; fixing the data mattered more than a fancier model |
| **C. Attribution (D1)** | [05](notebooks/05_analyze_attribution.ipynb) | Last-touch, first-touch, linear, and position-based models; a **third-order Markov chain** (order chosen by held-out likelihood) with a value-weighted removal effect; paired Poisson bootstrap | H2 holds against true last click but is reversed against GA's report; Organic Search over-credited by 15.4 pts |

### Share
- **Executive summary** for the Head of Marketing: [reports/00_executive_summary.md](reports/00_executive_summary.md)
- **Stakeholder deck** (14 slides with speaker notes): [link to the shared deck or its PDF]
- **Looker Studio dashboard** (overview, channel credit, and an interactive break-even page with lift, margin, and cost controls): [link once published]. The data and a step-by-step build guide are in [dashboard/](dashboard/)
- **Kaggle notebook**, a runnable condensed version: [kaggle/](kaggle/) · [link once published]
- **Charts:** [reports/figures/](reports/figures/). Chart colors come from a validated palette, checked for color-blind separation.

## Getting things right: corrections made along the way

Every one of these is recorded in the decision logs, because catching them is part of the analysis:
- **Wrong assumption about the lab's data.** I first assumed Google's lab used the public dataset. It uses a different, fuller table. The audit was redone on the lab's own table, which turned an estimate of the employee share (53%) into a verified figure (61%).
- **An ID that wasn't unique.** The lab's `unique_session_id` repeats for visits split at midnight. Joining predictions on it duplicated 1,666 rows, so the join now uses a fingerprint of all grouping columns.
- **Memoryless Markov chain.** A first-order chain credited Social with nearly 3× the purchases its journeys actually produced. A third-order chain fixed most of this.
- **A rejected revenue-credit method.** Splitting each purchase's revenue by global removal effects gave Paid Search less credit than last click. It was replaced by a value-weighted removal effect.

## Reproduce it

**Prerequisites:** Python 3.12, the [gcloud CLI](https://cloud.google.com/sdk/docs/install), and a Google Cloud project with the BigQuery API enabled (a free [BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox) project is enough).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
gcloud auth application-default login                      # Google account with a free BigQuery sandbox project
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Then run `notebooks/01` → `05` in order.
- **Downloads:** notebook 02's first run downloads the clean session table (0.81 GB scanned, about 8 minutes), and later runs read the local cache in `data/` (gitignored).
- **Committed results:** the BigQuery ML results (`data/raw/a11_*`, `a13_*`) and the tuning results (`data/processed/tuning_results.json`) are in the repository, so `RUN_BQML` (notebook 03) and `RUN_TUNING` (notebook 04) stay `False` by default. Set one to `True` to recompute: about 7 minutes for the BigQuery ML models, about 35 minutes for the tuning.
- **Run time** with `data/` filled: 01 and 02 under a minute each, 03 about 6 minutes (almost all of it paired bootstraps), 04 about 6–12 minutes, 05 about a minute. A busy machine can take twice as long.
- **Cost:** every query stays well inside BigQuery's free tier. The query helper ([src/bq.py](src/bq.py)) dry-runs each query first, refuses anything scanning more than 20 GB, and caps each job's billed bytes at the same 20 GB. The BigQuery ML model-creation statements skip the dry run, which doesn't reliably estimate their cost, and run under that cap alone; they scan under 0.3 GB each.

```
├── reports/     executive summary, phase write-ups (01_ask … 04_analyze), figures/
├── notebooks/   01_prepare … 05_analyze_attribution
├── sql/         prepare/ · process/ · audit/ (incl. the lab's BigQuery ML models)
├── src/         bq · validate · tables · journeys · lab_audit · modeling · breakeven · attribution · viz · dashboard
├── dashboard/   Looker Studio build guide and its CSV data (python -m src.dashboard)
├── kaggle/      self-contained Kaggle notebook
└── data/        local cache, rebuilt from sql/ (gitignored, except the BigQuery ML and tuning results)
```

## Limitations and ethics

- **Correlation, not causation.** Attribution models describe the paths people took, not what each channel caused, and the retargeting lift is borrowed from published experiments. The recommendations include holdout tests because only experiments can settle these questions.
- **Data gaps.** There's no ad-cost data, visitors are identified per device, and the data is 2016–17 Universal Analytics. The methods carry over to GA4's BigQuery export.
- **Privacy.** The data is anonymized by Google. Recommendations assume retargeting reaches only visitors who consented. The retargeting audience is 91% North American because that's where buyers are, and the report says so openly.

**Data:** Google Analytics 360 sample dataset (`bigquery-public-data.google_analytics_sample`) and the `data-to-insights.ecommerce.web_analytics` table used in Google's training labs, both published by Google for learning.

# ![Where should the Google Merchandise Store spend its next marketing dollar?](.github/banner.jpg)

Pann Phetra · [github.com/Pann13223029](https://github.com/Pann13223029)

BigQuery SQL · BigQuery ML · Python (pandas, scikit-learn, statsmodels) · Looker Studio

A marketing-analytics case study on Google's own online store: its Google Analytics 360 export in BigQuery (Aug 2016 – Aug 2017, 903,653 sessions). The project follows the Google Data Analytics case-study structure (**Ask → Prepare → Process → Analyze → Share → Act**), using the methods of the Advanced certificate: SQL on nested data, hypothesis tests, regression, tuned machine-learning models, bootstrap confidence intervals, and power analysis.

**Stakeholder:** the store's Head of Marketing, who has two decisions to make:
- **D1: channel budget.** Which channels actually produce buyers, compared with what the GA report credits them for?
- **D2: retargeting.** Which first-time visitors are worth paying to bring back?

## The answer

| Decision | Answer | Confidence |
|---|---|---|
| **D1: channel budget** | Don't move budget on GA's channel report. It likely over-credits Organic Search by up to about 15 points of purchases, and it credits visitors returning directly (34% of purchases) to earlier campaigns. Keep Paid Search, but bid below its value ceiling ($0.78–$1.21 a click at most) and split brand from non-brand. Hold Display's budget: its reported revenue was mostly one existing corporate buyer, so test it first. | **Moderate** for the Organic Search gap: the direction holds under every reading, but the size rests on an assumption the data can't check. **Low** for what Paid Search and Display cause, which only tests can show |
| **D2: retargeting** | Retarget only the top-scored first-time visitors. The model's top 10% holds 72% of real later buyers (61% for a two-line rule), but it's worth only about $710–$740 a month before ad costs. Measure the real lift with a 50/50 holdout. | **High** for the ranking; **low** for the lift, which is borrowed from published experiments |

**Why there's no "% budget shift per channel":** the data has no ad costs, and attribution credit isn't incremental, so moving money by credit could fund channels that didn't cause the sales. Test first: [notebook 06](notebooks/06_test_design.ipynb) sizes the tests.

→ **One-page executive summary for the stakeholder:** [reports/00_executive_summary.md](reports/00_executive_summary.md)

---

## Key findings

**1. About 41% of the store's revenue is Google employees, not marketing.** In the public data, their visits show up as "Referral" with the source hidden. Google's own training copy of the data shows the referrer: `mall.googleplex.com`, the internal employee store link. That let me check the employee filter against the true source: 99.2% of flagged visitors really are employees, and it catches 98.2% of employee purchases.

**2. Google's BigQuery ML teaching lab for this question publishes a score that employees inflate.** GSP229 teaches BigQuery ML on this data. I rebuilt it from the lab's own SQL and got the published scores back (0.724 / 0.909 vs 0.72 / 0.91). Its 0.91 ROC-AUC is inflated because 61% of its training positives (74% of its evaluation positives) are employees, and its top 1% of prospects are 98.8% employees. On outside visitors it scores 0.863, close to a version trained without employees (0.877). Fine for teaching; remove internal traffic before targeting.

![The lab model's top prospects are employees](reports/figures/lab_audit_top_ranked.png)

**3. A corrected retargeting model finds most later buyers, but a simple rule comes close and the payoff is small.** Trained only on outside visitors, with a 30-day window, and tested once on months it never saw:
- scored at their first visit, the top 10% of first-time visitors hold **72% of the outside customers who bought within the next 30 days** (95% CI 67–77%), against **61%** for a two-line rule: North American visitors first, then how far they got toward checkout;
- about 1 in 5 of the later buyers the top 10% reaches are Google employees whom only a later visit reveals, so they count as reach, not value;
- the top 10% is worth about **$710–$740 a month** in extra gross profit before ad costs, and the top 20% about **$1,025**.

![Cumulative gains](reports/figures/d2_gains_chart.png)

*The chart shows the population as first reported (70% vs 61%). Scored without hindsight about who is an employee, the model reaches 72% of real later buyers.*

**4. GA's channel report likely over-credits Organic Search by up to about 15 points of purchases.** When a visitor comes back by bookmark or typed URL, GA credits their earlier campaign. Traced through visitor journeys, a data-driven model gives Organic Search 38.2% of the 5,058 outside purchases, against GA's 53.8%: a 15.5-point gap (95% CI 14.4–16.6) if every direct return was self-initiated. On GA's own labels the gap is 1.8 points, and a benchmark from how returning visitors behave puts it at about 13–15; about 7 of the points are a lookback choice. **Visitors returning directly bring 34% of purchases and 44% of capped revenue**, which the report credits to earlier campaigns. First-ever visits that arrived direct add 19% and 20%.

![Credit gap](reports/figures/d1_credit_gap.png)

**5. Keep Paid Search, but treat its value as a ceiling.** A click is credited with $1.56–$2.43 under every attribution rule tested. At a 50% margin, the break-even bid is at most $0.78–$1.21 if every sale needed the ad, $0.39–$0.61 if half did, and $0.19–$0.30 if a quarter did. All 65 Paid Search purchases with a readable keyword came from searches for the store or its brand, the least incremental kind; 77% have no readable keyword.

**6. Display's reported revenue was mostly one existing corporate buyer.** One outside visitor, 278 visits from a single office desktop, holds 89% of the revenue GA credits to Display. It was already buying before its only Display click, and GA's campaign carry-over labelled its next 15 purchases Display. Reported as its own segment like employees (the key account: 16 purchase sessions, $128,413, 15% of outside revenue in the period), it leaves a Display click with an attributed value of $2.84–$3.87 under every rule. Only a holdout test can show what Display adds.

![Value per paid click](reports/figures/d1_value_per_click.png)

## Recommendations (Act)

| When | Action | Owner | At stake |
|---|---|---|---|
| Now | Report Google employees and the key account as their own segments | Web analytics | 41% of revenue (employees); $128k (key account) |
| Now | Show a multi-touch view beside GA's channel report, with the Organic Search range | Head of Marketing, Finance | Up to ~15 points of purchase credit |
| Now | Split brand from non-brand search, and keep bids under the value ceiling ($0.78–$1.21 at most) | Paid media | $1.56–$2.43 attributed per click |
| Now | Review any spend on YouTube promotion and affiliates | Paid media | ~98,000 YouTube visits in Oct–Nov 2016, no purchases; 9 affiliate sales all year |
| Test | Retarget only the top-scored first-time visitors, and measure the lift with a 50/50 holdout of the top 20% for 12 months | Retargeting lead | $710–$1,025 a month |
| Test | Hold Display's budget and run a 12-week 50/50 holdout measured on site visits | Paid media | Display's ~$17k of revenue a year |
| Explore | Retention: email and reminders for past visitors | Head of Marketing | 34% of purchases |
| Explore | A direct sales path for corporate (bulk) buyers | Head of Marketing, Sales | $248,552 of bulk purchase sessions, half of it one account |

---

## How the analysis was done

### Ask ([reports/01_ask.md](reports/01_ask.md))
The business task, stakeholders, 8 guiding questions, 4 hypotheses with pre-specified tests, and success criteria. The public dataset is heavily used, and Google's own lab already asks "will this visitor buy?", so the project is framed around **budget decisions** and around **auditing** that lab rather than repeating it.

### Prepare ([reports/02_prepare.md](reports/02_prepare.md) · [notebook 01](notebooks/01_prepare.ipynb))
- **Credibility and completeness:** a ROCCC credibility check and a field-by-field completeness review (city 56% redacted, campaign 97% unset, `timeOnSite` NULL means zero).
- **Employee traffic:** found as referral traffic with a hidden source, 66% of it from Google office cities.
- **Duplicates:** the 898 "duplicate" sessions turned out to be visits split at midnight.
- **Order values:** the top 1% of purchase sessions carry 29% of outside revenue, half of it one corporate account, so revenue is **capped per purchase session** and purchases count first.

### Process ([reports/03_process.md](reports/03_process.md) · [notebook 02](notebooks/02_process.ipynb))
- **One clean session table** from nested BigQuery data (`UNNEST` over hits for funnel steps), with 15 validation checks against raw totals, all passing. They catch dropped or duplicated rows; a unit-test suite covers the logic.
- **GA credit shifting:** 29% of outside purchases happen on return visits that GA credits to an earlier campaign. So attribution uses the channel visitors **actually arrived through**, and GA's own labels are kept as the baseline.

### Analyze ([reports/04_analyze.md](reports/04_analyze.md))
| Part | Notebook | Methods | Headline |
|---|---|---|---|
| **A. Audit of Google's lab (GSP229)** | [03](notebooks/03_analyze_lab_audit.ipynb) | Recreated the lab in **BigQuery ML** with its own SQL; unredacted ground truth; paired bootstrap; test with features removed to find the cause | H4 supported: employees inflate the published ROC-AUC by 0.047 (95% CI 0.035–0.059). On outside visitors the lab model scores 0.863, close to a version trained without employees (0.877) |
| **B. Retargeting model (D2)** | [04](notebooks/04_analyze_remarketing.ipynb) | χ² test, z-tests, logistic regression with odds ratios; logistic regression, random forest, and gradient boosting tuned with **time-based folds and an embargo**; one test on unseen months against two no-model rules; Wilson intervals; calibration and permutation importance; a re-evaluation without hindsight; break-even analysis | PR-AUC 0.061 vs 0.028 (funnel rule) and 0.049 (two-line rule); 72% of real later buyers in the top 10% vs 61% for the two-line rule; the top 10% is worth about $710–$740 a month |
| **C. Attribution (D1)** | [05](notebooks/05_analyze_attribution.ipynb) | Last-touch, first-touch, linear, and position-based models; a **third-order Markov chain** (order chosen by held-out likelihood) with a value-weighted removal effect; paired bootstrap that resamples visitors; a visitor-concentration check; an assumption dial with a benchmark; a Shapley decomposition | H2 holds against true last click but is reversed against GA's report. GA likely over-credits Organic Search by up to about 15 points; one key account held 89% of Display's reported revenue; a Paid Search click's value is a ceiling ($1.56–$2.43) |
| **D. Test design** | [06](notebooks/06_test_design.ipynb) | Power and minimum detectable effect for two-sample tests; Poisson power with variance inflation for weekly counts | A 50/50 holdout of the top 20% for 12 months detects a 14% purchase lift (a 10/90 split has 24% power at +10% after a year). A purchase-based Display holdout has 5–6% power, so Display is tested on site visits |

### Share
- **Executive summary** for the Head of Marketing: [reports/00_executive_summary.md](reports/00_executive_summary.md)
- **Charts:** [reports/figures/](reports/figures/). Chart colors come from a validated palette, checked for color-blind separation.
- **Looker Studio dashboard:** a step-by-step build guide and its data in [dashboard/](dashboard/) (overview, channel credit, and an interactive break-even page with lift, margin, and cost controls).
- **Kaggle notebook:** a runnable, condensed version, [kaggle/merch_store_marketing_case_study.ipynb](kaggle/merch_store_marketing_case_study.ipynb), with publishing steps in [kaggle/](kaggle/).

## Getting things right: corrections made along the way

Every one of these is recorded in the decision logs, because catching them is part of the analysis:
- **Wrong assumption about the lab's data.** I first assumed Google's lab used the public dataset. It uses a different, fuller table. The audit was redone on the lab's own table, which turned an estimate of the employee share (53%) into a verified figure (61%).
- **An ID that wasn't unique.** The lab's `unique_session_id` repeats for visits split at midnight. Joining predictions on it duplicated 1,666 rows, so the join now uses a fingerprint of all grouping columns.
- **Memoryless Markov chain.** A first-order chain credited Social with nearly 3× the purchases its journeys actually produced. A third-order chain fixed most of this; the rest is flagged, and the Markov values for Social and Affiliates aren't used in headlines.
- **A rejected revenue-credit method.** Splitting each purchase's revenue by global removal effects gave Paid Search less credit than last click. It was replaced by a value-weighted removal effect.

**A red-team review.** A reviewer then re-derived the D1 and D2 headlines from the data, looking for claims the evidence didn't support, and six findings changed the analysis. One outside corporate buyer held 89% of Display's reported revenue, so it's now reported as its own segment, the key account. The Organic Search over-credit depends on how GA's direct-return flag is read, so it's now a range with moderate confidence, and a check that had seemed to confirm it turned out to be an export-day artifact. "Direct" mixed visitors returning on their own with first-ever visits, so returning visitors now get 34% of purchases rather than the half first reported. The retargeting population left out employees using visits from later in the year, which a live campaign can't see; scored without that hindsight, the model still reaches 72% of real later buyers, and employees are 22% of the buyers it reaches. Both proposed tests were too small to detect their effects, so notebook 06 sizes designs that can. And every Paid Search purchase with a readable keyword was a brand search, so its value is now a ceiling, with scenarios for how many sales the ads actually caused.

## Reproduce it

**Prerequisites:** Python 3.12, the [gcloud CLI](https://cloud.google.com/sdk/docs/install), and a Google Cloud project with the BigQuery API enabled (a free [BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox) project is enough).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
gcloud auth application-default login                      # Google account with a free BigQuery sandbox project
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Then run `notebooks/01` → `06` in order.
- **Downloads:** notebook 02's first run downloads the clean session table (0.81 GB scanned, about 8 minutes), and later runs read the local cache in `data/` (gitignored).
- **Committed results:** the BigQuery ML results (`data/raw/a11_*`, `a13_*`) and the tuning results (`data/processed/tuning_results.json`) are in the repository, so `RUN_BQML` (notebook 03) and `RUN_TUNING` (notebook 04) stay `False` by default. Set one to `True` to recompute: about 7 minutes for the BigQuery ML models, about 35 minutes for the tuning.
- **Run time** with `data/` filled: 01 and 02 under a minute each, 03 about 6 minutes (almost all of it paired bootstraps), 04 about 6–13 minutes, 05 about 4 minutes, 06 about a minute. A busy machine can take twice as long.
- **Cost:** every query stays well inside BigQuery's free tier. The query helper ([src/bq.py](src/bq.py)) dry-runs each query first, refuses anything scanning more than 20 GB, and caps each job's billed bytes at the same 20 GB. The BigQuery ML model-creation statements skip the dry run, which doesn't reliably estimate their cost, and run under that cap alone; they scan under 0.3 GB each.

**Tests:** the suite runs on synthetic data, with no BigQuery access; checks on the real journey table run only when `data/` is cached. CI runs it on every push.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

```
├── reports/     executive summary, phase write-ups (01_ask … 04_analyze), figures/
├── notebooks/   01_prepare … 06_test_design
├── sql/         prepare/ · process/ · audit/ (incl. the lab's BigQuery ML models)
├── src/         bq · validate · tables · segments · journeys · lab_audit · modeling · breakeven · attribution · power · viz · dashboard
├── tests/       pytest suite for the headline calculations
├── dashboard/   Looker Studio build guide and its CSV data (python -m src.dashboard)
├── kaggle/      self-contained Kaggle notebook
└── data/        local cache, rebuilt from sql/ (gitignored, except the BigQuery ML and tuning results)
```

## Limitations and ethics

- **Attribution isn't causal.** Attribution models describe the paths people took, not what each channel caused, and the retargeting lift is borrowed from published experiments. Notebook 06 sizes the holdout tests that would measure both.
- **Paid Search incrementality is unknown.** Its attributed value is a ceiling, and 77% of its purchases have no readable keyword, so the brand share can't be measured in full.
- **The key-account exclusion is a judgment.** It's documented (Analyze, Part C, decision D-C5), and the account is reported as its own segment rather than dropped.
- **Data gaps.** There's no ad-cost data, visitors are identified per device, and the data is 2016–17 Universal Analytics. The methods carry over to GA4's BigQuery export.
- **Margin.** The 50% gross margin is applied to revenue that includes tax and shipping, which overstates gross profit somewhat.
- **Privacy.** The data is anonymized by Google. Recommendations assume retargeting reaches only visitors who consented. The retargeting audience is 91% North American because that's where buyers are, and the report says so openly.

**Data:** Google Analytics 360 sample dataset (`bigquery-public-data.google_analytics_sample`) and the `data-to-insights.ecommerce.web_analytics` table used in Google's training labs, both published by Google for learning. **Code and write-ups:** [MIT license](LICENSE).

**Banner photo:** Adeniji Abdullahi A on [Pexels](https://www.pexels.com/photo/a-happy-man-looking-at-a-cellphone-10843136/) (free Pexels license), background removed.

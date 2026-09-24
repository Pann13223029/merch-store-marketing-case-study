# Phase 4 — Analyze

- **Part A: Audit of Google's lab model for D2** (this section)
- Part B: Corrected remarketing model and break-even audience (D2), to come
- Part C: Multi-touch attribution and budget shifts (D1), to come

---

## Part A — Auditing Google's own model for this question (GSP229)

Notebook: [notebooks/03_analyze_lab_audit.ipynb](../notebooks/03_analyze_lab_audit.ipynb) · SQL: [`sql/audit/`](../sql/audit/)

Google's lab *Predict Visitor Purchases with a Classification Model in BigQuery ML* (GSP229) predicts `will_buy_on_return_visit` from a visitor's first visit and reports **ROC-AUC 0.91**. It answers nearly the same question as our D2, so we audited it first.

### Summary

| Finding | Evidence |
|---|---|
| The lab trains on a **different table** from the public sample | `data-to-insights.ecommerce.web_analytics`: 137,660 extra sessions, 13,314 extra purchase sessions, $3.0M extra revenue |
| That table **shows the referrer the public sample hides**, `mall.googleplex.com` | Plus `gdeals.googleplex.com`, `moma.corp.google.com`, and Google's internal Sites. This is ground truth for employee traffic |
| Our inferred employee flag (public sample) is accurate | **99.2% precision**; catches **92.7%** of employee visitors and **98.2%** of their purchases. What it misses is about 1.5% of the purchases we treat as external |
| **61% of the lab's training positives and 74% of its evaluation positives are Google employees** | Employees are only 5.9% / 9.4% of first visits |
| The lab's models **reproduce exactly** in BigQuery ML with the lab's own SQL | Model 1: 0.724 (published 0.72). Model 2: 0.909 (published 0.91) |
| **H4 supported:** the published ROC-AUC is inflated by employees | 0.910 on all first visits vs **0.863** on external visitors. Gap **0.047**, 95% CI 0.035 to 0.059 |
| **Mechanism:** the traffic-source features recognize employees | Without source, medium, and channel the gap disappears: 0.003, 95% CI −0.008 to +0.013 |
| **The model's top prospects are employees** | Top 1% of first visits: **98.8% employees**; only **2 of 1,020** are external visitors who went on to buy. As an employee detector its score reaches ROC-AUC 0.974 |
| Contamination makes the model **worse at its real job** | On external visitors: dropping source features gives +0.019 ROC-AUC (95% CI +0.007 to +0.031); training without employees gives +0.014 (95% CI +0.011 to +0.017) |

### 1. The lab's table is not the public sample

| Sessions matched on `(fullVisitorId, visitId, visitStartTime)` | Sessions | Purchase sessions | Revenue |
|---|---:|---:|---:|
| In both tables | 832,872 | 10,642 (identical in both) | $1.63M (identical within $40) |
| Only in the lab's table | 137,660 | 13,314 | $2.99M |
| Only in the public sample | 70,781 | 910 | $0.15M |

Where the tables overlap, they agree. The lab-only sessions include 36,968 from internal Google referrers (7,213 purchases, $1.59M) and 31,300 Direct sessions converting at 15.5%, which looks like employees returning by bookmark. The public sample appears to be a version of the data with internal referrers redacted and much internal traffic removed. Since the lab trains on its own table, the audit uses that table.

### 2. Ground truth for employee traffic

In the lab's table, the top Referral source is **`mall.googleplex.com`** (79,767 sessions, 7,556 purchases, $1.19M). Other internal hosts include Google's internal Sites discount page (`sites.google.com/a/google.com/…`, 1,967 purchases), `gdeals.googleplex.com` (1,853), and `moma.corp.google.com`, Google's intranet (572), plus about 30 small corp tools.

**Ground-truth rule:** a visitor is internal if any of their sessions in the lab's table came from `*.googleplex.com`, `*.corp.google.com`, other internal Google tool hosts, the employee perks site, or `sites.google.com/a/google.com/…`.

**Our Prepare-phase inference, checked against it** (public-sample visitors that also appear in the lab's table):

| | Visitors | Purchase sessions |
|---|---:|---:|
| Flagged, truly internal | 34,225 | 5,058 |
| Flagged, truly external | 291 | 11 |
| Not flagged, truly internal | 2,711 | 94 |
| **Precision / recall** | **99.2% / 92.7%** | **99.8% / 98.2%** |

Among flagged sessions that also appear in the lab's table, 249 came from `googleweblight.com`, a public Google mobile proxy that the public sample also redacts. That is one source of false positives. The misses are employees who never arrived through the redacted internal referral in the public sample. Together they are **about 1.5% of the purchases we treat as external**, too small to change any conclusion. The flag stays as it is (decision D-A1).

### 3. Who the lab learns from

| Lab split | First visits | Positives | Positive rate | **Positives who are employees** | Employees among all first visits |
|---|---:|---:|---:|---:|---:|
| Train (2016-08-01 – 2017-04-30) | 573,001 | 8,771 | 1.53% | **5,313 (60.6%)** | 5.9% |
| Eval (2017-05-01 – 2017-06-30) | 102,025 | 1,650 | 1.62% | **1,218 (73.8%)** | 9.4% |

(Prepare had estimated 53% from the public sample. The lab's own table, with ground truth, gives 61%.)

### 4. Replication

The lab's SQL ran unchanged (apart from the dataset name) in the sandbox project's `lab_audit` dataset:

| Model | Features | Published ROC-AUC | **Our BigQuery ML run** | scikit-learn replica |
|---|---|---:|---:|---:|
| `classification_model` | bounces, time on site | 0.72 | **0.724** | 0.751 |
| `classification_model_2` | + funnel progress, pageviews, source, medium, channel, device, country | 0.91 | **0.909** | 0.919 |

BigQuery ML reproduces both published values, so every headline number below comes from the lab's actual model. The scikit-learn replica fits to convergence where BigQuery ML's default optimizer stops early, which explains its gap on model 1. It's kept as a portable version (for example, for Kaggle) and shows the same pattern.

### 5. H4: the published ROC-AUC is inflated by employees

![ROC-AUC dumbbell](figures/lab_audit_auc_gap.png)

| Model (BigQuery ML) | ROC-AUC, all first visits | ROC-AUC, external only | Gap (paired bootstrap, 2,000 resamples) |
|---|---:|---:|---|
| Lab model, as published | 0.910 | 0.863 | **0.047** (95% CI 0.035 to 0.059); no resample ≤ 0 |
| Without source, medium, channel | 0.885 | 0.882 | 0.003 (95% CI −0.008 to +0.013); not significant |

**H4 is supported**, and the second row shows the mechanism: the internal referrers are values of `source`, so the traffic-source features let the model recognize employees.

### 6. Who the lab model would target

![Top-ranked visitors are employees](figures/lab_audit_top_ranked.png)

| Lab model's ranking | Visitors | Employees | Buyers | …of which external |
|---|---:|---:|---:|---:|
| Top 1% | 1,020 | **98.8%** | 158 | **2** |
| Top 5% | 5,101 | 87.9% | 818 | 43 |
| Top 10% | 10,202 | 76.8% | 1,153 | 112 |
| All first visits | 102,025 | 9.4% | 1,650 | 432 |

The model's score identifies employees (ROC-AUC **0.974**) better than it predicts purchases (0.910). A remarketing audience taken from its top ranks would be spent almost entirely on Google's own staff.

The two audit variants still put employees in 59–66% of their top 1%, because employees also browse deeply and add to cart. **Removing features isn't enough; employees must be removed from the population.**

### 7. How well each model ranks real customers

| Model (BigQuery ML), external visitors only | ROC-AUC | PR-AUC | Precision in top 1% | Lift in top 10% | ROC-AUC vs lab model (paired bootstrap) |
|---|---:|---:|---:|---:|---|
| Lab model, as published | 0.863 | 0.036 | 6.8% | 5.8× | — |
| Without source, medium, channel | 0.882 | 0.043 | 6.7% | 6.1× | **+0.019** (95% CI +0.007 to +0.031) |
| Trained on external visitors only | 0.877 | 0.038 | 6.4% | 6.2× | **+0.014** (95% CI +0.011 to +0.017) |

Employee contamination doesn't just inflate the published score; it **makes the model worse at ranking real customers**. Even at its best, more than 9 of every 10 visitors in the top 1% don't buy (base rate 0.47%). That's why D2 is judged on PR-AUC, lift, and a break-even cutoff, not ROC-AUC.

### Carried into Part B (D2)

1. **Population:** external visitors only. This is already the case in `remarketing_table`, and the flag is validated above.
2. **Evaluation:** PR-AUC and top-decile lift on external visitors in an out-of-time test, plus a break-even cutoff.
3. **Traffic-source features:** used only as channel groups fit on external data, and checked for stability.
4. **Baseline to beat:** the lab's feature set, refit on our corrected table and label, so the comparison uses the same population and the same 30-day label.

### Limitations

- **Ground truth depends on referrers.** Employees who never arrived through an internal referrer (for example, those who typed the URL) count as external in both tables, so the external-only results still contain some employees. The true inflation is probably *larger* than 0.047.
- **Google doesn't document how either table was produced.** The comparison shows *what* differs, not *why*.
- **The lab's label has no fixed time window**, so evaluation visitors from late June 2017 had less time to return. This affects all models equally, so the comparisons still hold.
- BigQuery ML models live in the sandbox project (`merch-store-capstone.lab_audit`). The notebook reads cached results unless `RUN_BQML = True`.

### Decision log

| ID | Decision | Why |
|---|---|---|
| D-A1 | Keep the inferred public-sample flag (Referral + `(direct)` + `/`) for D1 and D2; don't switch to the lab's table | Verified at 99.2% precision and 98.2% recall of employee purchases. What it misses is about 1.5% of external purchases. The public sample stays the documented, citable dataset |
| D-A2 | Audit the lab on **its own table** with the **referrer ground truth**, recreating its models in BigQuery ML with its own SQL | An exact replication (0.724 / 0.909 vs 0.72 / 0.91) makes the audit about the real model, not an approximation |
| D-A3 | Join model predictions on a fingerprint of all GROUP BY columns, not the lab's `unique_session_id` | The lab's ID isn't unique (sessions split at midnight). Joining on it duplicated 1,666 rows |

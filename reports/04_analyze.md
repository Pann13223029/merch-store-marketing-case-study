# Phase 4 — Analyze

- **Part A: Audit of Google's lab model for D2**
- **Part B: The corrected remarketing model and break-even audience (D2)**
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

---

## Part B — The corrected remarketing model (D2)

Notebook: [notebooks/04_analyze_remarketing.ipynb](../notebooks/04_analyze_remarketing.ipynb) · Code: [`src/modeling.py`](../src/modeling.py), [`src/breakeven.py`](../src/breakeven.py)

**Question:** which first-time visitors are worth paying to bring back?
**Population:** external first-time visitors who didn't buy on that visit.
**Label:** a purchase on a later visit within 30 days.
**Data:** train on first visits Aug 2016 – Apr 2017 (522,599 visitors, 1,433 later buyers, 0.27%); test once on May 1 – Jul 1, 2017 (91,431 visitors, 323 later buyers, 0.35%).

### Summary

| Finding | Evidence |
|---|---|
| **H1 supported:** the first visit's channel predicts a later purchase | χ² = 1,406, df = 6, p < 0.001. Paid Search 1.16% and Display 2.03% vs Social 0.008% (about 150–270×) |
| **H3 supported in part:** adding to cart matters; merely viewing a product doesn't, once engagement is controlled | Add-to-cart odds ratio **3.7** (95% CI 3.2–4.3); product view 1.16 (p = 0.06). North America **15.7**, mobile 0.36 |
| The model **clearly beats the funnel rule** (the Ask-phase success criterion) | PR-AUC **0.061 vs 0.028** (+0.032, 95% CI +0.019 to +0.048); top-decile lift **7.0× vs 5.4×** |
| It only **slightly beats the lab's features refit on corrected data** | PR-AUC +0.005 (not significant); top-10% precision +0.17 points (significant). **Fixing the data mattered more than the model** |
| **The top 10% of first visits hold 70% of later buyers**; the bottom 40% hold none | Cumulative gains on the test months |
| Results are **robust** | Embargoed training 0.058; without missed employees 0.061; well calibrated (predicted 0.34% vs observed 0.35%) |
| **Retargeting is worth little beyond the top 20%** | Max affordable cost per visitor: top 1% **$0.70**, 1–2% $0.18, 2–5% $0.11, 5–20% $0.07, below that < $0.01 |
| **The prize is modest** | Retargeting the top 10% (about 4,500 visitors/month) ≈ **$700/month** extra gross profit before ad cost |

### 1. Hypotheses (training months only)

**H1: return-purchase rate by first-visit channel**

![Return-purchase rate by channel](figures/h1_return_rate_by_channel.png)

| Channel | First visits | Later buyers (30 days) | Rate (95% CI) |
|---|---:|---:|---|
| Display | 1,332 | 27 | 2.03% (1.40–2.93%) |
| Paid Search | 10,736 | 125 | 1.16% (0.98–1.39%) |
| Direct | 74,248 | 467 | 0.63% (0.58–0.69%) |
| Organic Search | 213,473 | 773 | 0.36% (0.34–0.39%) |
| Referral | 16,441 | 23 | 0.14% (0.09–0.21%) |
| Affiliates | 9,143 | 3 | 0.03% (0.01–0.10%) |
| Social | 197,203 | 15 | 0.008% (0.005–0.013%) |

Chi-square test of independence: χ² = 1,406, df = 6, p < 0.001. One of 14 cells has an expected count below 5 (3.7), which is within Cochran's rule. Cramér's V = 0.05 looks negligible, but V is capped by the base rate when the outcome is this rare, so the relative risks above describe the effect better. **Social sends 38% of first-time visitors but only 1% of later buyers.**

**H3: product-level engagement on the first visit**

| First-visit engagement | Later purchase rate if yes | …if no | Ratio | One-sided z-test |
|---|---:|---:|---:|---|
| Viewed a product | 1.57% | 0.13% | 11.9× | z = 59, p < 0.001 |
| Added to cart | 3.59% | 0.17% | 21.1× | z = 81, p < 0.001 |

A multivariable logistic regression (522,576 first visits, McFadden pseudo-R² 0.27) holds engagement, channel, device, and region constant:

| Factor (vs reference) | Odds ratio (95% CI) |
|---|---|
| North America (vs rest of world) | **15.7** (12.4–19.8) |
| Added to cart | **3.7** (3.2–4.3) |
| Display / Direct / Paid Search (vs Organic Search) | 2.4 / 1.8 / 1.7 |
| Reached checkout | 1.9 (1.5–2.2) |
| log(pageviews), log(time on site) | 1.33, 1.23 per unit |
| Viewed a product | 1.16 (0.995–1.36), p = 0.06: **not significant** |
| Mobile / tablet (vs desktop) | 0.36 / 0.41 |
| Referral / Affiliates / Social (vs Organic Search) | 0.55 / 0.17 / 0.11 |

**H3 holds for adding to cart but not for merely viewing a product**, whose effect in the simple comparison came from general engagement.

### 2. Model development

- **Validation:** three expanding-window folds inside the training months. Each skips one month between training and validation, because training labels look 30 days ahead.
- **Selection rule, fixed before testing:** the highest mean validation PR-AUC.
- **Search:** 28 configurations. The first grid's best tree models sat at the grid's edge, so it was extended until the optima were interior.

| Candidate | Best settings | Validation PR-AUC | Per fold |
|---|---|---:|---|
| **Random forest** (chosen) | 300 trees, min leaf 10, √features | **0.0706** | 0.060 / 0.066 / 0.086 |
| Gradient boosting | lr 0.02, 4-leaf trees, min leaf 1,000, 300 rounds | 0.0701 | 0.059 / 0.068 / 0.083 |
| Logistic regression | L2, C = 0.01, log counts + one-hot | 0.0673 | 0.060 / 0.065 / 0.077 |
| *Baseline: lab's features, refit* | GSP229 features, unregularized logistic | *0.0560* | |
| *Baseline: funnel rule* | Furthest funnel step, then pageviews | *0.0277* | |

Boosting improves as trees get *smaller* down to 4 leaves, while 2-leaf stumps are worse. The signal is **mostly additive with small interactions**, which is why logistic regression comes close. The random forest and boosting models are tied within fold noise. The rule picks the forest, and logistic regression is kept as the **explainer** (its odds ratios are in §1).

### 3. Test results (evaluated once)

| Model | PR-AUC (95% CI) | ROC-AUC | Precision, top 1% | Lift, top 1% / 10% | Later buyers in top 10% |
|---|---|---:|---:|---:|---:|
| Funnel rule | 0.028 (0.021–0.042) | 0.806 | 4.3% | 12.1× / 5.4× | 54% |
| Lab's features, refit | 0.055 (0.042–0.077) | 0.901 | 8.3% | 23.5× / 6.5× | 65% |
| Gradient boosting | 0.062 (0.048–0.086) | 0.902 | 9.6% | 27.3× / 6.8× | 68% |
| Logistic regression | 0.063 (0.048–0.085) | 0.912 | 8.9% | 25.1× / 7.0× | 70% |
| **Random forest (chosen)** | **0.061** (0.047–0.082) | 0.912 | 9.6% | 27.3× / **7.0×** | **70%** |

| Paired comparison (bootstrap, same test visitors) | Difference (95% CI) |
|---|---|
| Random forest vs funnel rule, PR-AUC | **+0.032** (+0.019 to +0.048) |
| Random forest vs funnel rule, precision in top 10% | **+0.57 points** (+0.42 to +0.77) |
| Random forest vs lab's features refit, PR-AUC | +0.005 (−0.008 to +0.018): not significant |
| Random forest vs lab's features refit, precision in top 10% | +0.17 points (+0.04 to +0.30) |

![Cumulative gains](figures/d2_gains_chart.png)

The three model families are tied on the test months, and the validation winner isn't the test winner, which is what noise looks like when the real differences are small. **The large gain came from correcting the population and the label** (Part A and Process). The model mainly adds a sharper top of the ranking.

### 4. Robustness

| Check | Test PR-AUC | Lift, top 10% |
|---|---:|---:|
| Main result | 0.061 | 7.0× |
| Trained only on first visits through 2017-03-31, so no training label overlaps the test months | 0.058 | 6.8× |
| Without the 298 test visitors the lab's table identifies as employees our flag missed | 0.061 | 7.0× |

Calibration: mean predicted 0.34% vs observed 0.35% (Brier 0.0034). Predicted and observed rates agree by decile, and **the bottom 40% of scored visitors had no later buyers.**

### 5. What drives the score

Permutation importance (drop in test PR-AUC when an input is shuffled): **country 0.013, sub-continent 0.010**, furthest funnel step 0.007, **operating system 0.007**, add-to-cart 0.005, channel 0.004, device 0.003, time on site 0.003. Correlated engagement counts share credit, so each looks small alone.

The operating-system signal is real. Within North America, Mac first visits come back to buy at 1.5% and Chrome OS at 1.2%, against 0.7% on Windows and 0.3–0.4% on phones. The employees our flag missed lean toward Macs but are only about 1% of buyers, too few to explain it.

### 6. Who would be in the audience (ethics check)

| Top 10% by score vs all first-time visitors | Top 10% | All |
|---|---:|---:|
| Northern America | 91% | 42% |
| Desktop | 82% | 62% |
| Paid Search / Direct | 12% / 32% | 4% / 22% |
| Social | 1% | 6% |

The audience follows purchase behavior: visitors outside North America almost never come back to buy, perhaps because of shipping or pricing, which the data can't show. In practice it's a **geographically narrow audience**, and the recommendation should say so openly. The low mobile share points to a mobile-experience question for the Act phase.

### 7. Break-even (decision D-B2)

> Remarketing pays off when revenue per visitor (next 30 days) × incremental lift × gross margin ≥ cost per visitor.

Revenue per visitor is **measured**: what each score band actually spent in the 30 days after the first visit, in the test months, capped per visitor at $1,606. Central assumptions are **10% lift** and **50% margin**. The lift is anchored on randomized experiments: a display-retargeting campaign lifted purchases 10.5% ([Johnson, Lewis & Nubbemeyer 2017](https://journals.sagepub.com/doi/abs/10.1509/jmr.15.0297)), and the median conversion lift across 432 Google Display Network experiments was 8% ([Johnson, Lewis & Nubbemeyer](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2701578)). Margin isn't public, so it's varied from 30% to 70%.

![Break-even by band](figures/d2_breakeven_by_band.png)

| Score band | Visitors/month | 30-day buy rate | Revenue per visitor (95% CI) | Max affordable cost/visitor: central (range across lift 5–20%, margin 30–70%) |
|---|---:|---:|---|---|
| Top 1% | 456 | 9.6% | $14.07 ($8.93–$20.62) | **$0.70** ($0.21–$1.97) |
| 1–2% | 456 | 3.6% | $3.59 ($1.68–$5.94) | $0.18 ($0.05–$0.50) |
| 2–5% | 1,367 | 2.0% | $2.24 ($1.14–$3.78) | $0.11 ($0.03–$0.31) |
| 5–10% | 2,278 | 1.1% | $1.35 ($0.71–$2.30) | $0.07 ($0.02–$0.19) |
| 10–20% | 4,557 | 0.58% | $1.39 ($0.74–$2.17) | $0.07 ($0.02–$0.19) |
| 20–50% | 13,670 | 0.14% | $0.09 | < $0.01 |
| Bottom 50% | 22,783 | 0.01% | $0.01 | ≈ $0 |

| Target | Visitors/month | Later buyers reached | Extra gross profit/month before ad cost (central) |
|---|---:|---:|---:|
| Top 1% | 456 | 27% | $320 |
| Top 5% | 2,279 | 54% | $555 |
| Top 10% | 4,557 | 70% | $709 |
| Top 20% | 9,113 | 86% | $1,025 |

### What this means for D2 (carried to Act)

1. **Retarget only the top-scored first visits**, and include a band only if the actual cost per retargeted visitor is below that band's affordable cost.
2. **Keep the program small and cheap.** The whole top-10% audience is worth about $700/month in gross profit before ad cost, so the larger budget lever is probably channel mix (D1).
3. **Measure the real lift** with a randomized holdout (for example 10% of the audience not retargeted). The 10% lift is borrowed from the literature, not measured on this store.

### Limitations

- **Observational data.** Purchase rates reflect business as usual, including whatever retargeting the store already ran, which we can't see. Only a holdout test can measure true lift.
- **Assumptions.** Lift and margin are assumptions, shown with sensitivity ranges. Revenue per visitor in the top bands has wide confidence intervals (a few hundred buyers).
- **Short test period.** The test covers May–June 2017 only, so seasonality (for example the holiday season) isn't represented.
- **Cookie-based identity.** A visitor returning on another device looks like a non-returner, so return purchases are undercounted for everyone.

### Decision log

| ID | Decision | Why |
|---|---|---|
| D-B1 | Model selection by highest mean validation PR-AUC, fixed before testing, so the random forest is chosen. Logistic regression kept as the explainer | Avoids choosing on the test set. The families are statistically tied, so interpretability comes from the logistic model |
| D-B2 | Break-even reported as the **maximum affordable cost per visitor** by score band. Central 10% lift and 50% margin; sensitivity 5–20% and 30–70% | User's decision: no invented cost figure. The Head of Marketing compares with real costs |
| D-B3 | Tuning grid extended until tree-model optima were interior (28 configurations) | A best result at a grid's edge may not be the true optimum |
| D-B4 | H1 and H3 tested on training months only | Keeps the test months untouched for the single final evaluation |

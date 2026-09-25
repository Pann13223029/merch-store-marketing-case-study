# Phase 1 — Ask

## Business task

> **The Google Merchandise Store's Head of Marketing must decide how to allocate next quarter's budget.**
> Identify which channels actually *create* buyers compared with what last-click reporting credits them for, and define a remarketing audience of first-time visitors worth paying to bring back.

The deliverable is a **budget recommendation**, not a model. Every analysis below exists to support one of two decisions:

| Decision | Question it answers | Output |
|---|---|---|
| **D1 — Channel budget** | Which channels are over- or under-credited by last-click reporting? | Channel credit under 5 attribution models → % budget shift per channel, under stated cost assumptions (replaced by test designs; see the note under the success criteria) |
| **D2 — Remarketing audience** | Which first-time visitors are worth paying to bring back? | Audit of Google's own lab model for this question → corrected propensity model → break-even cutoff → audience size and expected revenue |

## Stakeholders

| Stakeholder | Role in this project | What they need from it |
|---|---|---|
| **Head of Marketing** (primary) | Owns the total budget; makes both decisions | A clear recommendation with $ impact and confidence |
| Paid media manager | Executes channel spend (Paid Search, Display, Affiliates) | Which channels to scale up or down |
| CRM / remarketing lead | Runs remarketing audiences in Google Ads | Audience definition, size, and a score cutoff |
| Finance | Approves budget shifts | Assumptions stated explicitly; ranges, not point estimates |
| Web analytics team | Owns GA tracking | Data caveats and tracking fixes (Act phase) |

## Guiding questions

**D1 — Channel budget**
1. How do channels differ in reach (sessions), conversion rate, and revenue?
2. What share of buyers take more than one visit, and which channels typically *start* versus *close* those journeys?
3. How does each channel's credit change from **last-click** to **first-touch, linear, position-based, and data-driven (Markov chain)** attribution?
4. Given plausible cost assumptions, which reallocation would most increase attributed revenue — and how sensitive is that to the assumptions?

**D2 — Remarketing audience (starting from Google's lab)**

Google's own lab, *Predict Visitor Purchases with a Classification Model in BigQuery ML* (GSP229), models this question with the label `will_buy_on_return_visit`, reaching 0.91 ROC-AUC. D2 audits that model and then builds a version ready for a real decision.

5. **Audit:** How much of the lab model's performance depends on Google employees, who make up 61% of its training positives (verified on the lab's own table)?
6. What first-visit signals (channel, device, geography, engagement, product interest) predict an external visitor's purchase on a *later* visit within 30 days?
7. How well does the corrected model rank visitors compared with a simple rule (e.g. "viewed a product page") and with the lab model, measured by PR-AUC and top-decile lift instead of ROC-AUC?
8. At what score cutoff does remarketing break even, and how many visitors and how much revenue does that audience cover?

## Hypotheses (formally tested in Analyze)

| # | Hypothesis | Test |
|---|---|---|
| H1 | Return-visit purchase rate differs by first-visit channel | Chi-square test of independence (+ Cramér's V for effect size; relative risks reported instead, see the decision log) |
| H2 | Last-click reporting under-credits channels that start journeys (e.g. Organic Search, Social) relative to data-driven attribution | Markov removal-effect credit vs last-click credit, with bootstrap 95% CIs |
| H3 | First visits with product-level engagement (product view / add-to-cart) are more likely to lead to a later purchase | Two-proportion z-test; logistic regression coefficient with odds ratio |
| H4 | The lab model's ROC-AUC is lower on external visitors than on all visitors, because part of its skill is recognizing internal traffic | Same model and eval set, scored with and without internal visitors; bootstrap 95% CI of the AUC difference |

## Success criteria

- **D1:** Attribution credit per channel reported with 95% confidence intervals; recommendation states the % shift, the expected revenue effect, and the cost assumption it depends on.
- **D2:** Model beats the simple-rule baseline on **PR-AUC** (the right metric at < 1% positives) and **top-decile lift**; break-even cutoff reported with the audience size and expected revenue it captures.
- **D2 audit:** Lab model replicated (ROC-AUC within ±0.02 of the published 0.91), then compared with the corrected model on the same external-only test split.
- **Overall:** A non-technical reader can act on the executive summary without reading the notebooks.

**Note, added after Analyze:** the D1 criterion's "% shift" isn't delivered. The data has no ad costs, and credited revenue isn't incremental: attribution shows which channels were on the path to a sale, not what extra spend would buy. A shift computed from credit could move money toward channels that didn't cause the sales. The project replaces it with test designs that would measure what the channels cause ([notebook 06](../notebooks/06_test_design.ipynb); Analyze, Part D).

## First look at the data (sanity check, 2026-09-24)

| Metric | Value |
|---|---|
| Date range | 2016-08-01 → 2017-08-01 (366 days) |
| Sessions / visitors | 903,653 / 714,167 |
| Purchase sessions | 11,552 (1.28% conversion) |
| Buyers / revenue | 10,022 / $1.78M |
| Buyers with multi-session journeys | 6,895 (69%) → attribution is worth doing |
| New visitors buying on a *later* visit vs first visit | 4,650 vs 4,436 (all traffic). **Corrected in Prepare:** excluding likely-internal traffic, it is 2,154 vs 2,893, so 43% of external buyers come back before buying |

| Channel | Sessions | Conv. rate | Revenue |
|---|---:|---:|---:|
| Referral | 104,838 | 5.08% | $717,600 |
| Direct | 143,026 | 1.44% | $498,530 |
| Organic Search | 381,561 | 0.90% | $377,076 |
| Display | 6,262 | 2.28% | $130,337 |
| Paid Search | 25,326 | 1.85% | $47,543 |
| Social | 226,117 | 0.05% | $8,397 |
| Affiliates | 16,403 | 0.05% | $654 |

**Open questions this raises for Prepare:** Referral converts at 4× the site average. Is that internal traffic from Google employees rather than marketing? Social brings 25% of sessions but almost no buyers. Is that YouTube traffic with a different purpose? Display earns $911 per purchase session. Are those bulk/corporate orders?

## Known limitations and assumptions

- **No cost data.** Attribution shows credit, not ROI. Budget advice is given under explicit cost scenarios with sensitivity ranges.
- **Cookie-based identity.** `fullVisitorId` is per browser/device, so cross-device journeys are split and multi-visit paths are undercounted.
- **Window edges.** Journeys that began before 2016-08-01 are cut off (12,544 visitors arrive with `visitNumber > 1`). Visitors who arrive near 2017-08-01 have no time to return, so D2 uses a fixed follow-up window and excludes late arrivals.
- **Legacy schema.** The data comes from Universal Analytics (GA360), which was retired in 2023. The methods carry over to the GA4 BigQuery export. The Act phase will note the mapping.
- **Public sample.** Google publishes this as a sample; it may not be the store's complete traffic.

## Ethics and privacy

- The data is anonymized by Google, with no PII. Visitor IDs are hashed.
- Remarketing requires user consent (GDPR / Consent Mode). The recommendation assumes only consented users are targeted.
- Audience rules based on geography or device can exclude groups unintentionally. The D2 audience will be checked for how it is composed across countries and devices.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-09-24 | Own-dataset track (not Cyclistic/Bellabeat) | Stronger portfolio differentiation |
| 2026-09-24 | Google Merchandise Store (GA360 sample, BigQuery) | Covers sales, e-commerce, and marketing; ties to the Google certificate and BigQuery |
| 2026-09-24 | Headline is a budget decision, not "will a visitor buy?" | The dataset is heavily used (Google's own BQML lab, 2018 Kaggle GStore competition). The original work is in the question |
| 2026-09-24 | Primary stakeholder: Head of Marketing | Owns both decisions (D1 + D2) |
| 2026-09-25 | **Correction:** D2 overlaps with Google's GSP229 lab (`will_buy_on_return_visit`), contrary to the original plan. The planned "end-of-session leakage" critique was also wrong, because the lab's features are available after the first visit | Verified against the lab text; stated from memory at first |
| 2026-09-25 | Reposition D2 as **audit and fix the lab**: replicate it, show that 53% of its training positives are likely employees, test the effect on AUC (H4), rebuild external-only with a 30-day window and PR-AUC | Turns the overlap into a critique that can be checked, rather than a lookalike |
| 2026-09-25 | **Correction:** the lab trains on `data-to-insights.ecommerce.web_analytics`, not the public sample. The audit was re-based on that table, where referrers are unredacted: **61%** of training positives are verified employees (the 53% was a public-sample estimate) | Checked against the lab's SQL and the table itself (reports/04_analyze.md, Part A) |
| 2026-09-25 | **Deviation from the plan:** H1's effect size is reported as relative risks, not Cramér's V. The switch was made after the test was run | With a 0.3% outcome, V stays near zero (0.05) even though the channels' rates differ by up to 270× (Analyze, Part B), so V doesn't describe the effect |
| 2026-09-25 | **Change of deliverable:** D1's "% budget shift per channel" is dropped and replaced by test designs (notebook 06) | There's no cost data, and credited revenue isn't incremental, so a shift computed from it would rest on numbers that don't measure what extra spend buys |

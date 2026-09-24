# Phase 3 — Process

## Pipeline

```
BigQuery: ga_sessions_* (903,653 sessions, nested hits)
   │  sql/process/01_sessions_clean.sql         0.81 GB scanned, run once, cached
   ▼
data/raw/sessions_clean.parquet                 one row per session, 32 columns, 15/15 checks pass
   │
   ├── src/tables.py:remarketing_table()  → data/processed/remarketing_table.parquet  (D2 model)
   └── src/journeys.py                    → data/processed/touches.parquet, journeys.parquet  (D1)
```

Heavy work (unnesting hits, the visitor-level internal flag) runs in BigQuery. Everything after that runs in pandas on the cached extract, so reruns cost nothing.

## Cleaning steps (sql/process/01_sessions_clean.sql)

| Step | Rule | Source |
|---|---|---|
| Session key | `fullVisitorId-visitId-visitStartTime` (midnight-split sessions share `visitId`) | Prepare finding 4 |
| Internal flag | Visitor-level: any visitor who ever entered via Referral + source `(direct)` + path `/` | D-P1, D-P1a |
| NULL totals | `pageviews`, `timeOnSite`, `bounces`, `transactions`, `revenue` NULL → 0 (the GA360 export omits zeros) | Prepare finding 3 |
| Excluded fields | City (56% redacted), campaign (97% not set), keyword (25% not provided) | Prepare finding 3 |
| Shopping behavior | From hit-level Enhanced Ecommerce actions: product detail views, distinct products viewed, add-to-cart, checkout, furthest funnel step | New |
| Revenue | `totalTransactionRevenue / 1e6` (USD, including tax and shipping) | GA360 schema |

## Validation (src/validate.py): 15 of 15 pass

Every check compares the cleaned extract against totals measured on the raw tables in Prepare.

| Check | Result |
|---|---|
| Rows / unique session keys | 903,653 / 903,653 |
| Purchase sessions / revenue | 11,552 / $1,780,149 (exact match) |
| Internal visitors / sessions / purchases / revenue | 37,332 / 76,536 / 5,408 / $730,376.50 |
| Internal flag constant within each visitor | ✅ |
| No NULLs or negatives in engagement and outcome columns | ✅ |
| `is_new_visit` agrees with `visit_number == 1` | 0 mismatches |
| Purchase sessions reach the purchase step in hit data | 100% |
| Revenue only on purchase sessions; purchases without revenue | ✅; 37 sessions |

Two checks failed on the first run. Both were investigated and neither is a data error:
- **Internal revenue was off by $1.** The exact value is $730,376.50. BigQuery rounds half up and Python rounds half to even. The check now allows $1 of tolerance.
- **2 of 450,630 bounced sessions have time on site.** Each is one pageview plus a non-interaction event, which GA counts as a bounce but still gives a duration. The check now allows under 0.01% exceptions.

## Finding: GA already shifts credit before any attribution model runs

When a visitor returns directly (typed URL or bookmark), GA360 labels the session with their **previous campaign** and sets `isTrueDirect = true`. Among external sessions:

- **96,809 sessions (11.7%)** carry a non-Direct channel label but were direct return visits (all have `visit_number > 1`).
- **1,790 of 6,144 external purchases (29%)** happened on these relabelled sessions.

Journeys built from `channelGrouping` would already contain GA's last-non-direct-click logic. **Decision D-PR1:** journeys use the **arrival channel** (isTrueDirect → Direct). The last-click baseline keeps **GA's labels**, which is what the store's dashboards showed. A **conservative** relabel (only sessions whose source is literally `(direct)`) runs as a sensitivity check.

## Analysis tables

### D2 audit: Google's GSP229 lab

The lab trains on a different table (`data-to-insights.ecommerce.web_analytics`), so its audit uses that table directly, with the lab's own SQL and BigQuery ML. See [reports/04_analyze.md](04_analyze.md), Part A. An earlier public-sample version of the lab's table was removed: it was an estimate on the wrong data, and it grouped rows slightly differently from the lab's SQL.

### D2 model: corrected remarketing table (`remarketing_table`)

| Rule | Why |
|---|---|
| External visitors only | Marketing can't remarket to employees |
| First visit did **not** purchase | Remarketing targets visitors who left without buying |
| Label = purchase on a **later** visit within **30 days** | A fixed window matching the remarketing decision; captures 95% of purchases (Prepare finding 6) |
| First visit on or before 2017-07-01 | Every 30-day window is fully inside the data |
| Same train/test months as the lab | So the models can be compared directly |

| Split | Visitors | Buyers within 30 days | Rate | 30-day revenue |
|---|---:|---:|---:|---:|
| Train (2016-08-01 – 2017-04-30) | 522,599 | 1,433 | 0.27% | $239,575 |
| Test (2017-05-01 – 2017-07-01) | 91,431 | 323 | 0.35% | $53,545 |

Edge case: 608 visitors (0.09%) have two separate visits with `visit_number = 1`; the earliest is kept.

### D1: journeys (`touches`, `journeys`)

| Rule | Why |
|---|---|
| A journey ends at a purchase or after a gap of more than 30 days | Separates distinct buying cycles |
| Converting journeys keep touches within 30 days before the purchase | GA's default lookback; covers 95% of purchases |
| Analysis period: journeys ending 2016-08-31 – 2017-07-01 | A full 30-day lookback and 30 days of observed follow-up for every journey |
| Revenue capped per order at $1,606 (99th percentile of external orders) | D-P2 |

| Metric | Value |
|---|---:|
| Journeys | 580,775 |
| Converting journeys | 5,074 (matches external purchases in period ✅) |
| Converting journeys with more than one touch | 42.5% |
| Converting journeys whose arrival path differs from GA's labels | 1,514 (29.8%) |
| Revenue: raw vs capped | $850,184 vs $683,533 |

**Corporate demand signal:** the top 1% of orders (51 orders over $1,606) carry **29.2%** of external revenue in the period ($248,552 of $850,184). Capping removes the $166,651 (19.6%) above the cap. These orders are reported separately in Analyze.

## Decision log

| ID | Decision | Why |
|---|---|---|
| D-PR1 | Journeys use the arrival channel (isTrueDirect → Direct); last-click baseline uses GA labels; conservative relabel as sensitivity | GA's session labels already carry last-non-direct-click credit |
| D-PR2 | D2 population = external first-visit non-buyers; label = later-visit purchase within 30 days; first visits on or before 2017-07-01 | Matches the remarketing decision; removes internal traffic and window bias |
| D-PR3 | D2 train/test split uses the lab's months (train through 2017-04-30, test 2017-05 – 2017-06 plus 2017-07-01) | Lets the models be compared directly; out-of-time test |
| D-PR4 | Journeys: 30-day inactivity or purchase ends a journey; 30-day lookback; period 2016-08-31 – 2017-07-01 | Complete lookback and follow-up for every journey |

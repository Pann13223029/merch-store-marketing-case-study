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
| Excluded fields | City (56% redacted), campaign (97% not set), keyword (25% not provided). Keyword and campaign stay out of `sessions_clean`; the Paid Search keyword split reads them from the separate p09 extract | Prepare finding 3 |
| Shopping behavior | From hit-level Enhanced Ecommerce actions: product detail views, distinct products viewed, add-to-cart, checkout, furthest funnel step | New |
| Revenue | `totalTransactionRevenue / 1e6` (USD, including tax and shipping) | GA360 schema |

## Validation (src/validate.py): 15 of 15 pass

Five checks compare the cleaned extract against totals measured on the raw tables in Prepare (rows, purchase sessions, revenue, internal visitors, internal sessions/purchases/revenue). The other ten are internal-consistency checks.

| Check | Result |
|---|---|
| Rows / unique session keys | 903,653 / 903,653 |
| Purchase sessions / revenue | 11,552 / $1,780,149 (no saved Prepare query gives the unrounded revenue total; the p02 segment totals, rounded to the dollar, add up to $1,780,150) |
| Internal visitors / sessions / purchases / revenue | 37,332 / 76,536 / 5,408 / $730,376.50 |
| Internal flag constant within each visitor | ✅ |
| No NULLs or negatives in engagement and outcome columns | ✅ |
| `is_new_visit` agrees with `visit_number == 1` | 0 mismatches |
| Purchase sessions reach the purchase step in hit data | 100% |
| Revenue only on purchase sessions; purchases without revenue | ✅; 37 sessions |

Two checks failed on the first run. Both were investigated and neither is a data error:
- **Internal revenue was off by $1.** The exact value is $730,376.50. BigQuery rounds half up and Python rounds half to even. The check now allows $1 of tolerance.
- **2 of 450,630 bounced sessions have time on site.** Each is one pageview plus a non-interaction event, which GA counts as a bounce but still gives a duration. The check now allows under 0.01% exceptions.

**What these checks can and can't catch.** The raw totals come from Prepare queries that use the same expressions as the cleaning SQL (the same internal-flag rule, the same revenue field). So the checks guard against dropped or duplicated rows, not against a wrong rule: a mistake in the flag's definition would appear on both sides and still pass. The logic is covered by the unit tests in [`tests/`](../tests/), which run on small synthetic tables with known answers: journey splitting, the 30-day lookback, the analysis period and the revenue cap, the D2 population and label, the segments, and the attribution rules. The flag-constancy and no-NULL checks hold by construction on the SQL output (a per-visitor window and `IFNULL`). They guard the cached extract and later SQL edits, not the SQL as written.

## Finding: GA already shifts credit before any attribution model runs

When a visitor returns directly (typed URL or bookmark), GA360 labels the session with their **previous campaign** and sets `isTrueDirect = true`. Among external sessions:

- **96,809 sessions (11.7%)** carry a non-Direct channel label but are flagged by GA as direct returns (all have `visit_number > 1`). GA sets `isTrueDirect` for typed or bookmarked returns, and also when two visits in a row carry identical campaign details, such as a repeat Google search. The export can't tell these apart, so Analyze (Part C, §7) treats the relabel as the top of a range.
- **1,790 of 6,144 external purchases (29%)** happened on these relabeled sessions.

Without the key account (Analyze, Part C), the figures are 96,579 of 826,839 sessions (11.7%) and 1,775 of 6,128 purchases (29.0%).

Journeys built from `channelGrouping` would already contain GA's last-non-direct-click logic. **Decision D-PR1:** journeys use the **arrival channel** (isTrueDirect → Direct). The last-click baseline keeps **GA's labels**, which is what the store's dashboards showed. A **conservative** relabel (only visits whose source reads `(direct)`) was planned as a sensitivity check. Analyze (Part C, §7) found that the `(direct)` form marks the export day, not the visit: on 142 days the export wrote `(direct)` for nearly every Organic and Paid Search session, fresh clicks included. So the conservative relabel is the full relabel applied on those days. It's kept only as one more rule in the value-per-click range, not as corroboration.

## Analysis tables

### D2 audit: Google's GSP229 lab

The lab trains on a different table (`data-to-insights.ecommerce.web_analytics`), so its audit uses that table directly, with the lab's own SQL and BigQuery ML. See [reports/04_analyze.md](04_analyze.md), Part A. An earlier public-sample version of the lab's table was removed: it was an estimate on the wrong data, and it grouped rows slightly differently from the lab's SQL.

### D2 model: corrected remarketing table (`remarketing_table`)

| Rule | Why |
|---|---|
| External visitors only | Marketing can't remarket to employees |
| First visit did **not** purchase | Remarketing targets visitors who left without buying |
| Label = purchase on a **later** visit within **30 days** | A fixed window matching the remarketing decision; captures 88.5% of later-visit purchases (1,900 of 2,146; Prepare finding 7) |
| First visit on or before 2017-07-01 | Every 30-day window is fully inside the data |
| Same train/test months as the lab | So the models can be compared directly |

| Split | Visitors | Buyers within 30 days | Rate | 30-day revenue |
|---|---:|---:|---:|---:|
| Train (2016-08-01 – 2017-04-30) | 522,599 | 1,433 | 0.27% | $239,575 |
| Test (2017-05-01 – 2017-07-01) | 91,431 | 323 | 0.35% | $53,545 |

Edge case: 608 visitors (0.09%) have two separate visits with `visit_number = 1`; the earliest is kept.

The external flag here is the whole-year visitor flag, which uses hindsight: some first-time visitors are marked as employees only by a later session. `remarketing_table(sessions, internal_flag="first_visit")` builds the version a live campaign would score, evaluated in Analyze (Part B, §8).

### D1: journeys (`touches`, `journeys`)

| Rule | Why |
|---|---|
| Google employees and the key account are left out | Both are reported as their own segments (D-P1, D-PR5) |
| A journey ends at a purchase or after a gap of more than 30 days | Separates distinct buying cycles |
| Converting journeys keep touches within 30 days before the purchase | GA's default lookback; 95% of new outside buyers make their first purchase within 30 days of their first visit (Prepare finding 6). GA's report itself reaches back up to 6 months |
| Analysis period: journeys ending 2016-08-31 – 2017-07-01 | A full 30-day lookback and 30 days of observed follow-up for every journey |
| Revenue capped **per purchase session** at $1,606: the 99th percentile of outside purchase sessions in the period, with the key account included | D-P2. A session can hold several transactions |

| Metric | Value |
|---|---:|
| Journeys | 580,759 |
| Converting journeys | 5,058 (matches outside purchase sessions in the period, key account excluded ✅) |
| Converting journeys with more than one touch | 42.4% |
| Converting journeys whose arrival path differs from GA's labels | 1,498 (29.6%) |
| Revenue: raw vs capped | $721,771 vs $664,824 |

The most common arrival paths of converting journeys are a single Organic Search visit (1,389), a single Direct visit (1,203), Organic Search > Direct (443) and Direct > Direct (440).

**Midnight-split sessions.** Like GA, the journeys count a session split at midnight as two touches (658 such pairs among the journey touches). The D2 first-visit table merges the two halves, because a first visit must be one row and a purchase in the second half must not count as a later visit. Counting both halves here adds 0.2% to paid clicks (Paid Search 17,575 instead of 17,545), which lowers the value per click by about as much, so the bid caps are slightly cautious: with the halves merged, the Paid Search ceiling would read $1.22 instead of $1.21.

**Bulk purchase sessions:** purchase sessions above the $1,606 cap hold **17.2%** of the D1 revenue ($124,395 of $721,771), and capping removes $56,947 (7.9%). With the key account included they hold 29.2% of outside revenue in the period ($248,552 of $850,184), half of it that one account. They're reported in Analyze (Part C, §1).

## Decision log

| ID | Decision | Why |
|---|---|---|
| D-PR1 | Journeys use the arrival channel (isTrueDirect → Direct); last-click baseline uses GA labels. The conservative relabel was planned as a sensitivity check; since its `(direct)` form turned out to mark export days, it's kept only as one more rule in the value-per-click range | GA's session labels already carry last-non-direct-click credit |
| D-PR2 | D2 population = external first-visit non-buyers; label = later-visit purchase within 30 days; first visits on or before 2017-07-01 | Matches the remarketing decision; removes internal traffic and window bias |
| D-PR3 | D2 train/test split uses the lab's months (train through 2017-04-30, test 2017-05 – 2017-06 plus 2017-07-01: 62 days, so per-month figures divide by 62 / 30.4 months) | Lets the models be compared directly; out-of-time test |
| D-PR4 | Journeys: 30-day inactivity or purchase ends a journey; 30-day lookback; period 2016-08-31 – 2017-07-01 | Complete lookback and follow-up for every journey |
| D-PR5 | Leave the key account (visitor `1957458976293878100`, [`src/segments.py`](../src/segments.py)) out of the D1 journeys and report it as its own segment, like employees. Keep the revenue cap at $1,606, as set with the account included | One outside buyer held 89% of Display's GA-credited revenue and was already buying before its only Display click (Analyze, Part C). Without it the 99th percentile would fall to $1,506; keeping the cap leaves every other purchase session capped as before |

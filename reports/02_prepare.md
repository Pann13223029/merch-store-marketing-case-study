# Phase 2 — Prepare

## Where the data lives and how it's organized

| | |
|---|---|
| Source | `bigquery-public-data.google_analytics_sample.ga_sessions_*`, Google Analytics 360 export from the Google Merchandise Store, published by Google as a BigQuery public dataset |
| Layout | 366 daily sharded tables (`ga_sessions_20160801` … `ga_sessions_20170801`) with one row per session |
| Structure | Nested and repeated fields: `totals`, `trafficSource`, `device`, `geoNetwork` are structs; `hits` is an array of pageviews and events, each containing a `product` array. Behavior inside a session requires `UNNEST(hits)` |
| Access | Queried from a BigQuery sandbox project (`merch-store-capstone`) through the Python client. The profiling queries below scanned 0.01–0.09 GB each |
| Reproducibility | Every query is saved in [`sql/prepare/`](../sql/prepare/). Results are cached to `data/raw/*.parquet` (gitignored and rebuildable from the SQL) |

## Credibility (ROCCC)

| Criterion | Rating | Evidence |
|---|---|---|
| **Reliable** | ⚠️ Medium | First-party tracking of Google's own store, but it is a *sample* with redacted fields, and 8.5% of sessions are likely internal traffic (below) |
| **Original** | ✅ High | First-party data collected by GA360 on the store itself; not resold or aggregated |
| **Comprehensive** | ⚠️ Medium | 12 months of session- and hit-level data with no tracking gaps. **But:** no marketing cost data, campaign names missing on 96.9% of sessions, city redacted on 56%, identity is per-device cookie |
| **Current** | ⚠️ Low | 2016–2017, on the Universal Analytics schema that was retired in 2023. Behavior patterns may have changed, but the methods carry over to GA4 |
| **Cited** | ✅ High | Published and documented by Google (GA360 BigQuery export schema), and widely used in Google training material |

**Verdict:** credible enough for a *methods-and-decision* case study, as long as the internal-traffic distortion is removed and the limitations are stated alongside every recommendation.

## Key findings

### 1. About 41% of revenue comes from likely internal (Google employee) traffic

Sessions labelled `Referral` with source `(direct)` and referral path `/` have their referring site redacted. **65.8% of them come from Google office cities** (Mountain View, Sunnyvale, San Francisco, New York…), and they convert at 7.61%, which is 10× the external rate. This points to Google's internal employee store link. **Verified in Analyze (Part A):** Google's lab table, `data-to-insights.ecommerce.web_analytics`, keeps the referrer unredacted, and the same sessions there come from `mall.googleplex.com` (61,252) and `moma.corp.google.com` (418). Against that ground truth, the flag has **99.2% precision** and catches **98.2% of employee purchases**.

Flagged at the **visitor level** (decision D-P1):

| Segment | Visitors | Sessions | Purchase sessions | Conv. rate | Revenue |
|---|---:|---:|---:|---:|---:|
| Internal (likely employees) | 37,332 (5.2%) | 76,536 (8.5%) | 5,408 (**47%**) | 7.07% | $730,377 (**41%**) |
| External | 676,835 | 827,117 | 6,144 | 0.74% | $1,049,773 |

Flagging whole visitors (not just the entry session) also removes **3,023 Direct sessions** of flagged employees. Most of them (2,798) came *before* the visitor's first visit through the internal link, and 225 came after it, as returns by bookmark or typed URL. Session-level flagging would have left all of them inflating the Direct channel. Flagging a visit by what the visitor did later uses hindsight: fine for reporting on the past, but a live campaign can't see it when it scores a first visit (Analyze, Part B, §8).

*Other redacted-referral paths* (`/offer/2145`, `/pagead/ads`, `/cm/CampaignMgmt`…; 1,697 sessions, $2.5k revenue) turned out, in the lab's unredacted table, to be Google-internal too: the employee deals site `gdeals.googleplex.com`, ad-preview tools, and corp dashboards. They stay as `Referral` because they're too small to change any conclusion. All employee traffic the flag misses together is about 1.5% of the purchases treated as external (Analyze, Part A).

### 2. No tracking gaps across the year

Monthly sessions range from 62k to 114k. The share of sessions with an Enhanced Ecommerce product-view hit stays between 8.9% and 19.1% every month, so **product-level tracking works for the full year**. Two things to handle:
- **2017-08 is a single day** (2,556 sessions). It will be excluded from monthly trends.
- **Oct–Nov 2016 has a traffic spike** (98k and 114k sessions) with *lower* conversion (0.89%, 0.81%) and less product browsing. It looks like a low-intent traffic campaign, to investigate in Analyze (it turned out to be YouTube; Part C).

### 3. Field completeness

| Field | Issue | Handling |
|---|---|---|
| `totals.timeOnSite` | NULL on 49.99% of sessions, almost exactly the bounce rate (49.87%) | NULL means a single-hit session, so **impute 0**. It is not missing data |
| `geoNetwork.city` | 56.2% "not available in demo dataset" | **Don't use city.** Use country / sub-continent |
| `trafficSource.campaign` | 96.9% "(not set)" | Campaign-level analysis isn't possible, so **work at channel level** |
| `trafficSource.keyword` | 24.6% "(not provided)" | Not used |
| `geoNetwork.country` | 0.16% "(not set)" | Keep as its own category |
| Purchase without revenue | 0.004% of sessions | Negligible; keep for conversion counts |

### 4. The 898 "duplicate" session keys aren't duplicates

All 898 `(fullVisitorId, visitId)` pairs appear exactly twice, with **two different start times one day apart**. These are sessions that crossed midnight, which GA splits into two sessions. **Unique key = `(fullVisitorId, visitId, visitStartTime)`**. Both rows stay, matching how GA counts sessions.

### 5. A few very large purchase sessions dominate some channels' revenue

Revenue here is per purchase session, all traffic, as GA labels it. A session can hold several transactions.

| Channel | Purchase sessions | Median | 99th percentile | Max | Revenue share from purchase sessions > $5k |
|---|---:|---:|---:|---:|---:|
| Referral | 5,322 | $62 | $1,182 | $8,258 | 4.7% |
| Organic Search | 3,443 | $44 | $1,015 | $9,228 | 5.8% |
| Direct | 2,061 | $66 | $1,961 | $25,249 | 15.8% |
| Paid Search | 469 | $45 | $798 | $1,526 | 0.0% |
| **Display** | 143 | $85 | $32,154 | **$47,082** | **72.7%** |

**73% of Display's revenue comes from a handful of purchase sessions over $5k** (likely bulk or corporate). Revenue-based attribution would reward Display for a few unusual purchases. This needs a decision in Process (D-P2). Analyze (Part C) found who made them: Display's four purchase sessions over $5k, including the $47,082 maximum and the $32,154 at its 99th percentile, all belong to one outside corporate buyer, reported from then on as its own segment (the key account).

### 6. Purchase timing supports 30-day windows

External visitors first seen in the window who bought (n = 5,047):

| Bought within | Same visit | 7 days | 14 days | 30 days | 60 days | 90 days |
|---|---:|---:|---:|---:|---:|---:|
| Cumulative % | 57.3% | 85.6% | 90.2% | **95.1%** | 97.8% | 99.0% |

A **30-day window captures 95% of these first purchases** (counting those made on the first visit) and matches GA's default attribution lookback. For purchases made on a *later* visit, which the remarketing label targets, it captures 88.6% (1,908 of 2,154; finding 7).

### 7. Corrected sizing for the remarketing model (D2)

| External new visitors | Returned at least once | Bought on first visit | Bought on a later visit | …within 30 days |
|---:|---:|---:|---:|---:|
| 668,583 | 74,349 (11.1%) | 2,893 | 2,154 | **1,908** |

The D2 population is first-visit non-buyers (about 665,700), and the target is a purchase within 30 days on a later visit. The **base rate is about 0.29%**: roughly 1,900 positive examples, enough to model, but it confirms that **PR-AUC and lift are the right metrics**, not accuracy.

## Licensing, privacy, security

- Google publishes the data as a BigQuery public dataset for learning and demonstration. I query it in place, and only aggregates and derived extracts are stored locally.
- It contains no PII. Visitor IDs are hashed and cities are partly redacted by Google.
- Local extracts are gitignored, except three small results that are slow to recreate: the BigQuery ML evaluation and predictions (`data/raw/a11_*`, `a13_*`) and the tuning results (`data/processed/tuning_results.json`). The predictions keep Google's hashed visitor and session IDs, the same ones anyone can query in the public tables. Credentials live in the gcloud config, outside the repo.

## Decision log

| ID | Decision | Why |
|---|---|---|
| D-P1 | Treat likely-internal traffic as a separate segment, **flagged at the visitor level**. Keep it in store-wide totals and exclude it from attribution (D1) and the remarketing model (D2) | Marketing can't buy employee visits. Leaving them in would make "Referral" look like the best channel. Visitor-level flagging stops their later direct visits from inflating Direct |
| D-P1a | Internal rule = `channelGrouping = 'Referral' AND source = '(direct)' AND referralPath = '/'` | Covers 97.5% of redacted-referral sessions, with 65.8% from Google office cities. The remaining paths are too small to matter. Later verified against unredacted referrers: 99.2% precision, 98.2% of employee purchases caught (D-A1) |
| D-P1b | Session key = `(fullVisitorId, visitId, visitStartTime)` | The 898 "duplicates" are midnight-split sessions, not errors |
| D-P1c | `timeOnSite` NULL → 0; city not used; channel-level (not campaign-level) analysis | See completeness table |
| D-P2 | Attribution credit is measured in **conversions first** (each purchase = 1). Revenue credit is shown second, with each purchase session **capped at the 99th percentile** ($1,606, set in Process). Bulk purchase sessions are reported separately | Conversions can't be skewed by a few $30k+ purchase sessions. Capping keeps revenue visible without letting outliers drive budget advice |

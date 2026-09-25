# Looker Studio dashboard: build guide

An interactive companion to the case study for the Head of Marketing. The data is five small CSV files in [`data/`](data/). All are aggregates rebuilt from the pipeline (`python -m src.dashboard`, after running notebooks 01–05). No visitor-level rows are included, and nothing expires.

**Time to build:** about 20–30 minutes. **Cost:** free.

| File | Rows | What it holds |
|---|---:|---|
| `monthly_channel.csv` | 175 | Sessions, purchases, and revenue (raw and capped) by month, channel (GA's label), and segment (External / Internal (Google employees)) |
| `channel_profile.csv` | 8 | External traffic by channel: sessions, visitors, purchases, conversion rate, revenue |
| `attribution_credit.csv` | 48 | Share of purchases and revenue credited to each channel under six attribution models, with 95% bootstrap intervals |
| `value_per_click.csv` | 21 | Revenue credited per actual visit for Paid Search, Display, and Affiliates under seven attribution rules |
| `retargeting_bands.csv` | 7 | First-time visitors by model score band in the test months: buy rate and 30-day revenue per visitor (95% CI) |

## 1. Add the data sources

For each CSV file: open [lookerstudio.google.com](https://lookerstudio.google.com) → **Create → Data source** → **File Upload** connector → upload the file → **Connect**. Then check the field types:

- `month` → **Date**. Shares, rates, and `purchase_share_ci_*` → **Percent**. `*_usd` and `value_per_click_usd` → **Currency (USD)**.
- `band_order` → **Number**. You'll use it to sort bands in the right order.

## 2. Create the report and theme

**Create → Report**, then add all five data sources. Under **Theme and layout → Customize**, set:

| Setting | Value | Why |
|---|---|---|
| Background | `#FCFCFB` | Same surface as the report charts |
| Primary text / secondary text | `#0B0B0B` / `#52514E` | |
| Chart palette, in this order | `#2A78D6` (blue), `#EB6834` (orange), `#1BAF7A` (aqua) | The first three slots of the validated chart palette |
| Font | IBM Plex Sans if it is in your font list, otherwise Roboto | |

## 3. Page 1: Overview

| Element | Data source | Settings |
|---|---|---|
| **Drop-down control** | monthly_channel | Control field `segment`; default **External** |
| **Scorecards** (four) | monthly_channel | `sessions`, `purchases`, `revenue_capped_usd`, and the calculated field below |
| **Time series** | monthly_channel | Dimension `month`; breakdown `channel`; metric `purchases` |
| **Time series** | monthly_channel | Dimension `month`; breakdown `channel`; metric `sessions`. It shows the YouTube (Social) spike in Oct–Nov 2016 |
| **Table** | channel_profile | `channel`, `sessions`, `purchases`, `conversion_rate`, `revenue_capped_usd`; sort by purchases, descending |

Calculated field in monthly_channel, **Employee share of revenue**:
```
SUM(CASE WHEN segment = "Internal (Google employees)" THEN revenue_usd ELSE 0 END) / SUM(revenue_usd)
```
This scorecard should read **41%**. Put it on the page with a text box: *"Excluded from every marketing figure: Google employees reach the store through an internal link."*

## 4. Page 2: Channel credit

| Element | Data source | Settings |
|---|---|---|
| **Drop-down control** | attribution_credit | Control field `model`; allow multiple selections; default **GA report (last non-direct click)** and **Markov (data-driven)** |
| **Bar chart** (horizontal) | attribution_credit | Dimension `channel`; breakdown `model`; metric `purchase_share` (MAX, since there's one row per channel and model) |
| **Table** | attribution_credit | `channel`, `model`, `purchase_share`, `purchase_share_ci_low`, `purchase_share_ci_high`, `revenue_share` |
| **Bar chart** | value_per_click | Dimension `model`; metric `value_per_click_usd`; a drop-down control on `channel` (default **Paid Search**) |

Text box: *"GA's report credits a visitor's earlier campaign when they come back by bookmark. That's why it gives Organic Search 54% of purchases, where the data-driven model gives 38%."*

## 5. Page 3: Retargeting break-even (interactive)

Add three **parameters** (Resource → Manage added data sources → retargeting_bands → Edit → **Add a parameter**):

| Parameter | Type | Default | Range |
|---|---|---:|---|
| `Lift` | Number (decimal) | 0.10 | 0.05–0.20 |
| `Margin` | Number (decimal) | 0.50 | 0.30–0.70 |
| `Cost per visitor` | Number (decimal) | 0.10 | 0–2 |

Add two **calculated fields** to retargeting_bands:
```
Max affordable cost      = revenue_per_visitor * Lift * Margin
Worth retargeting?       = CASE WHEN revenue_per_visitor * Lift * Margin >= Cost per visitor THEN "Include" ELSE "Skip" END
```

| Element | Settings |
|---|---|
| **Input box or slider controls** | One per parameter, so the viewer can change the lift, margin, and their actual cost per visitor |
| **Bar chart** (horizontal) | Dimension `band`; metric `Max affordable cost` (MAX); sort by `band_order` ascending |
| **Table** | `band`, `visitors_per_month`, `buy_rate`, `revenue_per_visitor`, `Max affordable cost`, `Worth retargeting?` |

With the defaults, the table should match the report: the top 1% is worth **$0.70** per visitor, and bands below the top 20% are worth less than a cent.

## 6. Share it

**Share → Manage access → Anyone with the link can view.** Then paste the link into the main [README](../README.md) under *Share*.

## Refreshing the data

After re-running the notebooks, run `python -m src.dashboard` and replace each file in its File Upload data source (**Edit connection → replace file**). The charts keep working as long as the column names stay the same.

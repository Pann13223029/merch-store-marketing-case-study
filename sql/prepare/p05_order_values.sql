-- Distribution of order value per purchase session: are there bulk/corporate outliers?
-- Median and 99th percentile are exact nearest-rank percentiles (PERCENTILE_DISC), so a re-run returns the
-- same values (APPROX_QUANTILES is approximate and can change between runs).
WITH purchases AS (
  SELECT channelGrouping AS channel, totals.totalTransactionRevenue / 1e6 AS rev
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
  WHERE totals.transactions > 0
),
pct AS (
  SELECT DISTINCT
    channel,
    PERCENTILE_DISC(rev, 0.5) OVER (PARTITION BY channel)            AS median_rev,
    PERCENTILE_DISC(rev, 0.99) OVER (PARTITION BY channel)           AS p99_rev
  FROM purchases
),
agg AS (
  SELECT
    channel,
    COUNT(*)                                                          AS purchase_sessions,
    MAX(rev)                                                          AS max_rev,
    100 * SUM(IF(rev > 5000, rev, 0)) / SUM(rev)                      AS pct_over_5k
  FROM purchases
  GROUP BY channel
)
SELECT
  agg.channel                                                         AS channel,
  agg.purchase_sessions                                               AS purchase_sessions,
  ROUND(pct.median_rev, 0)                                            AS median_order_usd,
  ROUND(pct.p99_rev, 0)                                               AS p99_order_usd,
  ROUND(agg.max_rev, 0)                                               AS max_order_usd,
  ROUND(agg.pct_over_5k, 1)                                           AS pct_revenue_from_orders_over_5k
FROM agg
JOIN pct USING (channel)
ORDER BY purchase_sessions DESC

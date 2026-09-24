-- Distribution of order value per purchase session: are there bulk/corporate outliers?
SELECT
  channelGrouping                                                     AS channel,
  COUNT(*)                                                            AS purchase_sessions,
  ROUND(APPROX_QUANTILES(totals.totalTransactionRevenue / 1e6, 100)[OFFSET(50)], 0) AS median_order_usd,
  ROUND(APPROX_QUANTILES(totals.totalTransactionRevenue / 1e6, 100)[OFFSET(99)], 0) AS p99_order_usd,
  ROUND(MAX(totals.totalTransactionRevenue / 1e6), 0)                AS max_order_usd,
  ROUND(100 * SUM(IF(totals.totalTransactionRevenue / 1e6 > 5000, totals.totalTransactionRevenue, 0))
        / SUM(totals.totalTransactionRevenue), 1)                    AS pct_revenue_from_orders_over_5k
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE totals.transactions > 0
GROUP BY channel
ORDER BY purchase_sessions DESC

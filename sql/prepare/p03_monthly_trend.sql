-- Monthly sessions, conversion and revenue: look for tracking gaps or breaks.
SELECT
  FORMAT_DATE('%Y-%m', PARSE_DATE('%Y%m%d', date))              AS month,
  COUNT(*)                                                      AS sessions,
  COUNTIF(totals.transactions > 0)                              AS purchase_sessions,
  ROUND(100 * COUNTIF(totals.transactions > 0) / COUNT(*), 2)   AS conv_pct,
  ROUND(SUM(totals.totalTransactionRevenue) / 1e6, 0)           AS revenue_usd,
  -- share of sessions with at least one Enhanced Ecommerce product-view hit (tracking coverage check)
  ROUND(100 * COUNTIF(EXISTS(SELECT 1 FROM UNNEST(hits) h WHERE h.eCommerceAction.action_type = '2'))
        / COUNT(*), 1)                                          AS pct_sessions_with_product_view
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
GROUP BY month
ORDER BY month

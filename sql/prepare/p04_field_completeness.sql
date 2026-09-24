-- Completeness of fields we plan to use: NULLs and placeholder values that mean "missing".
SELECT
  COUNT(*)                                                                    AS sessions,
  ROUND(100 * COUNTIF(totals.pageviews IS NULL) / COUNT(*), 2)                AS pct_null_pageviews,
  ROUND(100 * COUNTIF(totals.timeOnSite IS NULL) / COUNT(*), 2)               AS pct_null_time_on_site,
  ROUND(100 * COUNTIF(totals.bounces = 1) / COUNT(*), 2)                      AS pct_bounces,
  ROUND(100 * COUNTIF(geoNetwork.country = '(not set)') / COUNT(*), 2)        AS pct_country_not_set,
  ROUND(100 * COUNTIF(geoNetwork.city = 'not available in demo dataset') / COUNT(*), 2) AS pct_city_redacted,
  ROUND(100 * COUNTIF(device.browser = 'not available in demo dataset') / COUNT(*), 2)  AS pct_browser_redacted,
  ROUND(100 * COUNTIF(trafficSource.keyword = '(not provided)') / COUNT(*), 2)          AS pct_keyword_not_provided,
  ROUND(100 * COUNTIF(trafficSource.campaign = '(not set)') / COUNT(*), 2)              AS pct_campaign_not_set,
  ROUND(100 * COUNTIF(totals.transactions > 0 AND totals.totalTransactionRevenue IS NULL) / COUNT(*), 4)
                                                                              AS pct_purchase_without_revenue,
  COUNT(*) - COUNT(DISTINCT CONCAT(fullVisitorId, '-', CAST(visitId AS STRING)))        AS duplicate_session_keys
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

-- What are the 137,660 sessions that exist only in the lab's table?
WITH pub AS (
  SELECT DISTINCT fullVisitorId, visitId, visitStartTime
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
)
SELECT
  IF(REGEXP_CONTAINS(l.trafficSource.source, r'googleplex\.com$|corp\.google\.com$')
     OR (l.trafficSource.source = 'sites.google.com'
         AND STARTS_WITH(IFNULL(l.trafficSource.referralPath, ''), '/a/google.com/')),
     'Google-internal referrer', l.channelGrouping)          AS channel_or_internal,
  COUNT(*)                                                    AS sessions,
  COUNTIF(l.totals.transactions > 0)                          AS purchase_sessions,
  ROUND(SUM(l.totals.totalTransactionRevenue) / 1e6, 0)       AS revenue_usd
FROM `data-to-insights.ecommerce.web_analytics` l
LEFT JOIN pub p USING (fullVisitorId, visitId, visitStartTime)
WHERE p.fullVisitorId IS NULL
GROUP BY channel_or_internal
ORDER BY sessions DESC

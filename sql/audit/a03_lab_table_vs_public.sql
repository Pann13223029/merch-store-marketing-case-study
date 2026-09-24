-- How does the lab's training table (data-to-insights.ecommerce.web_analytics) differ from the
-- public sample (google_analytics_sample.ga_sessions_*)? Match sessions on the unique session key.
WITH lab AS (
  SELECT fullVisitorId, visitId, visitStartTime,
         IFNULL(totals.transactions, 0) > 0 AS purchased,
         IFNULL(totals.totalTransactionRevenue, 0) / 1e6 AS revenue
  FROM `data-to-insights.ecommerce.web_analytics`
),
pub AS (
  SELECT fullVisitorId, visitId, visitStartTime,
         IFNULL(totals.transactions, 0) > 0 AS purchased,
         IFNULL(totals.totalTransactionRevenue, 0) / 1e6 AS revenue
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
)
SELECT
  CASE WHEN lab.fullVisitorId IS NULL THEN 'public only'
       WHEN pub.fullVisitorId IS NULL THEN 'lab only'
       ELSE 'in both' END                                  AS membership,
  COUNT(*)                                                  AS sessions,
  COUNTIF(lab.purchased)                                    AS lab_purchase_sessions,
  COUNTIF(pub.purchased)                                    AS public_purchase_sessions,
  ROUND(SUM(lab.revenue), 0)                                AS lab_revenue_usd,
  ROUND(SUM(pub.revenue), 0)                                AS public_revenue_usd,
  COUNTIF(lab.purchased AND pub.purchased)                  AS purchase_in_both
FROM lab FULL OUTER JOIN pub USING (fullVisitorId, visitId, visitStartTime)
GROUP BY membership
ORDER BY sessions DESC

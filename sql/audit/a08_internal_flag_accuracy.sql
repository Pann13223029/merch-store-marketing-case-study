-- Accuracy of our inferred visitor-level internal flag (public sample) against ground truth from the
-- lab's unredacted table: a visitor is truly internal if ANY of their sessions there came from a
-- Google-internal referrer (googleplex.com, corp.google.com, other internal Google tools, the
-- employee perks site, or Google's internal Sites domain sites.google.com/a/google.com/).
WITH lab_s AS (
  SELECT fullVisitorId,
         REGEXP_CONTAINS(trafficSource.source,
           r'(googleplex\.com|corp\.google\.com|borg\.google\.com|adz\.google\.com|sandbox\.google\.com|perksplus\.com)(:\d+)?$')
         OR (trafficSource.source = 'sites.google.com'
             AND STARTS_WITH(IFNULL(trafficSource.referralPath, ''), '/a/google.com/')) AS true_internal_entry
  FROM `data-to-insights.ecommerce.web_analytics`
),
lab_v AS (
  SELECT fullVisitorId, LOGICAL_OR(true_internal_entry) AS truly_internal
  FROM lab_s GROUP BY fullVisitorId
),
pub_v AS (
  SELECT fullVisitorId,
         LOGICAL_OR(channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
                    AND trafficSource.referralPath = '/') AS our_flag,
         COUNT(*) AS sessions,
         COUNTIF(totals.transactions > 0) AS purchase_sessions,
         SUM(IFNULL(totals.totalTransactionRevenue, 0)) / 1e6 AS revenue_usd
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
  GROUP BY fullVisitorId
)
SELECT
  IF(our_flag, 'flagged internal', 'treated as external')          AS our_flag,
  CASE WHEN truly_internal IS NULL THEN 'visitor not in lab table'
       WHEN truly_internal THEN 'truly internal'
       ELSE 'truly external' END                                   AS ground_truth,
  COUNT(*)                                                         AS visitors,
  SUM(sessions)                                                    AS sessions,
  SUM(purchase_sessions)                                           AS purchase_sessions,
  ROUND(SUM(revenue_usd), 0)                                       AS revenue_usd
FROM pub_v LEFT JOIN lab_v USING (fullVisitorId)
GROUP BY 1, 2
ORDER BY 1, 2

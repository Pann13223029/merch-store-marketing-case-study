-- For external visitors first seen inside the window: how long from first visit to first purchase?
-- Drives the attribution lookback window and the retargeting follow-up window.
-- A first visit that crosses midnight appears as two sessions with the same visitId; a purchase in the
-- after-midnight half counts as the first visit (same visit, 0 days), as in the D2 label
-- (Prepare finding 7; Process, midnight-split sessions).
WITH s AS (
  SELECT fullVisitorId, visitId, visitNumber, visitStartTime,
         IFNULL(totals.transactions, 0) > 0 AS purchased,
         channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
           AND trafficSource.referralPath = '/' AS internal_entry
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
),
v AS (
  SELECT fullVisitorId,
         MIN(visitNumber) AS min_visit_number,
         LOGICAL_OR(internal_entry) AS is_internal,
         MIN(visitStartTime) AS first_ts,
         MIN(IF(purchased, visitStartTime, NULL)) AS first_purchase_ts,
         ARRAY_AGG(visitId ORDER BY visitStartTime LIMIT 1)[OFFSET(0)] AS first_vid,
         ARRAY_AGG(IF(purchased, visitId, NULL) IGNORE NULLS ORDER BY visitStartTime LIMIT 1)[SAFE_OFFSET(0)]
           AS first_purchase_vid
  FROM s GROUP BY fullVisitorId
),
buyers AS (
  SELECT IF(first_purchase_vid = first_vid, 0, (first_purchase_ts - first_ts) / 86400) AS days_to_purchase
  FROM v
  WHERE NOT is_internal AND min_visit_number = 1 AND first_purchase_ts IS NOT NULL
)
SELECT
  COUNT(*)                                                  AS buyers,
  ROUND(100 * COUNTIF(days_to_purchase = 0) / COUNT(*), 1)  AS pct_same_visit,
  ROUND(100 * COUNTIF(days_to_purchase > 0 AND days_to_purchase <= 1) / COUNT(*), 1)   AS pct_within_1d,
  ROUND(100 * COUNTIF(days_to_purchase <= 7) / COUNT(*), 1)  AS cum_pct_7d,
  ROUND(100 * COUNTIF(days_to_purchase <= 14) / COUNT(*), 1) AS cum_pct_14d,
  ROUND(100 * COUNTIF(days_to_purchase <= 30) / COUNT(*), 1) AS cum_pct_30d,
  ROUND(100 * COUNTIF(days_to_purchase <= 60) / COUNT(*), 1) AS cum_pct_60d,
  ROUND(100 * COUNTIF(days_to_purchase <= 90) / COUNT(*), 1) AS cum_pct_90d
FROM buyers

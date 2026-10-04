-- External (non-internal) visitors first seen inside the window: bought on first visit vs a later visit.
-- A first visit that crosses midnight appears as two sessions with the same visitId; a purchase in the
-- after-midnight half counts as the first visit, as in the D2 label
-- (Prepare finding 7; Process, midnight-split sessions).
WITH s AS (
  SELECT fullVisitorId, visitId, visitNumber, visitStartTime,
         IFNULL(totals.transactions, 0) > 0 AS purchased,
         channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
           AND trafficSource.referralPath = '/' AS internal_entry
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
),
v AS (
  SELECT fullVisitorId, MIN(visitNumber) AS min_visit_number, LOGICAL_OR(internal_entry) AS is_internal,
         MIN(visitStartTime) AS first_ts, MIN(IF(purchased, visitStartTime, NULL)) AS first_purchase_ts,
         ARRAY_AGG(visitId ORDER BY visitStartTime LIMIT 1)[OFFSET(0)] AS first_vid,
         ARRAY_AGG(IF(purchased, visitId, NULL) IGNORE NULLS ORDER BY visitStartTime LIMIT 1)[SAFE_OFFSET(0)]
           AS first_purchase_vid,
         COUNT(*) AS sessions
  FROM s GROUP BY fullVisitorId
)
SELECT
  COUNTIF(NOT is_internal AND min_visit_number = 1)                                          AS external_new_visitors,
  COUNTIF(NOT is_internal AND min_visit_number = 1 AND sessions > 1)                         AS external_new_who_returned,
  COUNTIF(NOT is_internal AND min_visit_number = 1 AND first_purchase_vid = first_vid)       AS bought_first_visit,
  COUNTIF(NOT is_internal AND min_visit_number = 1 AND first_purchase_ts > first_ts
          AND first_purchase_vid != first_vid)                                                AS bought_later_visit,
  COUNTIF(NOT is_internal AND min_visit_number = 1 AND first_purchase_ts > first_ts
          AND first_purchase_vid != first_vid
          AND first_purchase_ts - first_ts <= 30 * 86400)                                     AS bought_later_within_30d
FROM v

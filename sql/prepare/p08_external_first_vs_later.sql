-- External (non-internal) visitors first seen inside the window: bought on first visit vs a later visit.
WITH s AS (
  SELECT fullVisitorId, visitNumber, visitStartTime,
         IFNULL(totals.transactions, 0) > 0 AS purchased,
         channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
           AND trafficSource.referralPath = '/' AS internal_entry
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
),
v AS (
  SELECT fullVisitorId, MIN(visitNumber) AS min_visit_number, LOGICAL_OR(internal_entry) AS is_internal,
         MIN(visitStartTime) AS first_ts, MIN(IF(purchased, visitStartTime, NULL)) AS first_purchase_ts,
         COUNT(*) AS sessions
  FROM s GROUP BY fullVisitorId
)
SELECT
  COUNTIF(NOT is_internal AND min_visit_number = 1)                                          AS external_new_visitors,
  COUNTIF(NOT is_internal AND min_visit_number = 1 AND sessions > 1)                         AS external_new_who_returned,
  COUNTIF(NOT is_internal AND min_visit_number = 1 AND first_purchase_ts = first_ts)         AS bought_first_visit,
  COUNTIF(NOT is_internal AND min_visit_number = 1 AND first_purchase_ts > first_ts)         AS bought_later_visit,
  COUNTIF(NOT is_internal AND min_visit_number = 1 AND first_purchase_ts > first_ts
          AND first_purchase_ts - first_ts <= 30 * 86400)                                     AS bought_later_within_30d
FROM v

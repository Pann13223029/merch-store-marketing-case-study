-- Audit of Google's BQML lab (GSP229) label: will_buy_on_return_visit.
-- Lab definition (verbatim logic): label computed over ALL sessions (no date limit),
-- training rows = first visits (totals.newVisits = 1) dated 2016-08-01..2017-04-30,
-- evaluation rows = first visits dated 2017-05-01..2017-06-30.
-- We add our visitor-level internal flag to see who the positive examples are.
WITH all_s AS (
  SELECT fullVisitorId, date, totals.newVisits AS new_visits,
         IFNULL(totals.transactions, 0) > 0 AS purchased,
         channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
           AND trafficSource.referralPath = '/' AS internal_entry
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
),
label AS (
  SELECT fullVisitorId,
         IF(COUNTIF(purchased AND new_visits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit,
         LOGICAL_OR(internal_entry) AS is_internal
  FROM all_s GROUP BY fullVisitorId
),
first_visits AS (
  SELECT fullVisitorId, date,
         CASE WHEN date BETWEEN '20160801' AND '20170430' THEN 'train'
              WHEN date BETWEEN '20170501' AND '20170630' THEN 'eval'
              WHEN date BETWEEN '20170701' AND '20170801' THEN 'predict' END AS split
  FROM all_s WHERE new_visits = 1
)
SELECT split,
       COUNT(*)                                                            AS n_rows,
       SUM(will_buy_on_return_visit)                                       AS positives,
       ROUND(100 * AVG(will_buy_on_return_visit), 3)                       AS positive_rate_pct,
       SUM(IF(is_internal, will_buy_on_return_visit, 0))                   AS internal_positives,
       ROUND(100 * SUM(IF(is_internal, will_buy_on_return_visit, 0))
             / NULLIF(SUM(will_buy_on_return_visit), 0), 1)                AS pct_positives_internal,
       ROUND(100 * AVG(IF(is_internal, NULL, will_buy_on_return_visit)), 3) AS external_positive_rate_pct
FROM first_visits JOIN label USING (fullVisitorId)
WHERE split IS NOT NULL
GROUP BY split
ORDER BY MIN(date)

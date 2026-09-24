-- 898 (fullVisitorId, visitId) keys appear more than once. Are these sessions that cross
-- midnight and get split across two daily tables (same visitStartTime, consecutive dates)?
WITH s AS (
  SELECT fullVisitorId, visitId, visitStartTime, date, totals.hits AS hits,
         IFNULL(totals.transactions, 0) AS transactions
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
),
dupes AS (
  SELECT fullVisitorId, visitId,
         COUNT(*) AS n_rows,
         COUNT(DISTINCT visitStartTime) AS n_start_times,
         DATE_DIFF(PARSE_DATE('%Y%m%d', MAX(date)), PARSE_DATE('%Y%m%d', MIN(date)), DAY) AS day_span,
         SUM(transactions) AS transactions
  FROM s GROUP BY fullVisitorId, visitId HAVING COUNT(*) > 1
)
SELECT n_rows, n_start_times, day_span, COUNT(*) AS keys, SUM(transactions) AS transactions
FROM dupes GROUP BY 1, 2, 3 ORDER BY keys DESC

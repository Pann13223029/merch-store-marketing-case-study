-- Size of the likely-internal segment when flagged at the VISITOR level:
-- any visitor who ever arrived via the redacted internal referral (Referral, source "(direct)", path "/")
-- has all of their sessions flagged.
WITH s AS (
  SELECT fullVisitorId, channelGrouping,
         IFNULL(totals.transactions, 0) > 0 AS purchased,
         IFNULL(totals.totalTransactionRevenue, 0) / 1e6 AS revenue,
         channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
           AND trafficSource.referralPath = '/' AS internal_entry
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
),
flagged AS (
  SELECT fullVisitorId, LOGICAL_OR(internal_entry) AS is_internal
  FROM s GROUP BY fullVisitorId
)
SELECT
  IF(f.is_internal, 'Internal (likely employees)', 'External') AS segment,
  COUNT(DISTINCT s.fullVisitorId)                              AS visitors,
  COUNT(*)                                                     AS sessions,
  COUNTIF(s.purchased)                                         AS purchase_sessions,
  ROUND(100 * COUNTIF(s.purchased) / COUNT(*), 2)              AS conv_pct,
  ROUND(SUM(s.revenue), 0)                                     AS revenue_usd,
  -- where the internal visitors' OTHER sessions land (these would inflate Direct etc.)
  COUNTIF(NOT s.internal_entry AND s.channelGrouping = 'Direct') AS non_entry_direct_sessions
FROM s JOIN flagged f USING (fullVisitorId)
GROUP BY segment

-- Which referral paths make up the "Referral / (direct)" segment, and where are they from?
-- Used to decide the exact rule for flagging likely-internal (Google employee) traffic.
SELECT
  trafficSource.referralPath                                         AS referral_path,
  COUNT(*)                                                           AS sessions,
  COUNT(DISTINCT fullVisitorId)                                      AS visitors,
  ROUND(100 * COUNTIF(totals.transactions > 0) / COUNT(*), 2)        AS conv_pct,
  ROUND(100 * COUNTIF(geoNetwork.city IN
        ('Mountain View', 'Sunnyvale', 'San Francisco', 'New York', 'San Jose',
         'Palo Alto', 'Kirkland', 'Cambridge', 'Seattle', 'Los Angeles')) / COUNT(*), 1)
                                                                     AS pct_google_office_cities,
  ROUND(SUM(totals.totalTransactionRevenue) / 1e6, 0)                AS revenue_usd
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
GROUP BY referral_path
ORDER BY sessions DESC
LIMIT 10

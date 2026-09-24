-- Verify our inferred internal rule on the public sample (Referral + source "(direct)" + path "/")
-- against the true referrer, using the sessions that appear in both tables.
WITH pub AS (
  SELECT fullVisitorId, visitId, visitStartTime,
         CASE WHEN channelGrouping = 'Referral' AND trafficSource.source = '(direct)' AND trafficSource.referralPath = '/'
                THEN 'flagged internal (path /)'
              WHEN channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
                THEN CONCAT('redacted, other path: ', trafficSource.referralPath)
              ELSE 'not redacted' END AS public_label
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
),
lab AS (
  SELECT fullVisitorId, visitId, visitStartTime, trafficSource.source AS true_source
  FROM `data-to-insights.ecommerce.web_analytics`
)
SELECT public_label, true_source, COUNT(*) AS sessions
FROM pub JOIN lab USING (fullVisitorId, visitId, visitStartTime)
WHERE public_label != 'not redacted'
GROUP BY public_label, true_source
QUALIFY ROW_NUMBER() OVER (PARTITION BY public_label ORDER BY COUNT(*) DESC) <= 3
ORDER BY public_label, sessions DESC

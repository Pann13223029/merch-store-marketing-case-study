-- Referral sources in the lab's table: is the referrer that the public sample hides visible here?
SELECT channelGrouping AS channel, trafficSource.source AS source, trafficSource.referralPath AS referral_path,
       COUNT(*) AS sessions,
       COUNTIF(totals.transactions > 0) AS purchase_sessions,
       ROUND(SUM(totals.totalTransactionRevenue) / 1e6, 0) AS revenue_usd
FROM `data-to-insights.ecommerce.web_analytics`
WHERE channelGrouping = 'Referral'
GROUP BY channel, source, referral_path
ORDER BY sessions DESC
LIMIT 8

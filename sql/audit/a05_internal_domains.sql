-- Every Google-internal referrer in the lab's (unredacted) table.
-- Internal = *.googleplex.com, *.corp.google.com, or Google's internal Sites domain (sites.google.com/a/google.com/...).
SELECT trafficSource.source AS source,
       IF(trafficSource.source = 'sites.google.com', REGEXP_EXTRACT(trafficSource.referralPath, r'^(/a/[^/]+/)'), NULL) AS sites_domain,
       COUNT(*) AS sessions,
       COUNTIF(totals.transactions > 0) AS purchase_sessions,
       ROUND(SUM(totals.totalTransactionRevenue) / 1e6, 0) AS revenue_usd
FROM `data-to-insights.ecommerce.web_analytics`
WHERE REGEXP_CONTAINS(trafficSource.source, r'googleplex\.com$|corp\.google\.com$')
   OR (trafficSource.source = 'sites.google.com' AND STARTS_WITH(IFNULL(trafficSource.referralPath, ''), '/a/google.com/'))
GROUP BY source, sites_domain
ORDER BY sessions DESC

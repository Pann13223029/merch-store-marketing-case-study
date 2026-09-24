-- Visitor IDs that the lab's unredacted table shows arriving from a Google-internal referrer
-- (ground truth, same rule as a08/a09). Used to check residual employee traffic in our D2 table.
SELECT DISTINCT fullVisitorId AS full_visitor_id
FROM `data-to-insights.ecommerce.web_analytics`
WHERE REGEXP_CONTAINS(trafficSource.source,
        r'(googleplex\.com|corp\.google\.com|borg\.google\.com|adz\.google\.com|sandbox\.google\.com|perksplus\.com)(:\d+)?$')
   OR (trafficSource.source = 'sites.google.com'
       AND STARTS_WITH(IFNULL(trafficSource.referralPath, ''), '/a/google.com/'))

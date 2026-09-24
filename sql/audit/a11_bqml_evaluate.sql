-- Evaluate both BigQuery ML lab models on the lab's evaluation months (2017-05-01..2017-06-30),
-- on all first visits (as published) and on external visitors only (ground-truth internal flag).
WITH visitor AS (
  SELECT fullvisitorid,
         IF(COUNTIF(totals.transactions > 0 AND totals.newVisits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit,
         LOGICAL_OR(
           REGEXP_CONTAINS(trafficSource.source,
             r'(googleplex\.com|corp\.google\.com|borg\.google\.com|adz\.google\.com|sandbox\.google\.com|perksplus\.com)(:\d+)?$')
           OR (trafficSource.source = 'sites.google.com'
               AND STARTS_WITH(IFNULL(trafficSource.referralPath, ''), '/a/google.com/'))
         ) AS is_internal
  FROM `data-to-insights.ecommerce.web_analytics`
  GROUP BY fullvisitorid
),
m1_eval AS (
  SELECT s.fullVisitorId, v.is_internal, v.will_buy_on_return_visit,
         IFNULL(s.totals.bounces, 0) AS bounces, IFNULL(s.totals.timeOnSite, 0) AS time_on_site
  FROM `data-to-insights.ecommerce.web_analytics` s JOIN visitor v USING (fullvisitorid)
  WHERE s.totals.newVisits = 1 AND s.date BETWEEN '20170501' AND '20170630'
),
m2_eval AS (
  SELECT CONCAT(fullvisitorid, CAST(visitId AS STRING)) AS unique_session_id,
         v.is_internal, v.will_buy_on_return_visit,
         MAX(CAST(h.eCommerceAction.action_type AS INT64)) AS latest_ecommerce_progress,
         IFNULL(totals.bounces, 0) AS bounces, IFNULL(totals.timeOnSite, 0) AS time_on_site,
         IFNULL(totals.pageviews, 0) AS pageviews,
         trafficSource.source, trafficSource.medium, channelGrouping, device.deviceCategory,
         IFNULL(geoNetwork.country, '') AS country
  FROM `data-to-insights.ecommerce.web_analytics`, UNNEST(hits) AS h
  JOIN visitor v USING (fullvisitorid)
  WHERE totals.newVisits = 1 AND date BETWEEN '20170501' AND '20170630'
  GROUP BY unique_session_id, is_internal, will_buy_on_return_visit, bounces, time_on_site, totals.pageviews,
           trafficSource.source, trafficSource.medium, channelGrouping, device.deviceCategory, country
)
SELECT 'model_1' AS model, 'all (as published)' AS eval_rows, roc_auc, log_loss
FROM ML.EVALUATE(MODEL lab_audit.classification_model,
                 (SELECT * EXCEPT(fullVisitorId, is_internal) FROM m1_eval))
UNION ALL
SELECT 'model_1', 'external only', roc_auc, log_loss
FROM ML.EVALUATE(MODEL lab_audit.classification_model,
                 (SELECT * EXCEPT(fullVisitorId, is_internal) FROM m1_eval WHERE NOT is_internal))
UNION ALL
SELECT 'model_2', 'all (as published)', roc_auc, log_loss
FROM ML.EVALUATE(MODEL lab_audit.classification_model_2,
                 (SELECT * EXCEPT(unique_session_id, is_internal) FROM m2_eval))
UNION ALL
SELECT 'model_2', 'external only', roc_auc, log_loss
FROM ML.EVALUATE(MODEL lab_audit.classification_model_2,
                 (SELECT * EXCEPT(unique_session_id, is_internal) FROM m2_eval WHERE NOT is_internal))
ORDER BY model, eval_rows

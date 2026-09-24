-- Row-level scores from the three BigQuery ML lab models on the lab's evaluation months,
-- with the ground-truth internal flag, for bootstrap CIs and top-of-ranking analysis.
-- One row per evaluation row (102,025), matching the lab's evaluation set exactly.
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
eval_grouped AS (
  -- The lab's evaluation rows, with the lab's GROUP BY
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
),
eval_rows AS (
  -- The lab's unique_session_id is NOT unique: a visit split at midnight yields two rows with the same
  -- id but different totals. Predictions are joined on a fingerprint of all grouping columns instead.
  SELECT *,
         FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(
           unique_session_id, bounces, time_on_site, pageviews, source, medium,
           channelGrouping, deviceCategory, country))) AS row_key
  FROM eval_grouped
),
p2 AS (
  SELECT row_key,
         (SELECT p.prob FROM UNNEST(predicted_will_buy_on_return_visit_probs) p WHERE p.label = 1) AS score_model_2
  FROM ML.PREDICT(MODEL lab_audit.classification_model_2, (SELECT * FROM eval_rows))
),
p2ns AS (
  SELECT row_key,
         (SELECT p.prob FROM UNNEST(predicted_will_buy_on_return_visit_probs) p WHERE p.label = 1) AS score_no_source
  FROM ML.PREDICT(MODEL lab_audit.classification_model_2_no_source, (SELECT * FROM eval_rows))
),
p2ext AS (
  SELECT row_key,
         (SELECT p.prob FROM UNNEST(predicted_will_buy_on_return_visit_probs) p WHERE p.label = 1) AS score_external_trained
  FROM ML.PREDICT(MODEL lab_audit.classification_model_2_external, (SELECT * FROM eval_rows))
)
SELECT e.row_key, e.unique_session_id, e.is_internal, e.will_buy_on_return_visit,
       e.channelGrouping AS channel, e.source,
       p2.score_model_2, p2ns.score_no_source, p2ext.score_external_trained
FROM eval_rows e
JOIN p2 USING (row_key)
JOIN p2ns USING (row_key)
JOIN p2ext USING (row_key)

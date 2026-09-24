-- Two audit variants of the lab's classification_model_2, same SQL and settings otherwise:
--   classification_model_2_no_source   drops source, medium, channelGrouping (mechanism test:
--                                      if traffic-source features let the model recognise employees,
--                                      removing them should close the all-vs-external gap)
--   classification_model_2_external    trained on external visitors only (ground-truth flag)
CREATE OR REPLACE MODEL lab_audit.classification_model_2_no_source
OPTIONS (model_type = 'logistic_reg', labels = ['will_buy_on_return_visit']) AS
WITH all_visitor_stats AS (
  SELECT fullvisitorid,
         IF(COUNTIF(totals.transactions > 0 AND totals.newVisits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit
  FROM `data-to-insights.ecommerce.web_analytics`
  GROUP BY fullvisitorid
)
SELECT * EXCEPT(unique_session_id) FROM (
  SELECT
    CONCAT(fullvisitorid, CAST(visitId AS STRING)) AS unique_session_id,
    will_buy_on_return_visit,
    MAX(CAST(h.eCommerceAction.action_type AS INT64)) AS latest_ecommerce_progress,
    IFNULL(totals.bounces, 0) AS bounces,
    IFNULL(totals.timeOnSite, 0) AS time_on_site,
    IFNULL(totals.pageviews, 0) AS pageviews,
    device.deviceCategory,
    IFNULL(geoNetwork.country, '') AS country
  FROM `data-to-insights.ecommerce.web_analytics`, UNNEST(hits) AS h
  JOIN all_visitor_stats USING (fullvisitorid)
  WHERE totals.newVisits = 1 AND date BETWEEN '20160801' AND '20170430'
  GROUP BY unique_session_id, will_buy_on_return_visit, bounces, time_on_site, totals.pageviews,
           device.deviceCategory, country
);

CREATE OR REPLACE MODEL lab_audit.classification_model_2_external
OPTIONS (model_type = 'logistic_reg', labels = ['will_buy_on_return_visit']) AS
WITH all_visitor_stats AS (
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
)
SELECT * EXCEPT(unique_session_id) FROM (
  SELECT
    CONCAT(fullvisitorid, CAST(visitId AS STRING)) AS unique_session_id,
    will_buy_on_return_visit,
    MAX(CAST(h.eCommerceAction.action_type AS INT64)) AS latest_ecommerce_progress,
    IFNULL(totals.bounces, 0) AS bounces,
    IFNULL(totals.timeOnSite, 0) AS time_on_site,
    IFNULL(totals.pageviews, 0) AS pageviews,
    trafficSource.source,
    trafficSource.medium,
    channelGrouping,
    device.deviceCategory,
    IFNULL(geoNetwork.country, '') AS country
  FROM `data-to-insights.ecommerce.web_analytics`, UNNEST(hits) AS h
  JOIN all_visitor_stats USING (fullvisitorid)
  WHERE totals.newVisits = 1 AND date BETWEEN '20160801' AND '20170430' AND NOT is_internal
  GROUP BY unique_session_id, will_buy_on_return_visit, bounces, time_on_site, totals.pageviews,
           trafficSource.source, trafficSource.medium, channelGrouping, device.deviceCategory, country
);

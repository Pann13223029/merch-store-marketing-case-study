-- Google's GSP229 lab models, created with the lab's own SQL in BigQuery ML (cross-check of the
-- scikit-learn replication). Runs in the sandbox project's `lab_audit` dataset (US location).
CREATE SCHEMA IF NOT EXISTS lab_audit OPTIONS (location = 'US');

CREATE OR REPLACE MODEL lab_audit.classification_model
OPTIONS (model_type = 'logistic_reg', labels = ['will_buy_on_return_visit']) AS
SELECT * EXCEPT(fullVisitorId)
FROM
  (SELECT fullVisitorId,
          IFNULL(totals.bounces, 0) AS bounces,
          IFNULL(totals.timeOnSite, 0) AS time_on_site
   FROM `data-to-insights.ecommerce.web_analytics`
   WHERE totals.newVisits = 1 AND date BETWEEN '20160801' AND '20170430')
JOIN
  (SELECT fullvisitorid,
          IF(COUNTIF(totals.transactions > 0 AND totals.newVisits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit
   FROM `data-to-insights.ecommerce.web_analytics`
   GROUP BY fullvisitorid)
USING (fullVisitorId);

CREATE OR REPLACE MODEL lab_audit.classification_model_2
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
    trafficSource.source,
    trafficSource.medium,
    channelGrouping,
    device.deviceCategory,
    IFNULL(geoNetwork.country, '') AS country
  FROM `data-to-insights.ecommerce.web_analytics`, UNNEST(hits) AS h
  JOIN all_visitor_stats USING (fullvisitorid)
  WHERE totals.newVisits = 1 AND date BETWEEN '20160801' AND '20170430'
  GROUP BY unique_session_id, will_buy_on_return_visit, bounces, time_on_site, totals.pageviews,
           trafficSource.source, trafficSource.medium, channelGrouping, device.deviceCategory, country
);

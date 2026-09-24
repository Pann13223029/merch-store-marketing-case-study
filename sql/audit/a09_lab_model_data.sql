-- Exact replication of the training and evaluation data for Google's GSP229 lab model
-- (classification_model_2), on the lab's own table, with the lab's GROUP BY semantics.
-- Added for the audit (not used as features):
--   is_internal  visitor-level ground truth: any session referred by a Google-internal host
--                (googleplex.com, corp.google.com, internal tools, employee perks site,
--                or Google's internal Sites domain sites.google.com/a/google.com/)
--   split        'train' = lab training dates, 'eval' = lab evaluation dates
WITH all_visitor_stats AS (
  SELECT
    fullvisitorid,
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
SELECT
  CONCAT(fullvisitorid, CAST(visitId AS STRING))           AS unique_session_id,
  will_buy_on_return_visit,
  is_internal,
  IF(date <= '20170430', 'train', 'eval')                   AS split,
  MAX(CAST(h.eCommerceAction.action_type AS INT64))         AS latest_ecommerce_progress,
  IFNULL(totals.bounces, 0)                                 AS bounces,
  IFNULL(totals.timeOnSite, 0)                              AS time_on_site,
  IFNULL(totals.pageviews, 0)                               AS pageviews,
  trafficSource.source                                      AS source,
  trafficSource.medium                                      AS medium,
  channelGrouping                                           AS channel,
  device.deviceCategory                                     AS device_category,
  IFNULL(geoNetwork.country, '')                            AS country
FROM `data-to-insights.ecommerce.web_analytics`, UNNEST(hits) AS h
JOIN all_visitor_stats USING (fullvisitorid)
WHERE totals.newVisits = 1
  AND date BETWEEN '20160801' AND '20170630'
GROUP BY unique_session_id, will_buy_on_return_visit, is_internal, split, bounces, time_on_site,
         pageviews, source, medium, channel, device_category, country

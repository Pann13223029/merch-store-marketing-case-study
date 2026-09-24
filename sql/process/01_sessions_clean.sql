-- One clean row per session: the single source table for attribution (D1), the lab audit and
-- the remarketing model (D2). Cleaning rules come from reports/02_prepare.md (decision log D-P1*).
--
--   * Session key = (fullVisitorId, visitId, visitStartTime): midnight-split sessions share visitId.
--   * is_internal is flagged at the VISITOR level: any visitor who ever entered through the redacted
--     internal referral (Referral, source "(direct)", path "/") has every session flagged.
--   * NULL totals mean "zero" in the GA360 export (e.g. timeOnSite is NULL for single-hit sessions).
--   * City is not used (56% redacted); campaign is not used (97% not set).
--   * eCommerceAction.action_type: 2 = product detail view, 3 = add to cart, 5 = checkout, 6 = purchase.
WITH sessions AS (
  SELECT
    CONCAT(fullVisitorId, '-', CAST(visitId AS STRING), '-', CAST(visitStartTime AS STRING)) AS session_key,
    fullVisitorId                                                    AS full_visitor_id,
    visitId                                                          AS visit_id,
    visitStartTime                                                   AS visit_start_time,
    TIMESTAMP_SECONDS(visitStartTime)                                AS session_start,
    PARSE_DATE('%Y%m%d', date)                                       AS session_date,
    visitNumber                                                      AS visit_number,
    IFNULL(totals.newVisits, 0) = 1                                  AS is_new_visit,

    -- Acquisition
    channelGrouping                                                  AS channel,
    trafficSource.source                                             AS source,
    trafficSource.medium                                             AS medium,
    IFNULL(trafficSource.isTrueDirect, FALSE)                        AS is_true_direct,
    channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
      AND trafficSource.referralPath = '/'                           AS is_internal_entry,

    -- Technology and geography
    device.deviceCategory                                            AS device_category,
    device.operatingSystem                                           AS operating_system,
    device.browser                                                   AS browser,
    geoNetwork.continent                                             AS continent,
    geoNetwork.subContinent                                          AS sub_continent,
    geoNetwork.country                                               AS country,

    -- Engagement (NULL = 0 in the GA360 export)
    IFNULL(totals.hits, 0)                                           AS hits,
    IFNULL(totals.pageviews, 0)                                      AS pageviews,
    IFNULL(totals.timeOnSite, 0)                                     AS time_on_site,
    IFNULL(totals.bounces, 0) = 1                                    AS bounced,

    -- Shopping behaviour from hit-level Enhanced Ecommerce actions
    (SELECT COUNTIF(h.eCommerceAction.action_type = '2') FROM UNNEST(hits) h) AS product_detail_views,
    (SELECT COUNT(DISTINCT p.v2ProductName) FROM UNNEST(hits) h, UNNEST(h.product) p
      WHERE h.eCommerceAction.action_type = '2')                     AS distinct_products_viewed,
    (SELECT COUNTIF(h.eCommerceAction.action_type = '3') FROM UNNEST(hits) h) AS add_to_cart_events,
    (SELECT COUNTIF(h.eCommerceAction.action_type = '5') FROM UNNEST(hits) h) AS checkout_events,
    -- Furthest funnel step reached; same definition as the lab's latest_ecommerce_progress
    (SELECT IFNULL(MAX(CAST(h.eCommerceAction.action_type AS INT64)), 0) FROM UNNEST(hits) h)
                                                                     AS max_ecommerce_step,

    -- Outcome
    IFNULL(totals.transactions, 0)                                   AS transactions,
    IFNULL(totals.totalTransactionRevenue, 0) / 1e6                  AS revenue_usd
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
)
SELECT
  s.*,
  s.transactions > 0                                                 AS purchased,
  LOGICAL_OR(s.is_internal_entry) OVER (PARTITION BY s.full_visitor_id) AS is_internal
FROM sessions s

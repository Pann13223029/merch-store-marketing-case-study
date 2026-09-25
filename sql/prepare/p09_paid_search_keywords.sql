-- Paid Search sessions with their keyword, ad-click details and landing page, to split brand from
-- non-brand search (D1). Joined in Python to sessions_clean on (full_visitor_id, visit_start_time).
SELECT
  fullVisitorId                                                       AS full_visitor_id,
  visitStartTime                                                      AS visit_start_time,
  channelGrouping                                                     AS channel,
  trafficSource.source                                                AS source,
  trafficSource.medium                                                AS medium,
  trafficSource.keyword                                               AS keyword,
  trafficSource.campaign                                              AS campaign,
  trafficSource.adContent                                             AS ad_content,
  IFNULL(trafficSource.isTrueDirect, FALSE)                           AS is_true_direct,
  trafficSource.adwordsClickInfo.adNetworkType                        AS ad_network_type,
  trafficSource.adwordsClickInfo.criteriaParameters                   AS criteria_parameters,
  trafficSource.adwordsClickInfo.campaignId                           AS campaign_id,
  trafficSource.adwordsClickInfo.adGroupId                            AS ad_group_id,
  trafficSource.adwordsClickInfo.slot                                 AS ad_slot,
  (SELECT h.page.pagePath FROM UNNEST(hits) h WHERE h.isEntrance ORDER BY h.hitNumber LIMIT 1)
                                                                      AS landing_page,
  IFNULL(totals.transactions, 0)                                      AS transactions,
  IFNULL(totals.totalTransactionRevenue, 0) / 1e6                     AS revenue_usd
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE channelGrouping = 'Paid Search' OR trafficSource.medium = 'cpc'

-- Positive rate of the lab label by first-visit month (external visitors only):
-- if the label has no fixed window, later cohorts have less time to return and the rate should fall.
WITH all_s AS (
  SELECT fullVisitorId, date, totals.newVisits AS new_visits,
         IFNULL(totals.transactions, 0) > 0 AS purchased,
         channelGrouping = 'Referral' AND trafficSource.source = '(direct)'
           AND trafficSource.referralPath = '/' AS internal_entry
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
),
label AS (
  SELECT fullVisitorId,
         IF(COUNTIF(purchased AND new_visits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit,
         LOGICAL_OR(internal_entry) AS is_internal
  FROM all_s GROUP BY fullVisitorId
)
SELECT FORMAT_DATE('%Y-%m', PARSE_DATE('%Y%m%d', f.date))         AS first_visit_month,
       COUNT(*)                                                   AS external_first_visits,
       ROUND(100 * AVG(l.will_buy_on_return_visit), 3)            AS lab_label_rate_pct
FROM all_s f JOIN label l USING (fullVisitorId)
WHERE f.new_visits = 1 AND NOT l.is_internal
GROUP BY first_visit_month ORDER BY first_visit_month

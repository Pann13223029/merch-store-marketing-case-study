# Executive summary: where should the Google Merchandise Store spend its next marketing dollar?

**For:** Head of Marketing · **Data:** Google Analytics 360 export of the Google Merchandise Store, Aug 2016 – Aug 2017 (903,653 sessions) · **Full analysis:** [README](../README.md)

## Bottom line

1. **The channel report is misleading.** It gives Organic Search about **15 points more of the purchases** than a data-driven attribution model does, and it hides that **people coming back on their own produce about half of purchases and 61% of revenue**.
2. **Paid Search pays its way.** A click brings in **$1.60–$2.50** under every attribution model tested, so it breaks even below about **$0.80–$1.20 per click** at a 50% gross margin.
3. **Display can't be judged from this data.** Its value per click is anywhere from **$2.90 to $9.10**, depending on how 1,118 ambiguous return visits are credited, and those visits hold 8 of its 10 largest orders. **Test before changing its budget.**
4. **Retargeting should be small and narrow.** The top 10% of first-time visitors include **70% of the visitors who later buy**, and the bottom 40% include none. Even so, the whole program is worth only about **$700 a month** in extra gross profit before ad costs.

## Two things to know about the data

- **About 41% of the store's revenue comes from Google employees**, not from marketing. They arrive through Google's internal store link, and we separated them out; every number above excludes them. The public version of the data hides the internal link, but Google's own training copy shows it, which let us confirm our employee filter (99% precision).
- **Google's own tutorial model for this question mostly identifies employees.** It is the BigQuery ML lab "Predict Visitor Purchases" (ROC-AUC 0.91). Its top 1% of "likely buyers" are 98.8% employees. On outside visitors it scores 0.86, and it ranks real customers worse than a version trained without employees.

## Recommendations

| # | Action | Owner | Why |
|---|---|---|---|
| 1 | **Filter out internal traffic** (referrers from `googleplex.com`, `corp.google.com`, and internal Google sites) and report it as a separate segment | Web analytics | 41% of revenue and 47% of purchase sessions are employees; including them makes Referral look like the best channel |
| 2 | **Report channels with a multi-touch view** alongside GA's default, and stop crediting Organic Search with visitors who return by bookmark | Head of Marketing, Finance | GA's report over-credits Organic Search by 15.4 points and under-credits returning visitors by 16.9 |
| 3 | **Keep Paid Search and set maximum bids from the value of a click**, about $0.80–$1.20 at a 50% margin | Paid media | A click is worth $1.60–$2.50 under every model tested |
| 4 | **Run a 4–6 week Display holdout** (hold back 10–20% of audiences or regions) and measure extra purchases, including corporate orders | Paid media | 91% of Display's reported revenue comes from visits that may not be ad clicks |
| 5 | **Retarget only high-scoring first-time visitors**: include a score band only if its cost per visitor is below that band's value (top 1%: up to $0.70; 1–2%: $0.18; 5–20%: $0.07). **Keep a 10% holdout** to measure real lift | Retargeting lead | Beyond the top 20%, a visitor is worth less than one cent. The 10% lift used here comes from published experiments, not from this store |
| 6 | **Invest in returning visitors and corporate buyers**: email or CRM for past visitors, plus a direct sales path for bulk orders | Head of Marketing | Returning visitors bring 61% of revenue; 51 orders over $1,606 make up 29% of outside revenue |
| 7 | **Don't pay for YouTube promotion or affiliates to drive sales** | Paid media | YouTube's Oct–Nov 2016 traffic spike brought about 98,000 visits and no purchases; Affiliates made 9 sales all year |

## How confident are we?

- **High:** the employee finding (confirmed against Google's unredacted referrers), Organic Search being over-credited (true under every model), and the retargeting ranking (tested once, on months the model never saw).
- **Moderate:** the value of a Paid Search click, which assumes a 50% margin because the real margin isn't in the data.
- **Low:** Display's value, and the retargeting lift (borrowed from published experiments). Recommendations 4 and 5 therefore include tests that would measure them directly.

**Limits.** Every attribution model describes the paths customers took, not what each channel *caused*; only experiments can measure that. The data has no ad costs, visitors are tracked per device (so journeys across devices are split), and the period is 2016–17 on the older Universal Analytics schema. The methods carry over to GA4.

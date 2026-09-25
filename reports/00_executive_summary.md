# Executive summary: where should the Google Merchandise Store spend its next marketing dollar?

**For:** Head of Marketing · **By:** Pann Phetra · **Data:** Google Analytics 360 export of the Google Merchandise Store, Aug 2016 – Aug 2017 (903,653 sessions) · **Full analysis:** [README](../README.md)

## Bottom line

1. **Don't move channel budget on GA's channel report.** It likely over-credits Organic Search by up to about **15 points of purchases**. It also credits visitors returning directly, who bring **34% of purchases** and 44% of revenue (with the largest purchases capped), to earlier campaigns.
2. **Keep Paid Search, but treat its value as a ceiling.** A click is credited with **$1.56–$2.43**, so at a 50% gross margin bid at most **$0.78–$1.21**, and less if some buyers would have come anyway. Every purchase with a readable keyword came from a search for the store or its brand.
3. **Display's reported revenue was mostly one existing corporate buyer.** Without it, a Display click is credited with **$2.84–$3.87** under every attribution rule. **Hold its budget and test what it adds.**
4. **Retargeting should be small and narrow.** The model's top 10% of first-time visitors includes **72% of the outside customers who come back to buy within 30 days** (61% for a simple two-line rule), but retargeting it is worth only about **$710–$740 a month** in extra gross profit before ad costs.

## Two things to know about the data

- **About 41% of the store's revenue comes from Google employees**, not from marketing. They arrive through Google's internal store link, and I report them as their own segment; every marketing figure here excludes them. The public version of the data hides the internal link, but Google's own training copy shows it, which confirmed the employee filter (99% precision).
- **One corporate buyer, the key account, is reported on its own, too.** A single outside visitor, with 278 visits from one office desktop, made 16 purchase sessions worth **$128,413**: 15% of outside revenue in the analysis period, and half of all bulk revenue. It was already buying before its only Display click, and GA's campaign carry-over then labelled its next 15 purchases Display. Its orders reflect an existing relationship, not a channel.

## Recommendations

| When | Action | Owner | At stake |
|---|---|---|---|
| Now | **Report Google employees and the key account as their own segments** | Web analytics | 41% of revenue (employees); $128k (key account) |
| Now | **Show a multi-touch view beside GA's channel report**, with the Organic Search range | Head of Marketing, Finance | Up to ~15 points of purchase credit |
| Now | **Split brand from non-brand search**, and keep bids under the value ceiling ($0.78–$1.21 at most) | Paid media | $1.56–$2.43 attributed per click |
| Now | **Review any spend on YouTube promotion and affiliates** | Paid media | ~98,000 YouTube visits in Oct–Nov 2016, no purchases; 9 affiliate sales all year |
| Test | **Retarget only the top-scored first-time visitors**, and measure the lift with a 50/50 holdout of the top 20% for 12 months | Retargeting lead | $710–$1,025 a month |
| Test | **Hold Display's budget** and run a 12-week 50/50 holdout measured on site visits | Paid media | Display's ~$17k of revenue a year |
| Explore | **Retention:** email and reminders for past visitors | Head of Marketing | 34% of purchases |
| Explore | **A direct sales path for corporate (bulk) buyers** | Head of Marketing, Sales | $248,552 of bulk purchase sessions, half of it one account |

For retargeting, include a score band only if its cost per visitor is below what the band is worth: up to $0.70 for the top 1%, $0.18 for the next 1%, $0.11 for 2–5%, and $0.07 for 5–20%. Beyond the top 20%, a visitor is worth less than a cent.

## Decisions needed

1. **Report employees and the key account on their own.** A reporting change with no spend. Until it's made, every channel number mixes in staff and one corporate buyer.
2. **Run the 12-month 50/50 retargeting test of the top 20%.** About 9,200 top-scored first-time visitors a month are split between retargeted and held out. The test detects a purchase lift of about 14% or more, and return visits give an early read after about 1.5 months. Holding out only 10% would have just a 24% chance of detecting a 10% lift within a year.
3. **Hold Display's budget and test it on site visits for 12 weeks.** A test on purchases can't work: Display is credited with about 2 purchases a week, too few to detect even if every one were caused by the ads.
4. **Split brand from non-brand search and test pausing brand ads.** Purchases that follow a search for the store's own name are the ones most likely to happen without the ad.

## How confident is this?

- **High:** the employee finding (confirmed against Google's unredacted referrers); the key account's timeline (it bought before its only Display click); and the retargeting ranking (tested once on months the model never saw, and 9–11 points ahead of a two-line rule).
- **Moderate:** the size of the Organic Search over-credit and the returning-visitor share (the direction holds under every reading, but the size depends on how GA's direct-return flag is read); and the retargeting dollar value (it assumes a 10% lift and a 50% margin).
- **Low:** how many Paid Search sales the ads actually cause; what Display adds; and the retargeting lift (borrowed from published experiments). The tests above would measure them.

**Limits.** Every attribution model describes the paths customers took, not what each channel *caused*; only experiments can measure that. The data has no ad costs, visitors are tracked per device (so journeys across devices are split), revenue includes tax and shipping, and the period is 2016–17 on the older Universal Analytics schema. The methods carry over to GA4.

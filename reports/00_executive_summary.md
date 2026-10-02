# Executive summary: where should the next marketing dollar go?

**To:** Head of Marketing, Google Merchandise Store · **From:** Pann Phetra · **Data:** the store's Google Analytics 360 export, Aug 2016 – Aug 2017 (903,653 sessions) · **Full analysis:** [README](../README.md)

*A portfolio case study for my Google Data Analytics capstone, built on Google's public GA360 sample data. The Head of Marketing is the intended reader, not a client, and the project is not affiliated with or endorsed by Google. Analysis designed and directed by me; code written with AI assistance.*

You asked where the store's next marketing dollar should go. My answer: not where GA's channel report points, at least not yet. Two groups distort every channel number, Google's own employees and one corporate buyer, and the channels with real money at stake need a test before their budgets move. Here's what I found and what I'd do.

## The bottom line

1. **Don't move budget on GA's channel report.** It likely gives Organic Search up to about **15 points of purchases** it didn't earn. When visitors who first came from a campaign come back on their own, by bookmark or typed address, the report credits that campaign. Visitors returning on their own bring **34% of purchases** and 44% of revenue (with the largest purchases capped).
2. **Keep Paid Search, but treat its value as a ceiling.** A click is credited with **$1.56–$2.43** of revenue, so at a 50% gross margin I'd bid at most **$0.78–$1.21**, and less if some buyers would have come anyway. Every purchase with a readable keyword was on a brand keyword (the store's or Google's name).
3. **Display's reported revenue was mostly one existing corporate buyer.** Without it, a Display click is credited with **$2.84–$3.87** under every attribution rule. **Hold its budget and test what it adds.**
4. **Keep retargeting small and narrow.** The model's top 10% of first-time visitors includes **71% of the outside customers who come back to buy within 30 days** (a simple two-line rule gets 61%), but retargeting them is worth only up to about **$690–$745 a month** in extra gross profit, if the ads reach everyone in the audience (before ad costs).

## Two things to know about the data

- **About 41% of the store's revenue comes from Google's own employees, not from marketing.** They arrive through Google's internal store link, so I report them as their own segment, and every marketing figure here leaves them out. The public data hides that link, but Google's own training copy shows it, which let me confirm the employee filter (99% precision).
- **One corporate buyer, which I call the key account, is reported on its own, too.** A single outside visitor, 278 visits from one office desktop, made 16 purchase sessions worth **$128,413**: 15% of outside revenue in the period and half of all bulk revenue. It was already buying before its only Display click, and GA's campaign carry-over then labeled its next 15 purchases Display. Its orders reflect an existing relationship, not a channel.

## What I'd do

| When | Action | Owner | At stake |
|---|---|---|---|
| Now | **Report Google employees and the key account as their own segments** | Web analytics | 41% of revenue (employees); $128k (key account) |
| Now | **Show a multi-touch view beside GA's channel report**, with the Organic Search range | Head of Marketing, Finance | Up to ~15 points of purchase credit |
| Now | **Split brand from non-brand search**, and keep bids under the value ceiling ($0.78–$1.21 at most) | Paid media | $1.56–$2.43 attributed per click |
| Now | **Review any spend on YouTube promotion and affiliates** | Paid media | ~98,000 YouTube visits in Oct–Nov 2016, no purchases; 9 affiliate sales all year |
| Test | **Retarget only the top-scored first-time visitors**, and measure the lift with a 50/50 holdout of the top 20% for 12 months | Retargeting lead | Up to about $690–$950 a month |
| Test | **Hold Display's budget** and run a 12-week 50/50 holdout of Display's own audience, measured on site visits (powered only if that audience makes at most about 500–1,000 visits a week) | Paid media | Display's ~$17k of revenue a year |
| Explore | **Retention:** email and reminders for past visitors | Head of Marketing | 34% of purchases |
| Explore | **A direct sales path for corporate (bulk) buyers** | Head of Marketing, Sales | $248,552 of bulk purchase sessions, half of it one account |

For retargeting, include a score band only if reaching one of its visitors costs less than the band is worth: up to $0.61 for the top 1%, $0.21 for the next 1%, $0.12 for 2–5%, $0.07 for 5–10%, and $0.05 for 10–20%. These bands flag employees using the whole year; without that hindsight each value moves by a few cents, and the 2–5% and 5–10% bands can swap order; taken together, those visitors are worth about $0.09–$0.11 each. Beyond the top 20%, a visitor is worth about a cent or less.

## Decisions I need from you

1. **Report employees and the key account on their own.** It's a reporting change with no spend. Until it's made, every channel number mixes in staff and one corporate buyer.
2. **Run a 12-month 50/50 retargeting test of the top 20%.** About 9,000 top-scored first-time visitors a month would be split between retargeted and held out. The test detects a purchase lift of about 14% or more (a 10% lift has a 52% chance of being detected within a year; about 23 months gives 80%), and return visits give an early read about 2.5 months after launch (about 1.5 months of enrollment plus the 30-day follow-up). Holding out only 10% would give just a 23% chance of detecting a 10% lift within a year.
3. **Hold Display's budget and test it on site visits for 12 weeks.** A test on purchases can't work: Display is credited with about 2 purchases a week, too few to detect an effect even if the ads caused every one. Split Display's own audience lists, not all site traffic: across the whole site the visit test has only 17–30% power. It reaches 80% only if the randomized audience (both arms together) makes at most about 500–1,000 visits a week and the ads cause nearly all the visits GA credits to Display.
4. **Split brand from non-brand search, and test pausing brand ads.** Purchases on brand keywords are the ones most likely to happen without the ad.

## How sure I am

- **High:** the employee finding (confirmed against Google's unredacted referrers), the key account's timeline (it bought before its only Display click), and the retargeting ranking (tested on later first visits the model never saw in training or tuning, and 7–10 points ahead of a two-line rule).
- **Moderate:** the size of the Organic Search over-credit and the returning-visitor share (the direction holds under every reading, but the size depends on how GA's direct-return flag is read), and the retargeting dollar value (it assumes a 10% lift, a 50% margin and ads that reach everyone in the audience; the published experiments behind the lift measured it on people who saw an ad).
- **Low:** how many Paid Search sales the ads actually cause, what Display adds, and the retargeting lift (borrowed from published experiments). The tests above are designed to measure all three.

**What this can't tell you.** Attribution describes the paths customers took, not what each channel *caused*; only experiments can measure that. The data also has no ad costs, tracks visitors per device (so journeys across devices are split), includes tax and shipping in revenue, and covers 2016–17 on the older Universal Analytics schema. The methods carry over to GA4.

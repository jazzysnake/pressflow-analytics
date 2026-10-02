# 0004 — Forum data source: Portfolio.hu OTP thread

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Primary sentiment source is the Portfolio.hu OTP retail thread. Verified via
recon (server-rendered, scrapeable).

## Decision

Scrape `https://forum.portfolio.hu/topics/otp-reszvenyesek-ide/5224` with a
polite, resumable crawler.

**Verified facts (for implementers):**

- Pagination: `?oldal=N`; `oldal=1` = oldest page, newest = max `oldal`
  (~67,037 on 2026-10-01, grows daily). ~20–24 posts/page.
- Deep pages carry absolute timestamps (`2010. 11. 25. 10:17`); page 1 uses
  relative `ma, 21:09` / `tegnap, …`. Structure consistent back to ≥2010.
- Parsing markers:

| Field | Marker |
|---|---|
| author | `a.username` text |
| raw timestamp | `span.date` `title` attr |
| display post id | `.id` div `#1340744` |
| hsz_id | vote span `data-hsz_id` |
| user_id | vote span `data-user_id` |
| reply-to display id | "Előzmény" `a` anchor text |
| reply-to hsz_id | "Előzmény" `a` href `/prev/5224/<id>` |
| upvotes / downvotes | `span[data-vote="1"/"-1"] span` |
| text | `div.text > div` |

- Measured volume: ~100–125 posts/day normal, 384 on the 2026-10-01 crash day,
  ~29k posts/year, peak ~17/hr at 9:00, ~5–10/hr midday.
- Crawl strategy: binary-search `oldal` for a target date, walk newest→oldest,
  ~0.35s delay, retry+backoff, checkpoint/resume, dedupe on `hsz_id`.

## Consequences

- 10 years ≈ ~15k pages ≈ ~2 hrs of polite scraping.

## References

- Tickets: T1.1–T1.5

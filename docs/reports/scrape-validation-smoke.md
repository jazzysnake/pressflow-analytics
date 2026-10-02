# Scrape validation report

**Scope:** smoke scope — ~100 newest pages (oldal 67048→66949); full report to follow after the full 10-year crawl

**Dataset:** `data/processed/posts.parquet`

**Verdict:** PASS

## Summary

| Metric | Value |
|---|---|
| Posts | 1932 |
| Distinct pages | 100 |
| Posts/page (min / median / max) | 11 / 20 / 34 |
| First post (Budapest) | 2026-09-17T10:42:00+02:00 |
| Last post (Budapest) | 2026-10-02T23:35:00+02:00 |
| Coverage | 16 days |
| Gaps > 3 days | 0 |
| Duplicate `hsz_id` | 0 |
| Posts/day (mean / median / max) | 128.8 / 107 / 400 |
| `ts` parse failures | 0 |

## Missing-field rates

| Column | Missing / null rate |
|---|---|
| `post_id` | 0.00% |
| `hsz_id` | 0.00% |
| `author` | 0.00% |
| `user_id` | 0.00% |
| `ts` | 0.00% |
| `reply_to_post_id` | 28.21% |
| `reply_to_hsz_id` | 28.21% |
| `upvotes` | 0.00% |
| `downvotes` | 0.00% |
| `text` | 0.00% |
| `page` | 0.00% |

## Thresholds (T1.5 DoD)

| Check | Threshold | Result |
|---|---|---|
| non-null `text` | ≥ 95% | 100.00% (ok) |
| non-null `ts` | ≥ 95% | 100.00% (ok) |
| duplicate `hsz_id` | = 0 | 0 (ok) |

## Outliers & gaps

Pages outside [15, 30] posts (expect ~20–24):

- `oldal=66961`: 13 posts
- `oldal=66962`: 11 posts
- `oldal=66974`: 14 posts
- `oldal=66976`: 14 posts
- `oldal=66980`: 11 posts
- `oldal=66985`: 14 posts
- `oldal=67048`: 34 posts

No calendar gaps > 3 days.

No threshold violations.

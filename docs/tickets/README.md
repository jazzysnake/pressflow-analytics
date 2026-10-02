# Tickets

One file per task, grouped by epic folder. This index is the orchestration board.

**Status legend:** Not started · In progress · Done · Blocked · Gated

## Epics

| Epic | Folder | Goal | Tasks | Status |
|---|---|---|---|---|
| 00 | `epic-00-foundation/` | uv project, config, schemas | T0.1–T0.3 | Not started |
| 01 | `epic-01-forum-scraper/` | Scrape posts → Parquet | T1.1–T1.5 | Not started |
| 02 | `epic-02-price-data/` | OTP/SPY/USDHUF → Parquet | T2.1–T2.3 | Not started |
| 03 | `epic-03-sentiment-scoring/` | Off-the-shelf sentiment scores | T3.1–T3.5 | Not started |
| 04 | `epic-04-signal-calibration/` | Daily signal + calibration | T4.1–T4.5 | Not started |
| 05 | `epic-05-backtest/` | Engine + significance | T5.1–T5.6 | Not started |
| 06 | `epic-06-phase-2/` | Author model, intraday, cross-sectional | T6.1–T6.3 | Gated |

## Order & parallelization

- **Epic 01 and 02 are independent** — run in parallel.
- **T3.1** (hand-labeling prep) can start immediately alongside Epic 01.
- **T4.3 is the first kill gate** — run it before building the full backtest (Epic 05).
- **Epic 06 only starts if Epic 05 passes the kill criteria** (ADR-0014).

## Ticket template

Every ticket file contains: Epic, Status, Priority, Size, Depends on, Blocks,
Context, Objective, Implementation notes, Acceptance criteria (DoD), References.

Sizes: S (<0.5d) · M (0.5–1.5d) · L (1.5–3d).

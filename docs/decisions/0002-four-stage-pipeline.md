# 0002 — Four-stage pipeline

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Need a clean data flow from raw forum HTML to a backtested signal.

## Decision

Four stages, data passed as Parquet:

1. **Scrape** — forum posts → `posts.parquet` (Epic 01).
2. **Prices** — OTP/SPY/USDHUF → `prices.parquet` (Epic 02).
3. **Sentiment** — off-the-shelf scores → `scores.parquet` (Epic 03).
4. **Signal + backtest** — daily aggregation → calibration → long-only backtest
   with significance (Epics 04–05).

Epics 01 and 02 are independent and parallelizable.

## Consequences

- Clear stage boundaries and schemas; each stage independently testable.
- Natural kill gate at T4.3 (before the expensive full backtest).

## References

- `docs/tickets/README.md`
- ADR-0015 (schemas & tooling)

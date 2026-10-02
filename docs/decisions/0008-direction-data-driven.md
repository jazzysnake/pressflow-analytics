# 0008 — Direction decided by data (D3)

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

No prior on whether the crowd is a momentum or contrarian indicator. Daily
price movement isn't assumed to mean anything; the crowd's aggregated bets
might carry signal either way.

## Decision

Let the data decide: at T4.3, compute rank-IC vs next-day return in **both**
directions and pick whichever wins. No directional prior is baked into the
backtest.

## Consequences

- The relationship check is symmetric and honest; avoids confirmation bias.

## References

- Ticket T4.3

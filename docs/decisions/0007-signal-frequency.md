# 0007 — Signal frequency: daily (D2)

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Question: hourly or 3×/day to get more observations? Measured post volume says
no.

## Decision

**Daily** primary signal. Measured: normal hours have ~5–17 posts; one loud
poster would dominate an hourly bucket. 3×/day (~30–40 posts) is marginal, and
**subdividing time does not create independent evidence** (autocorrelation).

Daily features: engagement-weighted mean score, post count, disagreement
(std + entropy), rolling z-score of level.

Only intraday idea worth revisiting later: a 2-bucket overnight-vs-intraday
split, gated on the daily signal showing an edge (T6.2).

## Consequences

- ~2,500 daily observations over 10 years is the real lever for small-N.

## References

- ADR-0003 (statistical method), ADR-0006 (window)

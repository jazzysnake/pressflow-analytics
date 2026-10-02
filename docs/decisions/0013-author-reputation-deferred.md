# 0013 — Author reputation model deferred (D8)

- **Status:** Accepted (deferred)
- **Date:** 2026-10-01

## Context

"Wisdom of the crowd" could be sharpened by weighting posters by track record
(hierarchical Bayesian latent skill/contrarian tendency).

## Decision

Defer to **Phase 2** (T6.1), gated on Epic 05 passing the kill criteria. Phase 1
uses aggregate-only signal.

## Consequences

- Faster first answer; author model only built if the plain aggregate shows a
  signal worth sharpening.

## References

- Ticket T6.1, ADR-0014

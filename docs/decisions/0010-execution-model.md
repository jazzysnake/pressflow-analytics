# 0010 — Execution model & no lookahead (D5)

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Must be tradeable and leakage-free.

## Decision

- Signal = posts from day **t−1** (Europe/Budapest), full day.
- Position held day **t**, **close-to-close**.
- No information from day t may enter the t−1 signal; enforced by an automated
  leakage test (T4.2).
- Open-execution refinement deferred (only if an edge is found).

## Consequences

- Conservative and realistic; slightly optimistic (close-to-close entry) but
  accepted for exploration.

## References

- Ticket T4.2

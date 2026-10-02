# 0012 — Transaction costs (D7)

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Looked up (IBKR IBIE pricing, Aug 2026).

## Decision

- IBKR commission: **0.05%/side** (min €1.25 tiered / €3 fixed) → 0.10% round
  trip + small exchange pass-through.
- OTP spread ~0.05–0.15%.
- Realistic OTP round trip via IBKR: **~0.2–0.4%**; local Hungarian brokers
  ~0.4–0.9%.

**Charged in backtest:** base **0.3%** round trip on turnover, with sensitivity
at **0.15%** (optimistic) and **0.6%** (conservative/local).

## Consequences

- Edge must survive at least the base case.

## References

- Ticket T5.1, T5.5

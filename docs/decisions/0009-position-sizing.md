# 0009 — Position sizing: fractional Kelly, long-only (D4)

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Long-only (no uncapped loss). Raw Kelly is unstable when p̂ is estimated with
noise.

## Decision

- p̂ shrunk toward 0.5: `p̃ = 0.5 + γ·(p̂ − 0.5)`, γ default **0.5** (sweep 0.3/0.5).
- Long-only: `p̃ ≤ 0.5` → cash; else `position = min(cap, k·(2·p̃ − 1))`.
- `k = ¼` default (sweep ⅛/¼/½), `cap = 1.0` (no leverage, no short).

## Consequences

- Positions in [0, 1]; sizing scales with conviction; guardrails prevent blow-up.

## References

- ADR-0003 (calibration), ADR-0012 (costs), Ticket T5.1

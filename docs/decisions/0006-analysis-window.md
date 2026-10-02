# 0006 — Analysis window: 10 years (D1)

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Daily observations are the unit, and 3 years (~750 obs) is too few once
autocorrelation is accounted for (N_eff ≈ 320 at ρ≈0.4). The forum offers
~20 years.

## Decision

Default window **10 years** (~2,500 daily obs; N_eff ≈ 1,000+), with the recent
**3–5 years kept as a robustness sub-period**. Scraping cost is ~15k pages ≈
2 hrs — cheap enough that "more history" beats "finer time slicing".

## Consequences

- More regimes included (good for robustness, but older market/forum behavior
  may differ) — hence the sub-period check in T5.5.

## References

- ADR-0007 (frequency), ADR-0003 (effective N)

# 0014 — Kill criteria

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Exploration-first, kill fast. Define "no signal" before looking, to avoid
fooling ourselves.

## Decision

Kill or pivot if **any** of:

1. Best-signal |rank-IC| vs next-day return **< ~0.05** (gate at T4.3),
2. after-cost Sharpe **≤ SPY Sharpe** over the same window,
3. **block-bootstrap p-value** of the edge **> 0.10**,
4. max drawdown or turnover makes it untradeable.

First hard gate is T4.3 (before building the full backtest). Final decision is
T5.6.

## Consequences

- Disciplined go/no-go; p-value guards against autocorrelation-induced luck.

## References

- Tickets T4.3, T5.4, T5.6

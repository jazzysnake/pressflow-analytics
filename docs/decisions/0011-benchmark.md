# 0011 — Benchmark: S&P 500 total return in USD (D6)

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Success bar: beat "park it and forget it" (S&P 500 buy-and-hold) after costs by
a wide margin.

## Decision

- Primary benchmark: **SPY total return (Adj Close) in USD**.
- OTP converted to USD via `USDHUF`: `otp_usd = otp_adj_close_huf ÷ usdhuf` (USDHUF is HUF per 1 USD, so it's a division — not multiplication).
- Secondary benchmarks: OTP buy-and-hold (USD), cash.

## Consequences

- HUF→USD FX (~0.03%) only matters if actually buying SPY; noted, not charged
  to the strategy.

## References

- ADR-0005 (prices), Tickets T2.2, T5.2

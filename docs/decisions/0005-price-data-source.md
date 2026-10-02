# 0005 — Price data source: yfinance

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Need OTP, the S&P 500 benchmark, and HUF/USD FX. Verified working (uv, Python
3.12).

## Decision

Use `yfinance` daily data:
- `OTP.BD` — OTP, Budapest (Close + Adj Close)
- `SPY` — S&P 500 ETF (Adj Close = total return)
- `USDHUF=X` — FX for USD-denominated comparison
- `^GSPC` — index (reference only)

Fetch with `auto_adjust=False` so raw `Close` and `Adj Close` are both kept.
Use **Python 3.12** (3.14 lacks wheels for the ML stack).

## Consequences

- Known simplification: dividend adjustment for `OTP.BD` may be imperfect —
  QA in T2.3.

## References

- Tickets: T2.1–T2.3
- ADR-0015 (tooling)

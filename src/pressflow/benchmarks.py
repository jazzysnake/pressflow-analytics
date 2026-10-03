"""T5.2 — benchmark equity curves (ADR-0011).

Buy-and-hold curves on the same calendar as the strategy:

* ``spy``  — ``spy_usd / spy_usd[0]`` (S&P 500 total return in USD, primary);
* ``otp``  — ``otp_usd / otp_usd[0]`` (OTP in USD, secondary);
* ``cash`` — constant 1.0.

All three are reindexed to the supplied ``dates`` (the backtest calendar).
"""

from __future__ import annotations

import pandas as pd

from pressflow.schema import DATE, OTP_USD, SPY_USD

#: Benchmark curve names / dict keys.
SPY = "spy"
OTP = "otp"
CASH = "cash"
BENCHMARK_NAMES: tuple[str, ...] = (SPY, OTP, CASH)

_REQUIRED_PRICES_COLUMNS = (DATE, SPY_USD, OTP_USD)


def benchmark_curves(
    prices: pd.DataFrame, dates: pd.Index | pd.Series | pd.DatetimeIndex
) -> dict[str, pd.Series]:
    """Return indexed-to-1.0 SPY/OTP/cash curves on ``dates``."""
    missing = [c for c in _REQUIRED_PRICES_COLUMNS if c not in prices.columns]
    if missing:
        raise KeyError(f"prices missing columns: {missing}")

    ordered = prices.sort_values(DATE)
    index = pd.DatetimeIndex(ordered[DATE])
    spy_usd = ordered[SPY_USD].to_numpy(dtype="float64")
    otp_usd = ordered[OTP_USD].to_numpy(dtype="float64")

    target = pd.DatetimeIndex(dates)
    curves = {
        SPY: pd.Series(spy_usd / spy_usd[0], index=index),
        OTP: pd.Series(otp_usd / otp_usd[0], index=index),
        CASH: pd.Series(1.0, index=index),
    }
    return {name: curve.reindex(target) for name, curve in curves.items()}


__all__ = [
    "SPY",
    "OTP",
    "CASH",
    "BENCHMARK_NAMES",
    "benchmark_curves",
]

"""T4.2 — join signals with prices, leakage-free (ADR-0010).

The tradeable relationship is *signal from day t−1 → return on day t*. This
module builds the aligned signal→return ``panel``:

* for every BÉT trading day ``t`` it attaches the ``signals`` row of the
  **previous** trading day (``prices.date.shift(1)``);
* ``ret_log = ln(otp_usd[t] / otp_usd[t−1])``;
* ``ret_simple = otp_usd[t] / otp_usd[t−1] − 1``;
* ``up = ret_simple > 0``.

A guard (:func:`assert_no_leakage`) verifies the signal date is *strictly*
before the return date, so no day-``t`` information can enter the day-``t``
decision. The row count equals ``len(prices) − 1``.

Usage::

    uv run python -m pressflow.join_prices
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.schema import (
    BEAR_WMEAN,
    COUNT,
    DATE,
    DISAGREEMENT_STD,
    ENTROPY,
    OTP_USD,
    SCORE_WMEAN,
    SIGNALS_COLUMNS,
    Z_SCORE,
)

logger = logging.getLogger(__name__)

SIGNALS_FILENAME = "signals.parquet"
PRICES_FILENAME = "prices.parquet"
PROCESSED_FILENAME = "panel.parquet"

#: Name of the (previous trading day) signal-date column in the panel.
SIGNAL_DATE = "signal_date"
RET_LOG = "ret_log"
RET_SIMPLE = "ret_simple"
UP = "up"

#: Signal feature columns attached to each return day.
SIGNAL_FEATURES: tuple[str, ...] = SIGNALS_COLUMNS[1:]

#: Exact output column order of the panel.
PANEL_COLUMNS: tuple[str, ...] = (
    DATE,
    SIGNAL_DATE,
    SCORE_WMEAN,
    BEAR_WMEAN,
    COUNT,
    DISAGREEMENT_STD,
    ENTROPY,
    Z_SCORE,
    RET_LOG,
    RET_SIMPLE,
    UP,
)

_REQUIRED_PRICES = (DATE, OTP_USD)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def join_signals_prices(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Pair each BÉT day's return with the *previous* day's signal row.

    ``prices`` needs ``date`` and ``otp_usd``; ``signals`` needs ``date`` plus the
    signal features. Rows are preserved from the return side (left join) so the
    output always has ``len(prices) − 1`` rows, regardless of signal coverage.
    """
    _require_columns(prices, _REQUIRED_PRICES, "prices")
    _require_columns(signals, SIGNALS_COLUMNS, "signals")

    sorted_prices = prices[[DATE, OTP_USD]].sort_values(DATE).reset_index(drop=True)
    sorted_signals = signals.sort_values(DATE).reset_index(drop=True)

    work = pd.DataFrame(
        {
            DATE: sorted_prices[DATE],
            SIGNAL_DATE: sorted_prices[DATE].shift(1),
            "_otp": sorted_prices[OTP_USD],
            "_prev_otp": sorted_prices[OTP_USD].shift(1),
        }
    )
    # The first trading day has no previous close, hence no return.
    work = work.iloc[1:].reset_index(drop=True)

    work[RET_LOG] = np.log(work["_otp"] / work["_prev_otp"])
    work[RET_SIMPLE] = work["_otp"] / work["_prev_otp"] - 1.0
    work[UP] = work[RET_SIMPLE] > 0.0

    previous = sorted_signals.rename(columns={DATE: SIGNAL_DATE})
    panel = work[[DATE, SIGNAL_DATE, RET_LOG, RET_SIMPLE, UP]].merge(
        previous[[SIGNAL_DATE, *SIGNAL_FEATURES]], on=SIGNAL_DATE, how="left"
    )
    if panel[COUNT].notna().all():
        panel[COUNT] = panel[COUNT].astype("int64")
    return panel[list(PANEL_COLUMNS)]


def assert_no_leakage(panel: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Raise ``AssertionError`` if any signal comes from day ``t`` or later.

    Every non-null ``signal_date`` must be exactly the previous trading day of
    the corresponding ``date`` (and therefore strictly earlier).
    """
    _require_columns(prices, (DATE,), "prices")
    _require_columns(panel, (DATE, SIGNAL_DATE), "panel")

    expected = (
        prices[[DATE]]
        .sort_values(DATE)
        .reset_index(drop=True)[DATE]
        .shift(1)
        .iloc[1:]
        .reset_index(drop=True)
    )
    got = panel.reset_index(drop=True)
    if len(got) != len(expected):
        raise AssertionError(
            f"panel has {len(got)} rows, expected {len(expected)} (len(prices)-1)"
        )

    signal_dates = pd.DatetimeIndex(got[SIGNAL_DATE])
    return_dates = pd.DatetimeIndex(got[DATE])
    if (signal_dates >= return_dates).any():
        raise AssertionError("signal_date is not strictly before date (look-ahead!)")

    present = signal_dates.notna()
    if not (signal_dates[present] == expected[present]).all():
        raise AssertionError(
            "signal_date is not the previous trading day for every row"
        )


def previous_trading_days(prices: pd.DataFrame) -> pd.Series:
    """Return the previous trading day for each row (first is ``NaT``)."""
    _require_columns(prices, (DATE,), "prices")
    ordered = prices[[DATE]].sort_values(DATE).reset_index(drop=True)
    return ordered[DATE].shift(1)


def panel_path(config: Config | None = None) -> Path:
    """Absolute path of the (non-canonical, intermediate) joined panel."""
    config = config or get_config()
    return REPO_ROOT / config.paths.processed / PROCESSED_FILENAME


def save_panel(panel: pd.DataFrame, path: Path) -> None:
    """Write the joined panel to Parquet (dtype-inferred, intermediate artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, preserve_index=False)
    pq.write_table(table, path)


def build_and_save(config: Config | None = None) -> pd.DataFrame:
    """Read signals + prices, join them and write ``data/processed/panel.parquet``."""
    config = config or get_config()
    processed = REPO_ROOT / config.paths.processed
    signals_file = processed / SIGNALS_FILENAME
    prices_file = processed / PRICES_FILENAME
    if not signals_file.is_file():
        raise FileNotFoundError(f"signals not found: {signals_file} (run T4.1 first)")
    if not prices_file.is_file():
        raise FileNotFoundError(f"prices not found: {prices_file} (run T2.2 first)")

    signals = pd.read_parquet(signals_file)
    prices = pd.read_parquet(prices_file)
    panel = join_signals_prices(signals, prices)
    assert_no_leakage(panel, prices)
    out_file = panel_path(config)
    save_panel(panel, out_file)
    print(f"wrote {out_file} ({len(panel)} rows, {len(panel.columns)} columns)")
    return panel


def main() -> None:  # pragma: no cover - thin CLI wrapper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_and_save()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "SIGNALS_FILENAME",
    "PRICES_FILENAME",
    "PROCESSED_FILENAME",
    "SIGNAL_DATE",
    "RET_LOG",
    "RET_SIMPLE",
    "UP",
    "SIGNAL_FEATURES",
    "PANEL_COLUMNS",
    "join_signals_prices",
    "assert_no_leakage",
    "previous_trading_days",
    "panel_path",
    "save_panel",
    "build_and_save",
]

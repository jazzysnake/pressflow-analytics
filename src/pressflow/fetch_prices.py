"""T2.1 — fetch daily OHLCV price history via yfinance (ADR-0005).

Downloads OTP (Budapest), SPY, USDHUF and the S&P 500 index with
``auto_adjust=False`` so both raw ``Close`` and dividend-adjusted ``Adj Close``
are preserved, and writes a flattened, tz-aware raw Parquet artifact to
``data/raw/prices.parquet``.

The download range starts ``window_years + 1`` years before today (ADR-0006):
the extra year is a buffer for the trailing rolling z-score.

Only the transformation helpers are unit-tested; the network download lives in
:func:`fetch_and_save`.

Usage::

    uv run python -m pressflow.fetch_prices
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.schema import BUDAPEST_TZ, DATE

logger = logging.getLogger(__name__)

#: Tickers required by the price panel (ADR-0005).
TICKERS: tuple[str, ...] = ("OTP.BD", "SPY", "USDHUF=X", "^GSPC")

#: Filesystem-safe slug per ticker, used to flatten the yfinance MultiIndex.
TICKER_SLUGS: dict[str, str] = {
    "OTP.BD": "otp_bd",
    "SPY": "spy",
    "USDHUF=X": "usdhuf_x",
    "^GSPC": "gspc",
}

#: yfinance field -> flattened suffix (all fields kept, not just Close).
RAW_FIELDS: dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
}

# Raw column names consumed downstream by T2.2.
RAW_OTP_CLOSE = "otp_bd_close"
RAW_OTP_ADJ_CLOSE = "otp_bd_adj_close"
RAW_USDHUF = "usdhuf_x_close"
RAW_SPY_ADJ_CLOSE = "spy_adj_close"
RAW_SPX_CLOSE = "gspc_close"

#: Default raw artifact basename.
RAW_FILENAME = "prices.parquet"


def window_start(
    window_years: int,
    today: object | None = None,
    buffer_years: int = 1,
) -> pd.Timestamp:
    """Return the first day to download: ``today - (window_years + buffer_years)``.

    The one-year buffer (config-driven ``window_years`` + 1) feeds the trailing
    rolling z-score in later stages.
    """
    if window_years <= 0:
        raise ValueError(f"window_years must be positive, got {window_years}")
    if buffer_years < 0:
        raise ValueError(f"buffer_years must be non-negative, got {buffer_years}")
    if today is None:
        today = pd.Timestamp.now(tz=BUDAPEST_TZ).normalize().tz_localize(None)
    else:
        today = pd.Timestamp(today)
        if today.tz is not None:
            today = today.tz_convert(BUDAPEST_TZ).normalize().tz_localize(None)
    return today - pd.DateOffset(years=window_years + buffer_years)


def normalize_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Return a tz-aware, Europe/Budapest-midnight ``DatetimeIndex``.

    yfinance daily data is normally tz-naive calendar dates; localise those.
    If the source is tz-aware, convert first so the calendar day is Hungarian
    (matching the ``prices`` schema, ADR-0015).
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize(BUDAPEST_TZ)
    else:
        idx = idx.tz_convert(BUDAPEST_TZ)
    return idx.normalize().as_unit("us")


def flatten_download(downloaded: pd.DataFrame) -> pd.DataFrame:
    """Flatten a yfinance multi-ticker download into one tz-aware frame.

    The result has a leading tz-aware ``date`` column plus ``{slug}_{field}``
    columns for every ticker/field combination present (Close *and* Adj Close).
    """
    multi = isinstance(downloaded.columns, pd.MultiIndex)
    out: dict[str, pd.Series] = {}
    for ticker in TICKERS:
        slug = TICKER_SLUGS[ticker]
        for suffix, label in RAW_FIELDS.items():
            if multi:
                if (ticker, label) not in downloaded.columns:
                    continue
                series = downloaded[(ticker, label)]
            else:
                if label not in downloaded.columns:
                    continue
                series = downloaded[label]
            out[f"{slug}_{suffix}"] = series
    frame = pd.DataFrame(out)
    frame.index = normalize_dates(downloaded.index)
    frame.index.name = DATE
    return frame.reset_index()


def download_raw(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str | None = None,
    tickers: tuple[str, ...] = TICKERS,
) -> pd.DataFrame:
    """Download and flatten daily history for ``tickers`` (network call)."""
    kwargs: dict[str, object] = {
        "auto_adjust": False,
        "progress": False,
        "group_by": "ticker",
        "actions": False,
    }
    if end is not None:
        kwargs["end"] = end
    downloaded = yf.download(list(tickers), start=start, **kwargs)
    if len(tickers) == len(TICKERS) and tuple(tickers) == TICKERS:
        return flatten_download(downloaded)
    # Fallback for an arbitrary ticker subset (group_by='ticker' still applies).
    multi = isinstance(downloaded.columns, pd.MultiIndex)
    out: dict[str, pd.Series] = {}
    for ticker in tickers:
        slug = TICKER_SLUGS.get(ticker, ticker.replace(".", "_").replace("^", "").lower())
        for suffix, label in RAW_FIELDS.items():
            key = (ticker, label) if multi else label
            if key not in downloaded.columns:
                continue
            out[f"{slug}_{suffix}"] = downloaded[key]
    frame = pd.DataFrame(out)
    frame.index = normalize_dates(downloaded.index)
    frame.index.name = DATE
    return frame.reset_index()


def sanity_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker row counts and first/last dates for logging/QA."""
    dates = pd.to_datetime(raw[DATE])
    rows = []
    for ticker, slug in TICKER_SLUGS.items():
        col = f"{slug}_close"
        if col not in raw.columns:
            continue
        mask = raw[col].notna()
        if not mask.any():
            continue
        rows.append(
            {
                "ticker": ticker,
                "column": col,
                "rows": int(mask.sum()),
                "first_date": dates[mask].min().date(),
                "last_date": dates[mask].max().date(),
            }
        )
    if not rows:
        return pd.DataFrame(rows)
    summary = pd.DataFrame(rows)
    logger.info(
        "raw window: %s -> %s (%d rows)",
        dates.min().date(),
        dates.max().date(),
        len(raw),
    )
    return summary


def raw_path(config: Config | None = None) -> Path:
    """Absolute path of the raw prices artifact."""
    config = config or get_config()
    return REPO_ROOT / config.paths.raw / RAW_FILENAME


def fetch_and_save(config: Config | None = None) -> pd.DataFrame:
    """Download the full window and write ``data/raw/prices.parquet``."""
    config = config or get_config()
    start = window_start(config.window_years)
    logger.info("downloading %s from %s (buffer 1y)", TICKERS, start.date())
    raw = download_raw(start)
    path = raw_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(path, index=False)
    summary = sanity_summary(raw)
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"wrote {path} ({len(raw)} rows, {len(raw.columns)} columns)")
    return raw


def main() -> None:  # pragma: no cover - thin CLI wrapper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_and_save()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "TICKERS",
    "TICKER_SLUGS",
    "RAW_FIELDS",
    "RAW_OTP_CLOSE",
    "RAW_OTP_ADJ_CLOSE",
    "RAW_USDHUF",
    "RAW_SPY_ADJ_CLOSE",
    "RAW_SPX_CLOSE",
    "RAW_FILENAME",
    "window_start",
    "normalize_dates",
    "flatten_download",
    "download_raw",
    "sanity_summary",
    "raw_path",
    "fetch_and_save",
]

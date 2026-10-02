"""Tests for T2.1 — fetch OHLCV via yfinance (offline transformations)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pressflow.fetch_prices import (
    TICKERS,
    TICKER_SLUGS,
    flatten_download,
    normalize_dates,
    sanity_summary,
    window_start,
)
from pressflow.schema import BUDAPEST_TZ, DATE


def _multi_ticker_frame() -> pd.DataFrame:
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([list(TICKERS), fields])
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    values = np.arange(len(idx) * len(cols), dtype="float64").reshape(len(idx), len(cols))
    return pd.DataFrame(values, index=idx, columns=cols)


def test_window_start_adds_one_year_buffer():
    start = window_start(10, today="2026-10-02")
    assert start == pd.Timestamp("2015-10-02")


def test_window_start_custom_buffer():
    assert window_start(10, today="2026-10-02", buffer_years=0) == pd.Timestamp("2016-10-02")


def test_window_start_rejects_non_positive_window():
    with pytest.raises(ValueError, match="window_years"):
        window_start(0)


def test_normalize_dates_localises_naive_to_budapest_midnight():
    idx = normalize_dates(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert str(idx.tz) == BUDAPEST_TZ
    assert idx.dtype == "datetime64[us, Europe/Budapest]"
    assert (idx == idx.normalize()).all()
    assert idx[0].hour == 0


def test_normalize_dates_converts_tz_aware():
    idx = normalize_dates(pd.DatetimeIndex(["2024-01-02 05:00"], tz="UTC"))
    assert str(idx.tz) == BUDAPEST_TZ
    assert idx[0] == pd.Timestamp("2024-01-02", tz=BUDAPEST_TZ)
    assert idx[0].hour == 0


def test_flatten_download_produces_all_ticker_fields_and_tz_aware_date():
    flat = flatten_download(_multi_ticker_frame())
    assert flat.columns[0] == DATE
    for slug in TICKER_SLUGS.values():
        for suffix in ("open", "high", "low", "close", "adj_close", "volume"):
            assert f"{slug}_{suffix}" in flat.columns
    assert str(pd.DatetimeIndex(flat[DATE]).tz) == BUDAPEST_TZ
    # 4 tickers x 6 fields + date
    assert flat.shape == (3, 25)


def test_flatten_download_keeps_raw_values():
    flat = flatten_download(_multi_ticker_frame())
    assert flat["otp_bd_close"].iloc[0] == 3.0  # OTP.BD is first ticker, Close is 4th
    assert flat["spy_adj_close"].iloc[0] == 10.0  # SPY second block, Adj Close index 4


def test_raw_parquet_roundtrip_keeps_tz_aware_date(tmp_path):
    flat = flatten_download(_multi_ticker_frame())
    path = tmp_path / "prices.parquet"
    flat.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)
    pdt.assert_frame_equal(reloaded, flat)
    assert str(pd.DatetimeIndex(reloaded[DATE]).tz) == BUDAPEST_TZ


def test_sanity_summary_reports_counts_and_dates():
    flat = flatten_download(_multi_ticker_frame())
    summary = sanity_summary(flat)
    assert set(summary["ticker"]) == set(TICKERS)
    assert (summary["rows"] == 3).all()
    assert (summary["first_date"].astype(str) == "2024-01-02").all()

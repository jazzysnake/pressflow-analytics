"""Tests for T2.2 — currency & benchmark alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pressflow.align_prices import (
    REQUIRED_RAW,
    build_panel,
    processed_path,
    save_panel,
    to_budapest_index,
)
from pressflow.fetch_prices import (
    RAW_OTP_ADJ_CLOSE,
    RAW_OTP_CLOSE,
    RAW_SPX_CLOSE,
    RAW_SPY_ADJ_CLOSE,
    RAW_USDHUF,
    normalize_dates,
)
from pressflow.schema import (
    BUDAPEST_TZ,
    DATE,
    OTP_ADJ_CLOSE_HUF,
    OTP_CLOSE_HUF,
    OTP_USD,
    PRICES_COLUMNS,
    PRICES_SCHEMA,
    SPX_CLOSE,
    SPY_ADJ_CLOSE_USD,
    SPY_USD,
    USDHUF,
)

DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


def _raw() -> pd.DataFrame:
    """Synthetic raw frame: OTP missing 2024-01-04 (non-BÉT), USDHUF missing
    2024-01-03 (BÉT), SPY/^GSPC missing 2024-01-08 (BÉT)."""
    return pd.DataFrame(
        {
            DATE: normalize_dates(pd.to_datetime(DATES)),
            RAW_OTP_CLOSE: [100.0, 101.0, np.nan, 103.0, 104.0],
            RAW_OTP_ADJ_CLOSE: [90.0, 91.0, np.nan, 93.0, 94.0],
            RAW_USDHUF: [300.0, np.nan, 310.0, 320.0, 330.0],
            RAW_SPY_ADJ_CLOSE: [400.0, 401.0, 402.0, 403.0, np.nan],
            RAW_SPX_CLOSE: [5000.0, 5010.0, 5020.0, 5030.0, np.nan],
        }
    )


def test_build_panel_columns_and_order_match_schema():
    panel = build_panel(_raw())
    assert list(panel.columns) == list(PRICES_COLUMNS)
    assert len(panel.columns) == 8


def test_build_panel_calendar_is_otp_trading_days():
    panel = build_panel(_raw())
    assert len(panel) == 4  # 2024-01-04 has no OTP bar
    got = set(pd.to_datetime(panel[DATE]).dt.strftime("%Y-%m-%d"))
    assert got == {"2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"}


def test_build_panel_has_no_nan_on_bet_days():
    panel = build_panel(_raw())
    assert not panel.isna().any().any()


def test_build_panel_date_is_tz_aware_budapest_midnight():
    panel = build_panel(_raw())
    dates = pd.DatetimeIndex(panel[DATE])
    assert str(dates.tz) == BUDAPEST_TZ
    assert dates.dtype == "datetime64[us, Europe/Budapest]"
    assert (dates == dates.normalize()).all()


def test_otp_usd_is_adj_close_divided_by_usdhuf():
    panel = build_panel(_raw())
    expected = panel[OTP_ADJ_CLOSE_HUF] / panel[USDHUF]
    np.testing.assert_allclose(panel[OTP_USD], expected)
    # Division, not multiplication (ADR-0011).
    assert panel[OTP_USD].iloc[0] == pytest.approx(90.0 / 300.0)


def test_spy_usd_equals_spy_adj_close():
    panel = build_panel(_raw())
    np.testing.assert_allclose(panel[SPY_USD], panel[SPY_ADJ_CLOSE_USD])


def test_usdhuf_forward_filled_on_missing_bet_day():
    panel = build_panel(_raw())
    by_date = panel.set_index(pd.to_datetime(panel[DATE]).dt.strftime("%Y-%m-%d"))
    # 2024-01-03 had no FX quote; it inherits 2024-01-02's 300.0.
    assert by_date.loc["2024-01-03", USDHUF] == 300.0
    # 2024-01-08 had no SPY/^GSPC bar; they inherit the previous values.
    assert by_date.loc["2024-01-08", SPY_ADJ_CLOSE_USD] == 403.0
    assert by_date.loc["2024-01-08", SPX_CLOSE] == 5030.0


def test_build_panel_raises_on_missing_raw_column():
    raw = _raw().drop(columns=[RAW_SPX_CLOSE])
    with pytest.raises(KeyError, match=RAW_SPX_CLOSE):
        build_panel(raw)


def test_build_panel_raises_when_no_otp_bars():
    raw = _raw()
    raw[RAW_OTP_CLOSE] = np.nan
    with pytest.raises(ValueError, match="no OTP.BD bars"):
        build_panel(raw)


def test_to_budapest_index_normalises_naive_and_aware():
    naive = to_budapest_index(pd.to_datetime(["2024-01-02 13:45"]))
    assert naive[0].hour == 0
    assert str(naive.tz) == BUDAPEST_TZ


def test_required_raw_matches_fetch_columns():
    assert set(REQUIRED_RAW) == {
        RAW_OTP_CLOSE,
        RAW_OTP_ADJ_CLOSE,
        RAW_USDHUF,
        RAW_SPY_ADJ_CLOSE,
        RAW_SPX_CLOSE,
    }


def test_save_panel_writes_canonical_parquet_schema(tmp_path):
    panel = build_panel(_raw())
    path = tmp_path / "prices.parquet"
    save_panel(panel, path)
    assert pq.read_schema(path) == PRICES_SCHEMA
    reloaded = pd.read_parquet(path)
    assert list(reloaded.columns) == list(PRICES_COLUMNS)
    assert str(pd.DatetimeIndex(reloaded[DATE]).tz) == BUDAPEST_TZ


def test_processed_path_uses_config(tmp_path):
    from pressflow.config import load_config, REPO_ROOT

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        (REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert processed_path(cfg) == REPO_ROOT / cfg.paths.processed / "prices.parquet"

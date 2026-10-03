"""Tests for T4.2 — join signals with prices (no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressflow.join_prices import (
    DATE,
    PANEL_COLUMNS,
    RET_LOG,
    RET_SIMPLE,
    SIGNAL_DATE,
    UP,
    assert_no_leakage,
    join_signals_prices,
    previous_trading_days,
    save_panel,
)
from pressflow.schema import (
    BEAR_WMEAN,
    BUDAPEST_TZ,
    COUNT,
    DISAGREEMENT_STD,
    ENTROPY,
    SCORE_WMEAN,
    SIGNALS_COLUMNS,
    Z_SCORE,
)

DATES = pd.DatetimeIndex(
    [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
    ],
    tz=BUDAPEST_TZ,
)
OTP = [100.0, 110.0, 99.0, 108.9, 120.0]


def _prices() -> pd.DataFrame:
    return pd.DataFrame({DATE: DATES, "otp_usd": OTP})


def _signals(cover: int = 4) -> pd.DataFrame:
    """One signal row per day for the first ``cover`` days."""
    rows = []
    for i in range(cover):
        rows.append(
            {
                DATE: DATES[i],
                SCORE_WMEAN: float(i),
                BEAR_WMEAN: 0.1 * i,
                COUNT: i + 1,
                DISAGREEMENT_STD: 0.01 * i,
                ENTROPY: 0.5 + i,
                Z_SCORE: 10.0 * i,
            }
        )
    return pd.DataFrame(rows)


# --- shape / columns -------------------------------------------------------


def test_panel_columns_are_date_signals_returns_up():
    panel = join_signals_prices(_signals(), _prices())
    assert list(panel.columns) == list(PANEL_COLUMNS)
    for col in SIGNALS_COLUMNS[1:]:
        assert col in panel.columns


def test_row_count_is_prices_minus_one():
    panel = join_signals_prices(_signals(), _prices())
    assert len(panel) == len(_prices()) - 1


def test_return_date_is_current_day():
    panel = join_signals_prices(_signals(), _prices())
    assert list(pd.DatetimeIndex(panel[DATE])) == list(DATES[1:])


# --- signal pairing --------------------------------------------------------


def test_signal_date_is_previous_trading_day():
    panel = join_signals_prices(_signals(), _prices())
    assert list(pd.DatetimeIndex(panel[SIGNAL_DATE])) == list(DATES[:-1])


def test_signal_values_come_from_previous_day():
    panel = join_signals_prices(_signals(), _prices())
    # Row 0 pairs day d1's return with d0's signal.
    assert panel[SCORE_WMEAN].iloc[0] == 0.0
    assert panel[Z_SCORE].iloc[0] == 0.0
    assert panel[SCORE_WMEAN].iloc[1] == 1.0
    assert panel[Z_SCORE].iloc[2] == 20.0
    assert panel[COUNT].iloc[3] == 4


def test_returns_match_otp_ratios():
    panel = join_signals_prices(_signals(), _prices())
    ratios = np.asarray(OTP[1:]) / np.asarray(OTP[:-1])
    np.testing.assert_allclose(panel[RET_SIMPLE], ratios - 1.0)
    np.testing.assert_allclose(panel[RET_LOG], np.log(ratios))
    assert panel[RET_LOG].iloc[0] == pytest.approx(np.log(1.1))


def test_up_label_matches_simple_return_sign():
    panel = join_signals_prices(_signals(), _prices())
    assert list(panel[UP]) == [True, False, True, True]
    assert panel[UP].dtype == bool


def test_prices_below_bet_day_are_ignored_order_insensitive():
    prices = _prices().sample(frac=1.0, random_state=1).reset_index(drop=True)
    signals = _signals().sample(frac=1.0, random_state=2).reset_index(drop=True)
    panel = join_signals_prices(signals, prices)
    assert list(pd.DatetimeIndex(panel[DATE])) == list(DATES[1:])
    assert list(pd.DatetimeIndex(panel[SIGNAL_DATE])) == list(DATES[:-1])


# --- coverage --------------------------------------------------------------


def test_missing_prior_signal_keeps_row_with_nan_features():
    panel = join_signals_prices(_signals(cover=2), _prices())
    # All return days are kept even though only the first two have signals.
    assert len(panel) == len(_prices()) - 1
    assert panel[SCORE_WMEAN].iloc[0] == 0.0
    assert panel[SCORE_WMEAN].iloc[1] == 1.0
    assert pd.isna(panel[SCORE_WMEAN].iloc[2])
    # signal_date is still the previous trading day; only the features are null.
    assert panel[SIGNAL_DATE].iloc[2] == DATES[2]


# --- leakage guard ---------------------------------------------------------


def test_no_leakage():
    prices = _prices()
    panel = join_signals_prices(_signals(), prices)
    # Every signal belongs to the previous trading day and is strictly earlier.
    assert_no_leakage(panel, prices)
    signal_dates = pd.DatetimeIndex(panel[SIGNAL_DATE])
    return_dates = pd.DatetimeIndex(panel[DATE])
    assert (signal_dates < return_dates).all()
    assert not signal_dates.isna().any()
    assert list(signal_dates) == list(DATES[:-1])


def test_assert_no_leakage_rejects_same_day_signal():
    prices = _prices()
    panel = join_signals_prices(_signals(), prices)
    bad = panel.copy()
    bad[SIGNAL_DATE] = bad[DATE]  # pretend we used same-day information
    with pytest.raises(AssertionError, match="strictly before"):
        assert_no_leakage(bad, prices)


def test_assert_no_leakage_rejects_wrong_previous_day():
    prices = _prices()
    panel = join_signals_prices(_signals(), prices)
    bad = panel.copy()
    # Row 2's return date is d3; d0 is earlier but not the previous trading day.
    bad.loc[2, SIGNAL_DATE] = DATES[0]
    with pytest.raises(AssertionError, match="previous trading day"):
        assert_no_leakage(bad, prices)


def test_previous_trading_days():
    prev = previous_trading_days(_prices())
    assert pd.isna(prev.iloc[0])
    assert list(pd.DatetimeIndex(prev.iloc[1:])) == list(DATES[:-1])


# --- persistence -----------------------------------------------------------


def test_save_panel_roundtrip(tmp_path):
    panel = join_signals_prices(_signals(), _prices())
    path = tmp_path / "panel.parquet"
    save_panel(panel, path)
    reloaded = pd.read_parquet(path)
    assert list(reloaded.columns) == list(PANEL_COLUMNS)
    assert len(reloaded) == len(panel)

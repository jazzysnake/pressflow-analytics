"""Tests for T5.3 — performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressflow.metrics import (
    TRADING_DAYS,
    align_ret_simple,
    compute_metrics,
    curve_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    trade_count,
    turnover,
)
from pressflow.schema import (
    BUDAPEST_TZ,
    COST,
    DATE,
    EQUITY,
    OTP_USD,
    POSITION,
    RET,
)

DAY0 = pd.Timestamp("2024-01-02", tz=BUDAPEST_TZ)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([DAY0 + pd.Timedelta(days=i) for i in range(n)])


# --- curve metrics ---------------------------------------------------------


def test_flat_curve_has_zero_metrics():
    equity = pd.Series(np.ones(10))
    metrics = curve_metrics(equity, pd.Series(np.zeros(10)))
    assert metrics["total_return"] == 0.0
    assert metrics["cagr"] == 0.0
    assert metrics["sharpe"] == 0.0
    assert metrics["max_drawdown"] == 0.0


def test_total_return_hand_computed():
    equity = pd.Series([1.0, 1.1, 1.21, 1.331])
    assert curve_metrics(equity)["total_return"] == pytest.approx(0.331)


def test_cagr_on_two_year_double_curve():
    n = 504  # two years of 252 trading days
    equity = pd.Series(2.0 ** (np.arange(n) / (n - 1)))
    metrics = curve_metrics(equity)
    assert metrics["n_days"] == 504
    assert equity.iloc[0] == pytest.approx(1.0)
    assert equity.iloc[-1] == pytest.approx(2.0)
    assert metrics["total_return"] == pytest.approx(1.0)
    assert metrics["cagr"] == pytest.approx(2.0 ** (TRADING_DAYS / n) - 1.0)
    assert metrics["cagr"] == pytest.approx(np.sqrt(2.0) - 1.0)


def test_max_drawdown_known_peak_trough():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_of_monotonic_up_curve_is_zero():
    assert max_drawdown(pd.Series([1.0, 1.1, 1.21])) == 0.0


# --- sharpe ----------------------------------------------------------------


def test_sharpe_of_constant_daily_return_series_is_zero():
    # Zero variance -> defined as 0.0 rather than inf/nan.
    assert sharpe_ratio(pd.Series([0.01, 0.01, 0.01, 0.01])) == 0.0


def test_sharpe_hand_computed():
    returns = pd.Series([0.02, 0.0, 0.02, 0.0])
    expected = np.mean(returns) / np.std(returns, ddof=0) * np.sqrt(TRADING_DAYS)
    assert sharpe_ratio(returns) == pytest.approx(expected)
    assert sharpe_ratio(returns) == pytest.approx(np.sqrt(TRADING_DAYS))


def test_sharpe_short_series_is_zero():
    assert sharpe_ratio(pd.Series([0.01])) == 0.0


# --- strategy metrics ------------------------------------------------------


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {DATE: _dates(4), OTP_USD: [100.0, 110.0, 99.0, 108.9]}
    )


def _backtest(positions, ret_simple, equity=None) -> pd.DataFrame:
    n = len(positions)
    ret = np.asarray(positions, dtype="float64") * np.asarray(ret_simple, dtype="float64")
    if equity is None:
        equity = np.cumprod(1.0 + ret)
    return pd.DataFrame(
        {
            DATE: _dates(n + 1)[1:],  # returns start on the second trading day
            POSITION: positions,
            RET: ret,
            COST: np.zeros(n),
            EQUITY: equity,
        }
    )


def test_align_ret_simple_matches_prices():
    prices = _prices()
    df = _backtest([0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
    got = align_ret_simple(df, prices)
    np.testing.assert_allclose(got, [0.1, -0.1, 0.1])


def test_hit_rate_counts_winning_long_days():
    prices = _prices()
    # Long on d1 and d3 (both positive); flat on d2.
    df = _backtest([0.5, 0.0, 0.5], [0.1, -0.1, 0.1])
    assert hit_rate(df, prices) == pytest.approx(1.0)
    # Long everywhere: 2 of 3 days positive.
    df2 = _backtest([0.5, 0.5, 0.5], [0.1, -0.1, 0.1])
    assert hit_rate(df2, prices) == pytest.approx(2.0 / 3.0)


def test_hit_rate_is_zero_when_never_long():
    prices = _prices()
    df = _backtest([0.0, 0.0, 0.0], [0.1, -0.1, 0.1])
    assert hit_rate(df, prices) == 0.0


def test_turnover_is_mean_absolute_position_change():
    df = _backtest([0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
    # |0.5-0| + |0.5-0.5| + |0-0.5| = 1.0 over 3 days
    assert turnover(df) == pytest.approx(1.0 / 3.0)


def test_trade_count_counts_transitions_across_zero():
    df = _backtest([0.0, 0.5, 0.5, 0.0, 0.0, 0.3], [0.0] * 6)
    # cash->long, long->cash, cash->long = 3 crossings.
    assert trade_count(df) == 3


def test_trade_count_zero_for_always_long():
    df = _backtest([0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
    assert trade_count(df) == 1  # single entry from cash


# --- full table ------------------------------------------------------------


def test_compute_metrics_table_structure():
    prices = _prices()
    df = _backtest(
        [0.5, 0.0, 0.5], [0.1, -0.1, 0.1], equity=[1.05, 1.05, 1.1025]
    )
    benchmarks = {
        "spy": pd.Series([1.0, 1.05, 1.02, 1.10]),
        "otp": pd.Series([1.0, 1.10, 0.99, 1.089]),
        "cash": pd.Series([1.0, 1.0, 1.0, 1.0]),
    }
    metrics = compute_metrics(df, benchmarks, prices)

    assert set(metrics) == {"strategy", "spy", "otp", "cash"}
    for name in ("strategy", "spy", "otp", "cash"):
        for key in ("total_return", "cagr", "sharpe", "max_drawdown", "n_days"):
            assert key in metrics[name]
    for key in ("hit_rate", "turnover", "trades"):
        assert key in metrics["strategy"]

    assert metrics["spy"]["total_return"] == pytest.approx(0.10)
    assert metrics["otp"]["total_return"] == pytest.approx(0.089)
    assert metrics["cash"]["sharpe"] == 0.0


def test_compute_metrics_missing_columns_raise():
    prices = _prices()
    df = _backtest([0.5], [0.1]).drop(columns=[EQUITY])
    with pytest.raises(KeyError, match="backtest_df missing columns"):
        compute_metrics(df, {}, prices)

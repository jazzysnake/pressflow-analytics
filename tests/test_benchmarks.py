"""Tests for T5.2 — benchmark curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressflow.benchmarks import (
    BENCHMARK_NAMES,
    CASH,
    OTP,
    SPY,
    benchmark_curves,
)
from pressflow.schema import BUDAPEST_TZ, DATE, OTP_USD, SPY_USD

DATES = pd.DatetimeIndex(
    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], tz=BUDAPEST_TZ
)
SPY_VALS = [400.0, 404.0, 400.0, 420.0]
OTP_VALS = [20.0, 22.0, 19.8, 24.0]


def _prices() -> pd.DataFrame:
    return pd.DataFrame({DATE: DATES, SPY_USD: SPY_VALS, OTP_USD: OTP_VALS})


def test_returns_spy_otp_cash_keys():
    curves = benchmark_curves(_prices(), DATES)
    assert set(curves) == set(BENCHMARK_NAMES)


def test_curves_indexed_to_one_at_first_date():
    curves = benchmark_curves(_prices(), DATES)
    assert curves[SPY].iloc[0] == pytest.approx(1.0)
    assert curves[OTP].iloc[0] == pytest.approx(1.0)
    assert curves[CASH].iloc[0] == pytest.approx(1.0)


def test_spy_and_otp_are_ratios_to_first_price():
    curves = benchmark_curves(_prices(), DATES)
    np.testing.assert_allclose(curves[SPY].to_numpy(), np.asarray(SPY_VALS) / SPY_VALS[0])
    np.testing.assert_allclose(curves[OTP].to_numpy(), np.asarray(OTP_VALS) / OTP_VALS[0])


def test_cash_is_constant_one():
    curves = benchmark_curves(_prices(), DATES)
    np.testing.assert_allclose(curves[CASH].to_numpy(), 1.0)


def test_all_curves_reindexed_to_requested_dates():
    requested = DATES[1:3]
    curves = benchmark_curves(_prices(), requested)
    for curve in curves.values():
        assert list(curve.index) == list(requested)
        assert len(curve) == 2
    # Rebased on the *first* price, then sliced to the requested dates.
    np.testing.assert_allclose(
        curves[SPY].to_numpy(), np.asarray(SPY_VALS[1:3]) / SPY_VALS[0]
    )


def test_missing_price_dates_become_nan_after_reindex():
    requested = DATES.append(pd.DatetimeIndex(["2024-01-08"], tz=BUDAPEST_TZ))
    curves = benchmark_curves(_prices(), requested)
    assert pd.isna(curves[SPY].iloc[-1])


def test_prices_may_be_unsorted():
    shuffled = _prices().sample(frac=1.0, random_state=0).reset_index(drop=True)
    curves = benchmark_curves(shuffled, DATES)
    np.testing.assert_allclose(curves[SPY].to_numpy(), np.asarray(SPY_VALS) / SPY_VALS[0])


def test_missing_columns_raise():
    with pytest.raises(KeyError, match="prices missing columns"):
        benchmark_curves(_prices().drop(columns=[OTP_USD]), DATES)

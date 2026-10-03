"""Tests for T5.1 — backtest engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pressflow.backtest import P_UP, run_backtest
from pressflow.config import get_config
from pressflow.join_prices import RET_SIMPLE
from pressflow.schema import (
    BACKTEST_COLUMNS,
    BUDAPEST_TZ,
    COST,
    DATE,
    EQUITY,
    POSITION,
    RET,
)

DAY0 = pd.Timestamp("2024-01-02", tz=BUDAPEST_TZ)


def _panel(p_up, ret_simple) -> pd.DataFrame:
    n = len(p_up)
    return pd.DataFrame(
        {
            DATE: [DAY0 + pd.Timedelta(days=i) for i in range(n)],
            RET_SIMPLE: ret_simple,
            P_UP: p_up,
        }
    )


def _cfg(**kelly_overrides):
    base = get_config()
    kelly = replace(base.kelly, **kelly_overrides)
    return replace(base, kelly=kelly)


def test_output_columns_match_backtest_schema():
    out = run_backtest(_panel([0.6, 0.7], [0.01, 0.02]), get_config())
    assert list(out.columns) == list(BACKTEST_COLUMNS)


def test_hand_computed_positions_costs_returns_equity():
    # p_up -> p_tilde = 0.5 + 0.5*(p_up-0.5); position = min(1, 0.25*(2*p_tilde-1)).
    panel = _panel([0.9, 0.9, 0.4, 1.0], [0.02, 0.01, 0.0, 0.03])
    out = run_backtest(panel, get_config())  # k=0.25, gamma=0.5, cap=1.0, base=0.003

    # p_up=0.9 -> p_tilde=0.7 -> position=0.1
    expected_position = [0.1, 0.1, 0.0, 0.125]
    np.testing.assert_allclose(out[POSITION], expected_position)

    # cost = 0.003 * |pos_t - pos_{t-1}|, pos_{-1}=0
    expected_cost = [0.003 * 0.1, 0.0, 0.003 * 0.1, 0.003 * 0.125]
    np.testing.assert_allclose(out[COST], expected_cost, rtol=1e-12)

    expected_ret = [
        0.1 * 0.02 - 0.0003,
        0.1 * 0.01 - 0.0,
        0.0 * 0.0 - 0.0003,
        0.125 * 0.03 - 0.000375,
    ]
    np.testing.assert_allclose(out[RET], expected_ret, rtol=1e-12)

    expected_equity = np.cumprod([1.0 + r for r in expected_ret])
    np.testing.assert_allclose(out[EQUITY], expected_equity, rtol=1e-12)
    assert out[EQUITY].iloc[0] == pytest.approx(1.0017)


def test_position_always_in_unit_interval():
    rng = np.random.default_rng(0)
    p_up = rng.uniform(0.0, 1.0, size=500)
    ret = rng.normal(0.0, 0.02, size=500)
    out = run_backtest(_panel(p_up, ret), get_config())
    assert (out[POSITION] >= 0.0).all()
    assert (out[POSITION] <= 1.0).all()


def test_cash_when_p_up_at_or_below_half():
    panel = _panel([0.5, 0.4999, 0.2, 0.0], [0.05, 0.05, 0.05, 0.05])
    out = run_backtest(panel, get_config())
    np.testing.assert_allclose(out[POSITION], 0.0)
    # No position -> no cost, no return.
    np.testing.assert_allclose(out[COST], 0.0)
    np.testing.assert_allclose(out[RET], 0.0)


def test_cap_is_respected():
    cfg = _cfg(cap=0.05)
    panel = _panel([1.0, 0.99, 0.9], [0.0, 0.0, 0.0])
    out = run_backtest(panel, cfg)
    assert (out[POSITION] <= 0.05).all()
    assert out[POSITION].max() == pytest.approx(0.05)


def test_turnover_cost_charged_on_entry_and_exit():
    # enter (0 -> 0.1), hold, exit (0.1 -> 0), re-enter (0 -> 0.1)
    panel = _panel([0.9, 0.9, 0.4, 0.9], [0.0, 0.0, 0.0, 0.0])
    out = run_backtest(panel, get_config())
    np.testing.assert_allclose(out[COST], [0.0003, 0.0, 0.0003, 0.0003])


def test_larger_shrink_gamma_sizes_up():
    panel = _panel([1.0], [0.0])
    small = run_backtest(panel, _cfg(shrink_gamma=0.3))
    large = run_backtest(panel, _cfg(shrink_gamma=0.9))
    # p_tilde = 0.5 + gamma*0.5, position = 0.25*(2*p_tilde - 1) = 0.25*gamma.
    assert small[POSITION].iloc[0] == pytest.approx(0.25 * 0.3)
    assert large[POSITION].iloc[0] == pytest.approx(0.25 * 0.9)
    assert large[POSITION].iloc[0] > small[POSITION].iloc[0]


def test_missing_columns_raise():
    panel = _panel([0.9], [0.01]).drop(columns=[P_UP])
    with pytest.raises(KeyError, match="panel missing columns"):
        run_backtest(panel, get_config())


def test_output_sorted_by_date():
    panel = _panel([0.9, 0.8, 0.7], [0.01, 0.01, 0.01])
    shuffled = panel.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = run_backtest(shuffled, get_config())
    assert out[DATE].is_monotonic_increasing

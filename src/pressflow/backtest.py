"""T5.1 — backtest engine (ADR-0009, ADR-0010, ADR-0012).

Turns a calibrated ``p_up`` (P(next-day up)) and the realised next-day
``ret_simple`` into a long-only, costed equity curve:

* shrink the probability: ``p̃ = 0.5 + γ·(p_up − 0.5)``;
* long-only sizing: ``p̃ ≤ 0.5`` → cash; else
  ``position = min(cap, k·(2·p̃ − 1))``;
* costs on turnover: ``cost_t = base · |position_t − position_{t−1}|``
  (``position_{t−1}`` starts at 0);
* daily net return ``ret_t = position_t · ret_simple_t − cost_t``, compounded
  from an initial equity of 1.0.

``run_backtest`` is a pure function of ``(panel, cfg)`` so it is trivially
unit-testable. The panel must carry ``date``, ``ret_simple`` and ``p_up``
(T4.4 supplies the calibrated ``p_up`` later).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pressflow.config import Config
from pressflow.join_prices import RET_SIMPLE
from pressflow.schema import (
    BACKTEST_COLUMNS,
    COST,
    DATE,
    EQUITY,
    POSITION,
    RET,
)

#: Column carrying the calibrated P(next-day up); supplied by T4.4.
P_UP = "p_up"

_REQUIRED_PANEL_COLUMNS = (DATE, RET_SIMPLE, P_UP)


def run_backtest(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run the long-only strategy on ``panel`` and return the backtest frame.

    Returns a frame with exactly :data:`pressflow.schema.BACKTEST_COLUMNS`,
    sorted by date, with equity compounded from 1.0.
    """
    missing = [c for c in _REQUIRED_PANEL_COLUMNS if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing columns: {missing}")

    frame = panel[[DATE, RET_SIMPLE, P_UP]].sort_values(DATE).reset_index(drop=True)

    gamma = cfg.kelly.shrink_gamma
    p_tilde = 0.5 + gamma * (frame[P_UP].to_numpy(dtype="float64") - 0.5)

    raw_position = cfg.kelly.k * (2.0 * p_tilde - 1.0)
    position = np.where(p_tilde <= 0.5, 0.0, np.minimum(cfg.kelly.cap, raw_position))
    position = np.clip(position, 0.0, cfg.kelly.cap)

    previous = np.empty_like(position)
    previous[0] = 0.0
    previous[1:] = position[:-1]

    cost = cfg.costs.base * np.abs(position - previous)
    ret = position * frame[RET_SIMPLE].to_numpy(dtype="float64") - cost
    equity = np.cumprod(1.0 + ret)

    return pd.DataFrame(
        {
            DATE: frame[DATE].to_numpy(),
            POSITION: position,
            RET: ret,
            COST: cost,
            EQUITY: equity,
        }
    )[list(BACKTEST_COLUMNS)]


__all__ = ["P_UP", "run_backtest"]

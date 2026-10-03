"""T5.3 — performance metrics and report (ADR-0011).

Standard performance table for the long-only strategy vs the SPY / OTP
buy-and-hold benchmarks:

* total return, CAGR ``(equity_end/equity_start)**(252/n_days) - 1``;
* annualized Sharpe (rf = 0) = ``mean(daily ret)/std(daily ret)·sqrt(252)``
  (population std; ``0.0`` for a zero-variance series);
* max drawdown (negative) = ``min(equity/cummax − 1)``;
* hit rate = fraction of long days (``position > 0``) with ``ret_simple > 0``;
* turnover = ``mean(|Δposition|)`` (first day measured from cash);
* trades = count of position transitions across zero.

:func:`render_report` writes ``docs/reports/backtest.md`` plus matplotlib plots;
the plots are intentionally not exercised by unit tests (matplotlib is an
optional extra).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressflow.config import REPO_ROOT
from pressflow.schema import DATE, EQUITY, OTP_USD, POSITION, RET

logger = logging.getLogger(__name__)

#: Trading days per year used to annualize CAGR and Sharpe.
TRADING_DAYS = 252

DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "reports" / "backtest.md"
DEFAULT_PLOTS_DIRNAME = "plots"


# --- curve helpers ---------------------------------------------------------


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough drawdown of ``equity`` (<= 0)."""
    values = np.asarray(equity, dtype="float64")
    peak = np.maximum.accumulate(values)
    return float(np.min(values / peak - 1.0))


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualized Sharpe with rf = 0 (population std); 0.0 if variance is 0."""
    values = np.asarray(returns, dtype="float64")
    values = values[~np.isnan(values)]
    if values.size < 2:
        return 0.0
    std = values.std(ddof=0)
    if std == 0.0:
        return 0.0
    return float(values.mean() / std * np.sqrt(TRADING_DAYS))


def curve_metrics(equity: pd.Series, returns: pd.Series | None = None) -> dict[str, float]:
    """Total return, CAGR, Sharpe and max drawdown of an equity curve."""
    values = np.asarray(equity, dtype="float64")
    if returns is None:
        returns = pd.Series(values).pct_change().dropna()
    start = float(values[0])
    end = float(values[-1])
    n_days = len(values)

    total_return = end / start - 1.0 if start != 0.0 else 0.0
    if start > 0.0 and end > 0.0 and n_days > 0:
        cagr = (end / start) ** (TRADING_DAYS / n_days) - 1.0
    else:  # pragma: no cover - defensive, equity is positive in practice
        cagr = 0.0

    return {
        "n_days": int(n_days),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
    }


# --- strategy-specific metrics --------------------------------------------


def align_ret_simple(backtest_df: pd.DataFrame, prices: pd.DataFrame) -> np.ndarray:
    """Realised ``ret_simple`` from ``prices`` aligned to the backtest dates."""
    ordered = prices[[DATE, OTP_USD]].sort_values(DATE)
    index = pd.DatetimeIndex(ordered[DATE])
    series = pd.Series(
        ordered[OTP_USD].pct_change().to_numpy(dtype="float64"), index=index
    )
    return series.reindex(pd.DatetimeIndex(backtest_df[DATE])).to_numpy(dtype="float64")


def hit_rate(backtest_df: pd.DataFrame, prices: pd.DataFrame) -> float:
    """Fraction of long days (position > 0) whose realised return is positive."""
    position = backtest_df[POSITION].to_numpy(dtype="float64")
    ret_simple = align_ret_simple(backtest_df, prices)
    is_long = position > 0.0
    if not is_long.any():
        return 0.0
    wins = np.nansum((ret_simple > 0.0) & is_long)
    return float(wins / int(is_long.sum()))


def turnover(backtest_df: pd.DataFrame) -> float:
    """Mean absolute daily position change (first day measured from cash)."""
    position = backtest_df[POSITION].to_numpy(dtype="float64")
    previous = np.concatenate([[0.0], position[:-1]])
    return float(np.mean(np.abs(position - previous)))


def trade_count(backtest_df: pd.DataFrame) -> int:
    """Number of position transitions across zero (entries + exits)."""
    position = backtest_df[POSITION].to_numpy(dtype="float64")
    is_long = position > 0.0
    previous = np.concatenate([[False], is_long[:-1]])
    return int(np.sum(is_long != previous))


def compute_metrics(
    backtest_df: pd.DataFrame,
    benchmarks: dict[str, pd.Series],
    prices: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Full performance table for the strategy and every benchmark curve."""
    for column in (DATE, POSITION, RET, EQUITY):
        if column not in backtest_df.columns:
            raise KeyError(f"backtest_df missing columns: {[column]}")

    strategy = curve_metrics(backtest_df[EQUITY], backtest_df[RET])
    strategy.update(
        hit_rate=hit_rate(backtest_df, prices),
        turnover=turnover(backtest_df),
        trades=trade_count(backtest_df),
    )

    metrics: dict[str, dict[str, Any]] = {"strategy": strategy}
    for name, curve in benchmarks.items():
        metrics[name] = curve_metrics(curve, curve.pct_change().dropna())
    return metrics


# --- report ----------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metrics_markdown(metrics: dict[str, dict[str, Any]]) -> str:
    rows = [
        ("Total return", "total_return", _pct),
        ("CAGR", "cagr", _pct),
        ("Sharpe", "sharpe", _fmt),
        ("Max drawdown", "max_drawdown", _pct),
        ("Hit rate", "hit_rate", _pct),
        ("Turnover", "turnover", _fmt),
        ("Trades", "trades", _fmt),
    ]
    names = list(metrics)
    header = "| Metric | " + " | ".join(n.capitalize() for n in names) + " |"
    separator = "|" + "---|" * (len(names) + 1)
    lines = [header, separator]
    for label, key, formatter in rows:
        cells = []
        for name in names:
            value = metrics[name].get(key)
            cells.append("—" if value is None else formatter(value))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(
    backtest_df: pd.DataFrame,
    benchmarks: dict[str, pd.Series],
    prices: pd.DataFrame,
    path: Path | None = None,
    plots_dir: Path | None = None,
    summary: str | None = None,
) -> Path:
    """Write ``docs/reports/backtest.md`` with the metrics table and plots.

    ``summary`` is an optional plain-language statement of the result (wired by
    T5.3b); when given it is rendered as a ``## Result`` section above the
    metrics table so the report states the outcome without editorializing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = path or DEFAULT_REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    plots = plots_dir or output.parent / DEFAULT_PLOTS_DIRNAME
    plots.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(backtest_df, benchmarks, prices)
    dates = pd.DatetimeIndex(backtest_df[DATE])
    equity = pd.Series(backtest_df[EQUITY].to_numpy(), index=dates)

    # 1. Equity curves (log scale).
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.to_numpy(), label="strategy")
    for name, curve in benchmarks.items():
        ax.plot(curve.index, curve.to_numpy(), label=name, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("Equity curves (log scale)")
    ax.set_ylabel("Equity (indexed to 1.0)")
    ax.legend()
    fig.tight_layout()
    equity_png = plots / "equity.png"
    fig.savefig(equity_png)
    plt.close(fig)

    # 2. Strategy drawdown.
    values = equity.to_numpy(dtype="float64")
    drawdown = values / np.maximum.accumulate(values) - 1.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dates, drawdown, 0.0, color="tab:red", alpha=0.4)
    ax.set_title("Strategy drawdown")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    drawdown_png = plots / "drawdown.png"
    fig.savefig(drawdown_png)
    plt.close(fig)

    # 3. Position over time.
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(dates, backtest_df[POSITION].to_numpy(), color="tab:blue")
    ax.set_title("Position over time")
    ax.set_ylabel("Fraction long")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    position_png = plots / "position.png"
    fig.savefig(position_png)
    plt.close(fig)

    intro = [
        "# Backtest report",
        "",
        "Metrics for the long-only, costed strategy vs the SPY / OTP",
        "buy-and-hold benchmarks (ADR-0011).",
        "",
    ]
    if summary:
        intro += ["## Result", "", summary, ""]
    body = "\n".join(
        [
            *intro,
            _metrics_markdown(metrics),
            "",
            "## Plots",
            "",
            f"![Equity curves]({equity_png.relative_to(output.parent).as_posix()})",
            "",
            f"![Drawdown]({drawdown_png.relative_to(output.parent).as_posix()})",
            "",
            f"![Position]({position_png.relative_to(output.parent).as_posix()})",
            "",
        ]
    )
    output.write_text(body, encoding="utf-8")
    print(f"wrote {output}")
    return output


__all__ = [
    "TRADING_DAYS",
    "DEFAULT_REPORT_PATH",
    "max_drawdown",
    "sharpe_ratio",
    "curve_metrics",
    "align_ret_simple",
    "hit_rate",
    "turnover",
    "trade_count",
    "compute_metrics",
    "render_report",
]

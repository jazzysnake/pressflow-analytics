"""T5.3b — run the backtest end-to-end (ADR-0011).

Wires the pure stages built earlier into one runnable command:

1. load the T4.4 calibrated panel (``data/processed/panel_calibrated.parquet``,
   carrying ``date`` / ``ret_simple`` / ``p_up``) and the T2.2 prices
   (``data/processed/prices.parquet``);
2. :func:`pressflow.backtest.run_backtest` → the costed strategy frame (exactly
   :data:`pressflow.schema.BACKTEST_COLUMNS`);
3. :func:`pressflow.benchmarks.benchmark_curves` → SPY / OTP / cash on the
   strategy calendar;
4. :func:`pressflow.metrics.compute_metrics` → the strategy-vs-benchmark table;
5. :func:`pressflow.metrics.render_report` → ``docs/reports/backtest.md`` and the
   ``equity.png`` / ``drawdown.png`` / ``position.png`` plots.

The strategy frame is also written to ``data/processed/backtest.parquet`` (the
data directory is gitignored). The report carries a plain-language summary of
the result — if the strategy underperforms the benchmarks, it says so plainly
and without editorializing.

CLI::

    uv run python -m pressflow.backtest_run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pressflow.backtest import P_UP, run_backtest
from pressflow.benchmarks import CASH, OTP, SPY, benchmark_curves
from pressflow.calibrate import calibrated_panel_path
from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.join_prices import RET_SIMPLE
from pressflow.metrics import compute_metrics, render_report
from pressflow.schema import BACKTEST_SCHEMA, DATE

logger = logging.getLogger("pressflow.backtest_run")

PRICES_FILENAME = "prices.parquet"
BACKTEST_FILENAME = "backtest.parquet"

__all__ = [
    "PRICES_FILENAME",
    "BACKTEST_FILENAME",
    "prices_path",
    "backtest_path",
    "save_backtest",
    "result_summary",
    "run",
    "main",
]


# --- paths / artifact ------------------------------------------------------


def prices_path(config: Config | None = None) -> Path:
    """Absolute path of the processed prices artifact (T2.2)."""
    config = config or get_config()
    return REPO_ROOT / config.paths.processed / PRICES_FILENAME


def backtest_path(config: Config | None = None) -> Path:
    """Absolute path of the processed backtest artifact."""
    config = config or get_config()
    return REPO_ROOT / config.paths.processed / BACKTEST_FILENAME


def save_backtest(backtest_df: pd.DataFrame, path: Path) -> None:
    """Write the backtest frame with the canonical ``BACKTEST_SCHEMA``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        backtest_df, schema=BACKTEST_SCHEMA, preserve_index=False
    )
    pq.write_table(table, path)


# --- plain-language result -------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def result_summary(metrics: dict[str, dict[str, Any]]) -> str:
    """One plain paragraph stating the strategy result vs the benchmarks.

    Derived only from the metrics table (no new math); it reports
    under-/out-performance against SPY (primary, ADR-0011) and OTP without
    editorializing.
    """
    strategy = metrics["strategy"]
    sentences = [
        f"The strategy returned {_pct(strategy['total_return'])} "
        f"(CAGR {_pct(strategy['cagr'])}, Sharpe {strategy['sharpe']:.2f}, "
        f"max drawdown {_pct(strategy['max_drawdown'])}, "
        f"{int(strategy['trades'])} trades)."
    ]

    for name, label in ((SPY, "SPY"), (OTP, "OTP")):
        bench = metrics.get(name)
        if bench is None:
            continue
        delta = strategy["total_return"] - bench["total_return"]
        if delta > 0:
            verb = f"outperformed {label}"
        elif delta < 0:
            verb = f"underperformed {label}"
        else:
            verb = f"matched {label}"
        sentences.append(
            f"It {verb} ({_pct(bench['total_return'])} total return) by "
            f"{abs(delta) * 100:.2f} percentage points."
        )

    cash = metrics.get(CASH)
    if cash is not None:
        delta = strategy["total_return"] - cash["total_return"]
        relation = "above" if delta >= 0 else "below"
        sentences.append(
            f"Cash returned {_pct(cash['total_return'])} over the same window, "
            f"so the strategy finished {relation} the risk-free alternative."
        )

    return " ".join(sentences)


# --- pipeline --------------------------------------------------------------


def run(
    config: Config | None = None,
    *,
    panel_path: Path | None = None,
    prices_path_override: Path | None = None,
    report_path: Path | None = None,
    plots_dir: Path | None = None,
    save_backtest_file: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run the backtest end-to-end and write the parquet artifact + report.

    Returns the strategy frame and the metrics table. Paths default to the
    configured ``data/processed`` locations and ``docs/reports/backtest.md``.
    """
    config = config or get_config()
    panel_file = panel_path or calibrated_panel_path(config)
    prices_file = prices_path_override or prices_path(config)
    if not panel_file.is_file():
        raise FileNotFoundError(
            f"calibrated panel not found: {panel_file} (run T4.4 first)"
        )
    if not prices_file.is_file():
        raise FileNotFoundError(f"prices not found: {prices_file} (run T2.2 first)")

    panel = pd.read_parquet(panel_file)
    missing = [c for c in (DATE, RET_SIMPLE, P_UP) if c not in panel.columns]
    if missing:
        raise KeyError(f"calibrated panel missing columns: {missing}")
    prices = pd.read_parquet(prices_file)

    backtest_df = run_backtest(panel, config)
    benchmarks = benchmark_curves(prices, backtest_df[DATE])
    metrics = compute_metrics(backtest_df, benchmarks, prices)

    if save_backtest_file:
        out = backtest_path(config)
        save_backtest(backtest_df, out)
        print(f"wrote {out} ({len(backtest_df)} rows)")

    render_report(
        backtest_df,
        benchmarks,
        prices,
        path=report_path,
        plots_dir=plots_dir,
        summary=result_summary(metrics),
    )
    return backtest_df, metrics


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the backtest end-to-end and write the T5.3 report."
    )
    parser.add_argument("--panel", type=Path, default=None, help="calibrated panel")
    parser.add_argument("--prices", type=Path, default=None, help="processed prices")
    parser.add_argument("--report", type=Path, default=None, help="report markdown")
    parser.add_argument("--plots-dir", type=Path, default=None, help="plot directory")
    parser.add_argument(
        "--no-save-backtest", action="store_true", help="skip the parquet artifact"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    backtest_df, metrics = run(
        panel_path=args.panel,
        prices_path_override=args.prices,
        report_path=args.report,
        plots_dir=args.plots_dir,
        save_backtest_file=not args.no_save_backtest,
    )
    strategy = metrics["strategy"]
    print(
        f"backtest {len(backtest_df)} rows; "
        f"total return={strategy['total_return'] * 100:+.2f}%, "
        f"Sharpe={strategy['sharpe']:.2f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

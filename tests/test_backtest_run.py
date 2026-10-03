"""Tests for T5.3b — run the backtest end-to-end."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pressflow.backtest import run_backtest
from pressflow.backtest_run import (
    BACKTEST_FILENAME,
    backtest_path,
    main,
    prices_path,
    result_summary,
    run,
    save_backtest,
)
from pressflow.calibrate import CALIBRATED_PANEL_FILENAME
from pressflow.config import Config, get_config
from pressflow.join_prices import RET_SIMPLE
from pressflow.metrics import compute_metrics
from pressflow.schema import (
    BACKTEST_COLUMNS,
    BUDAPEST_TZ,
    DATE,
    OTP_USD,
    SPY_USD,
)

DAY0 = pd.Timestamp("2024-01-02", tz=BUDAPEST_TZ)


# --- fixtures --------------------------------------------------------------


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range(DAY0, periods=n, freq="D")


def _panel(n: int = 6) -> pd.DataFrame:
    """Calibrated panel: a tiny long signal on a losing series."""
    return pd.DataFrame(
        {
            DATE: _dates(n),
            RET_SIMPLE: [-0.01] * n,
            "p_up": [0.9] * n,
        }
    )


def _prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Processed prices with one extra trading day before the panel."""
    all_dates = pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1), *dates])
    n = len(all_dates)
    return pd.DataFrame(
        {
            DATE: all_dates,
            SPY_USD: np.linspace(100.0, 130.0, n),
            OTP_USD: np.linspace(50.0, 70.0, n),
        }
    )


def _tmp_config(tmp_path: Path) -> Config:
    base = get_config()
    return replace(base, paths=replace(base.paths, processed=str(tmp_path)))


def _write_inputs(tmp_path: Path) -> tuple[Config, pd.DataFrame]:
    config = _tmp_config(tmp_path)
    panel = _panel()
    prices = _prices(pd.DatetimeIndex(panel[DATE]))
    panel.to_parquet(tmp_path / CALIBRATED_PANEL_FILENAME, index=False)
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    return config, panel


# --- paths -----------------------------------------------------------------


def test_artifact_paths_use_configured_processed_dir(tmp_path: Path):
    config = _tmp_config(tmp_path)
    assert prices_path(config) == tmp_path / "prices.parquet"
    assert backtest_path(config) == tmp_path / BACKTEST_FILENAME


# --- result summary --------------------------------------------------------


def _metrics(strategy_return: float, spy_return: float, otp_return: float):
    def curve(total_return: float) -> dict[str, float]:
        return {
            "total_return": total_return,
            "cagr": total_return,
            "sharpe": 0.1,
            "max_drawdown": -0.05,
            "n_days": 10,
        }

    strategy = curve(strategy_return)
    strategy.update(hit_rate=0.5, turnover=0.01, trades=7)
    return {"strategy": strategy, "spy": curve(spy_return), "otp": curve(otp_return), "cash": curve(0.0)}


def test_result_summary_states_underperformance_plainly():
    summary = result_summary(_metrics(strategy_return=-0.02, spy_return=0.36, otp_return=0.73))
    assert "underperformed SPY" in summary
    assert "underperformed OTP" in summary
    assert "below the risk-free alternative" in summary


def test_result_summary_states_outperformance_plainly():
    summary = result_summary(_metrics(strategy_return=0.50, spy_return=0.36, otp_return=0.20))
    assert "outperformed SPY" in summary
    assert "outperformed OTP" in summary
    assert "above the risk-free alternative" in summary


def test_result_summary_handles_matching_return():
    summary = result_summary(_metrics(strategy_return=0.36, spy_return=0.36, otp_return=0.10))
    assert "matched SPY" in summary


# --- save ------------------------------------------------------------------


def test_save_backtest_writes_canonical_schema(tmp_path: Path):
    bt = run_backtest(_panel(), get_config())
    out = tmp_path / BACKTEST_FILENAME
    save_backtest(bt, out)

    assert out.is_file()
    loaded = pd.read_parquet(out)
    assert list(loaded.columns) == list(BACKTEST_COLUMNS)
    assert len(loaded) == len(bt)


# --- end-to-end wiring (rendering stubbed) ---------------------------------


def test_run_writes_backtest_and_returns_metrics(tmp_path: Path, monkeypatch):
    config, panel = _write_inputs(tmp_path)

    calls: dict[str, object] = {}

    def _fake_render(backtest_df, benchmarks, prices, path=None, plots_dir=None, summary=None):
        calls["benchmarks"] = benchmarks
        calls["summary"] = summary
        Path(path).write_text("stub", encoding="utf-8")
        return Path(path)

    monkeypatch.setattr("pressflow.backtest_run.render_report", _fake_render)

    report = tmp_path / "backtest.md"
    backtest_df, metrics = run(
        config,
        panel_path=tmp_path / CALIBRATED_PANEL_FILENAME,
        prices_path_override=tmp_path / "prices.parquet",
        report_path=report,
    )

    # 1. parquet artifact with the canonical columns.
    out = tmp_path / BACKTEST_FILENAME
    assert out.is_file()
    loaded = pd.read_parquet(out)
    assert list(loaded.columns) == list(BACKTEST_COLUMNS)
    assert len(loaded) == len(panel)

    # 2. run() matches calling the engine/benchmarks/metrics directly.
    prices = pd.read_parquet(tmp_path / "prices.parquet")
    expected = run_backtest(panel, config)
    pd.testing.assert_frame_equal(backtest_df, expected)
    assert set(calls["benchmarks"]) == {"spy", "otp", "cash"}
    expected_metrics = compute_metrics(expected, calls["benchmarks"], prices)
    assert metrics == expected_metrics

    # 3. the plain-language summary was handed to the report renderer.
    assert "underperformed SPY" in calls["summary"]
    assert report.is_file()


def test_run_can_skip_backtest_artifact(tmp_path: Path, monkeypatch):
    config, _ = _write_inputs(tmp_path)
    monkeypatch.setattr("pressflow.backtest_run.render_report", lambda *a, **k: None)
    run(
        config,
        panel_path=tmp_path / CALIBRATED_PANEL_FILENAME,
        prices_path_override=tmp_path / "prices.parquet",
        report_path=tmp_path / "backtest.md",
        save_backtest_file=False,
    )
    assert not (tmp_path / BACKTEST_FILENAME).exists()


def test_run_missing_panel_raises(tmp_path: Path):
    config, _ = _write_inputs(tmp_path)
    missing = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError, match="calibrated panel not found"):
        run(config, panel_path=missing, prices_path_override=tmp_path / "prices.parquet")


def test_run_missing_prices_raises(tmp_path: Path):
    config, _ = _write_inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="prices not found"):
        run(
            config,
            panel_path=tmp_path / CALIBRATED_PANEL_FILENAME,
            prices_path_override=tmp_path / "nope.parquet",
        )


def test_run_missing_panel_columns_raise(tmp_path: Path):
    config = _tmp_config(tmp_path)
    bad = pd.DataFrame({DATE: _dates(3), RET_SIMPLE: [0.0, 0.0, 0.0]})
    bad.to_parquet(tmp_path / CALIBRATED_PANEL_FILENAME, index=False)
    prices = _prices(_dates(3))
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    with pytest.raises(KeyError, match="calibrated panel missing columns"):
        run(
            config,
            panel_path=tmp_path / CALIBRATED_PANEL_FILENAME,
            prices_path_override=tmp_path / "prices.parquet",
        )


# --- report + plots (matplotlib is an optional extra) ----------------------


def test_report_contains_metrics_table_and_three_plots(tmp_path: Path):
    pytest.importorskip("matplotlib")
    config, _ = _write_inputs(tmp_path)
    report = tmp_path / "backtest.md"
    plots = tmp_path / "plots"

    run(
        config,
        panel_path=tmp_path / CALIBRATED_PANEL_FILENAME,
        prices_path_override=tmp_path / "prices.parquet",
        report_path=report,
        plots_dir=plots,
    )

    text = report.read_text(encoding="utf-8")
    assert "## Result" in text
    assert "underperformed SPY" in text
    assert "| Total return |" in text
    for name in ("equity", "drawdown", "position"):
        assert f"{name}.png" in text
        plot = plots / f"{name}.png"
        assert plot.is_file()
        assert plot.stat().st_size > 0


def test_main_cli_writes_report_without_data_dir(tmp_path: Path):
    pytest.importorskip("matplotlib")
    _, _ = _write_inputs(tmp_path)
    report = tmp_path / "backtest.md"
    rc = main(
        [
            "--panel",
            str(tmp_path / CALIBRATED_PANEL_FILENAME),
            "--prices",
            str(tmp_path / "prices.parquet"),
            "--report",
            str(report),
            "--plots-dir",
            str(tmp_path / "plots"),
            "--no-save-backtest",
        ]
    )
    assert rc == 0
    assert report.is_file()

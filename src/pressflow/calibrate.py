"""T4.4 — walk-forward calibration to P(next-day up) (ADR-0003, ADR-0009).

Maps the best signal from T4.3 (default ``score_wmean``) to a calibrated
probability that the *next* trading day is up, with a strict expanding-window
(walk-forward) discipline:

* for every day ``t`` a logistic regression is fit **only** on rows strictly
  before ``t`` (never on the full sample), then used to predict ``t``;
* the first prediction waits for :data:`pressflow.config.Calibration.min_warmup`
  usable (non-null feature) observations.

Two probabilities are produced per day:

* ``p_up`` — the **raw** calibrated probability, consumed by the backtest
  (T5.1).  The backtest applies its own Kelly shrink
  ``p̃ = 0.5 + γ·(p_up − 0.5)`` (ADR-0009), so T4.4 must *not* pre-shrink it
  (double-shrink would over-damp the signal).
* ``p_up_shrunk`` — the shrunk value ``0.5 + γ·(p̂ − 0.5)`` with ``γ =
  kelly.shrink_gamma`` from config. Stored for the calibration report only.

Outputs ``docs/reports/calibration.md`` (calibration curve, Brier score,
no-lookahead notes) and the intermediate ``data/processed/panel_calibrated.parquet``.

CLI::

    uv run python -m pressflow.calibrate
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.join_prices import RET_SIMPLE, UP, panel_path
from pressflow.schema import DATE

logger = logging.getLogger("pressflow.calibrate")

#: Raw calibrated P(next-day up) — the column the backtest consumes (T5.1).
P_UP = "p_up"

#: Shrunk probability ``0.5 + γ·(p_up − 0.5)`` — report/diagnostics only.
P_UP_SHRUNK = "p_up_shrunk"

#: True where a walk-forward model actually produced ``p_up`` (excludes warm-up).
P_UP_FITTED = "p_up_fitted"

#: Neutral probability used for rows too early to calibrate (→ cash).
NEUTRAL_P = 0.5

#: Number of bins in the reliability/calibration curve.
CALIBRATION_BINS = 10

#: Report + plot locations.
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "reports" / "calibration.md"
DEFAULT_PLOT_PATH = REPO_ROOT / "docs" / "reports" / "calibration_curve.png"

#: Intermediate calibrated panel (never committed — ``data/**`` is gitignored).
CALIBRATED_PANEL_FILENAME = "panel_calibrated.parquet"

#: A factory returning a fresh, unfitted scikit-learn estimator.
ModelFactory = Callable[[], Any]

__all__ = [
    "P_UP",
    "P_UP_SHRUNK",
    "P_UP_FITTED",
    "NEUTRAL_P",
    "CALIBRATION_BINS",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_PLOT_PATH",
    "CALIBRATED_PANEL_FILENAME",
    "walk_forward_p_up",
    "shrink",
    "brier_score",
    "calibration_curve",
    "calibrate_panel",
    "calibrated_panel_path",
    "save_calibrated_panel",
    "render_report",
    "write_report",
    "build_report",
    "main",
]


# --- model factory (lazy sklearn: matches the ``ml`` optional extra) -------


def default_model_factory() -> ModelFactory:
    """Return a factory of plain L2 logistic regressions.

    scikit-learn is an optional extra (``uv sync --extra ml``), so the import
    is deferred until calibration is actually run.
    """
    from sklearn.linear_model import LogisticRegression

    return lambda: LogisticRegression(max_iter=1000)


# --- core walk-forward calibration -----------------------------------------


def walk_forward_p_up(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    min_warmup: int,
    model_factory: ModelFactory | None = None,
) -> np.ndarray:
    """Expanding-window calibrated P(up) for each row.

    ``features`` is ``(n, k)`` and ``labels`` is ``(n,)`` (1 = next-day up).
    For row ``i`` the estimator is fit on every row ``j < i`` with a complete
    feature vector (and at least one example of each class), then used to
    predict row ``i``.  Rows before ``min_warmup`` usable observations — or
    rows with missing features — get ``NaN`` (callers map those to cash).

    Fitting strictly on the past is what makes the result leakage-free: no
    information from day ``t`` or later can influence the prediction for ``t``.
    """
    factory = model_factory or default_model_factory()
    X = np.asarray(features, dtype="float64")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(labels).astype(int)
    if len(X) != len(y):
        raise ValueError(f"features and labels length mismatch: {len(X)} vs {len(y)}")
    if min_warmup < 1:
        raise ValueError(f"min_warmup must be >= 1, got {min_warmup}")

    n = len(y)
    valid = ~np.isnan(X).any(axis=1)
    out = np.full(n, np.nan, dtype="float64")

    for i in range(n):
        if not valid[i]:
            continue
        past = np.flatnonzero(valid[:i])
        if len(past) < min_warmup:
            continue
        y_past = y[past]
        if np.unique(y_past).size < 2:
            # Degenerate window: fall back to the past up-rate (still causal).
            out[i] = float(y_past.mean())
            continue
        model = factory()
        model.fit(X[past], y_past)
        out[i] = float(model.predict_proba(X[i : i + 1])[0, 1])
    return out


def shrink(p_up: np.ndarray | float, gamma: float) -> np.ndarray:
    """Shrink probabilities toward 0.5: ``0.5 + γ·(p − 0.5)`` (ADR-0009)."""
    values = np.asarray(p_up, dtype="float64")
    return NEUTRAL_P + gamma * (values - NEUTRAL_P)


def brier_score(p_up: np.ndarray, up: np.ndarray) -> float:
    """Mean squared error of the probability forecast (lower is better)."""
    p = np.asarray(p_up, dtype="float64")
    y = np.asarray(up).astype("float64")
    mask = ~np.isnan(p)
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def calibration_curve(
    p_up: np.ndarray,
    up: np.ndarray,
    *,
    n_bins: int = CALIBRATION_BINS,
) -> tuple[float, list[tuple[float, float, int, float | None, float | None]]]:
    """Reliability curve of predicted vs realised up-rate.

    Returns ``(ece, rows)`` where each row is
    ``(lo, hi, n, mean_predicted, empirical_up_rate)``; empty bins carry
    ``None`` for the two means.  Bins are left-open/right-closed, except the
    first which is closed on both ends so predictions of exactly 0.0 count.
    """
    p = np.asarray(p_up, dtype="float64")
    y = np.asarray(up).astype("float64")
    mask = ~np.isnan(p)
    p, y = p[mask], y[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[tuple[float, float, int, float | None, float | None]] = []
    ece = 0.0
    n_total = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0.0:
            in_bin = (p >= lo) & (p <= hi)
        else:
            in_bin = (p > lo) & (p <= hi)
        n = int(in_bin.sum())
        if n == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        mean_pred = float(p[in_bin].mean())
        up_rate = float(y[in_bin].mean())
        ece += (n / n_total) * abs(up_rate - mean_pred) if n_total else 0.0
        rows.append((lo, hi, n, mean_pred, up_rate))
    return ece, rows


# --- panel calibration -----------------------------------------------------


def calibrate_panel(
    panel: pd.DataFrame,
    config: Config | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return ``panel`` sorted by date with ``p_up`` and ``p_up_shrunk`` added.

    ``p_up`` is the **raw** calibrated probability (the backtest shrinks it
    itself). Rows too early to calibrate — or with missing features — get the
    neutral 0.5 (→ cash), so the frame stays aligned with the price panel.
    """
    config = config or get_config()
    features = tuple(feature_cols or config.calibration.features)
    missing = [c for c in (*features, RET_SIMPLE, UP) if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing columns: {missing}")

    ordered = panel.sort_values(DATE).reset_index(drop=True)
    raw = walk_forward_p_up(
        ordered[list(features)].to_numpy(dtype="float64"),
        ordered[UP].to_numpy(),
        min_warmup=config.calibration.min_warmup,
    )
    filled = np.where(np.isnan(raw), NEUTRAL_P, raw)

    out = ordered.copy()
    out[P_UP] = filled
    out[P_UP_SHRUNK] = shrink(filled, config.kelly.shrink_gamma)
    out[P_UP_FITTED] = ~np.isnan(raw)
    return out


def calibrated_panel_path(config: Config | None = None) -> Path:
    """Absolute path of the intermediate calibrated panel."""
    config = config or get_config()
    return REPO_ROOT / config.paths.processed / CALIBRATED_PANEL_FILENAME


def save_calibrated_panel(panel: pd.DataFrame, path: Path) -> None:
    """Write the calibrated panel to Parquet (intermediate artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, preserve_index=False)
    pq.write_table(table, path)


# --- report ----------------------------------------------------------------


def _curve_table(
    rows: Sequence[tuple[float, float, int, float | None, float | None]]
) -> str:
    lines = ["| bin | n | mean p̂ | realised up-rate |", "|---|---|---|---|"]
    for lo, hi, n, mean_pred, up_rate in rows:
        if n == 0:
            lines.append(f"| {lo:.1f}–{hi:.1f} | 0 | – | – |")
        else:
            lines.append(
                f"| {lo:.1f}–{hi:.1f} | {n} | {mean_pred:.3f} | {up_rate:.3f} |"
            )
    return "\n".join(lines)


def _plot_curve(
    rows: Sequence[tuple[float, float, int, float | None, float | None]],
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(m, u) for _, _, n, m, u in rows if n > 0 and m is not None]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", label="perfect calibration")
    if pts:
        xs = [m for m, _ in pts]
        ys = [u for _, u in pts]
        ax.plot(xs, ys, "o-", color="tab:blue", label="calibrated P(up)")
    ax.set_xlabel("mean predicted P(up)")
    ax.set_ylabel("realised up-rate")
    ax.set_title("Calibration curve (walk-forward)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def render_report(
    calibrated: pd.DataFrame,
    config: Config,
    *,
    plot_path: Path = DEFAULT_PLOT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> str:
    """Render the ``docs/reports/calibration.md`` body."""
    features = config.calibration.features
    gamma = config.kelly.shrink_gamma
    scored = calibrated[calibrated[P_UP_FITTED]]

    p_raw = scored[P_UP].to_numpy(dtype="float64")
    p_shrunk = scored[P_UP_SHRUNK].to_numpy(dtype="float64")
    y = scored[UP].to_numpy()
    base_rate = float(np.mean(y)) if len(y) else float("nan")

    brier_raw = brier_score(p_raw, y)
    brier_shrunk = brier_score(p_shrunk, y)
    brier_base = brier_score(np.full_like(y, base_rate, dtype="float64"), y)
    brier_half = brier_score(np.full_like(p_raw, NEUTRAL_P), y)

    ece, rows = calibration_curve(p_raw, y)

    leak = (pd.to_datetime(calibrated[DATE]) <= pd.to_datetime(calibrated["signal_date"])).sum()

    lines: list[str] = []
    lines.append("# T4.4 — Calibration to P(next-day up)\n")
    lines.append(
        "Generated by `pressflow.calibrate` (T4.4) from "
        f"`data/processed/{panel_path(config).name}` — {len(calibrated)} panel rows, "
        f"{len(scored)} with a walk-forward calibrated probability "
        "(ADR-0003, ADR-0009).\n"
    )

    lines.append("## Method\n")
    lines.append(
        "- **Model:** L2 logistic regression (scikit-learn), "
        f"feature(s) `{', '.join(features)}` (best signal from T4.3).\n"
        "- **Scheme:** walk-forward expanding window — for each day `t` the "
        "model is fit on days strictly before `t` and predicts `t` "
        "(no lookahead, ADR-0010).\n"
        f"- **Warm-up:** the first `min_warmup = {config.calibration.min_warmup}` "
        "usable observations are not calibrated (they map to cash).\n"
        f"- **Shrink:** `p̃ = 0.5 + {gamma}·(p̂ − 0.5)` (ADR-0009); the raw `p_up` "
        "is what the T5.1 backtest consumes — it applies its own shrink, so "
        "pre-shrinking here would double-count (stored separately as "
        f"`{P_UP_SHRUNK}`).\n"
    )

    lines.append("## Scores\n")
    lines.append(
        "| forecast | Brier (raw sample) | note |\n|---|---|---|\n"
        f"| calibrated `p_up` | **{brier_raw:.4f}** | walk-forward logistic |\n"
        f"| shrunk `p̃` | {brier_shrunk:.4f} | γ = {gamma} |\n"
        f"| constant base rate ({base_rate:.3f}) | {brier_base:.4f} | climatology |\n"
        f"| constant 0.5 | {brier_half:.4f} | naive |\n"
    )
    lines.append(
        "\nLower Brier is better. Brier scores are evaluated on the calibrated "
        "sample only (the warm-up rows carry the neutral 0.5 and are excluded "
        "from the curve and scores).\n"
    )

    lines.append("## Calibration curve\n")
    lines.append(
        f"Binned predicted vs realised up-rate over {len(scored)} days "
        f"(ECE = {ece:.4f}):\n"
    )
    lines.append(_curve_table(rows))
    lines.append("")
    try:
        plot_ref = plot_path.relative_to(report_path.parent).as_posix()
    except ValueError:  # pragma: no cover - unusual custom paths
        plot_ref = plot_path.name
    lines.append(f"![Calibration curve]({plot_ref})\n")

    lines.append("## No-lookahead\n")
    lines.append(
        f"- Rows with `signal_date >= date`: **{int(leak)}** (must be 0) — the "
        "signal is day `t−1`, the label is day `t`'s return (T4.2).\n"
        "- The expanding-window fit uses only rows `< t`, so no day `t` or "
        "later information enters the prediction for `t`; enforced by "
        "`tests/test_calibration.py::test_no_lookahead_ignores_future_rows`.\n"
    )

    lines.append("## Artifacts\n")
    lines.append(
        f"- `data/processed/{CALIBRATED_PANEL_FILENAME}` — per-day `{P_UP}` "
        f"(raw, feeds the backtest) and `{P_UP_SHRUNK}` (report only).\n"
    )
    return "\n".join(lines)


def write_report(text: str, path: Path = DEFAULT_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_report(
    config: Config | None = None,
    *,
    panel_path_override: Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    plot_path: Path = DEFAULT_PLOT_PATH,
    save_panel_file: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Calibrate the panel, write the artifact and the markdown report."""
    config = config or get_config()
    source = panel_path_override or panel_path(config)
    if not source.is_file():
        raise FileNotFoundError(f"panel not found: {source} (run T4.2 first)")

    panel = pd.read_parquet(source)
    calibrated = calibrate_panel(panel, config)
    if save_panel_file:
        out = calibrated_panel_path(config)
        save_calibrated_panel(calibrated, out)
        logger.info("wrote %s", out)

    scored = calibrated[calibrated[P_UP_FITTED]]
    _, rows = calibration_curve(
        scored[P_UP].to_numpy(dtype="float64"), scored[UP].to_numpy()
    )
    _plot_curve(rows, plot_path)
    text = render_report(
        calibrated, config, plot_path=plot_path, report_path=report_path
    )
    write_report(text, report_path)
    logger.info("wrote %s", report_path)
    return calibrated, text


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward calibrate the panel to P(next-day up) (T4.4)."
    )
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument(
        "--no-save-panel", action="store_true", help="skip the parquet artifact"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    calibrated, _ = build_report(
        panel_path_override=args.panel,
        report_path=args.report,
        plot_path=args.plot,
        save_panel_file=not args.no_save_panel,
    )
    scored = calibrated[calibrated[P_UP_FITTED]]
    print(
        f"calibrated {len(scored)} rows; Brier={brier_score(scored[P_UP], scored[UP]):.4f}"
    )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

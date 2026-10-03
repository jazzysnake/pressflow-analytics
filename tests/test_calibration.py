"""Tests for T4.4 — walk-forward calibration to P(next-day up)."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

# scikit-learn is an optional extra (``uv sync --extra ml``); the calibration
# model needs it, so skip the suite (rather than error) on a clean ``uv sync``.
pytest.importorskip("sklearn")

from pressflow.backtest import run_backtest
from pressflow.calibrate import (
    NEUTRAL_P,
    P_UP,
    P_UP_FITTED,
    P_UP_SHRUNK,
    brier_score,
    calibrate_panel,
    calibration_curve,
    render_report,
    shrink,
    walk_forward_p_up,
)
from pressflow.config import get_config
from pressflow.join_prices import UP
from pressflow.schema import BUDAPEST_TZ, DATE, POSITION

DAY0 = pd.Timestamp("2024-01-02", tz=BUDAPEST_TZ)


class _ConstModel:
    """Stub estimator returning a constant probability (hand-computable)."""

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def fit(self, X, y):  # noqa: N803 - sklearn signature
        self.n_fit = len(y)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn signature
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


def _synthetic_signals(n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    feature = np.linspace(-2.0, 2.0, n)
    p = 1.0 / (1.0 + np.exp(-1.5 * feature))
    up = (rng.random(n) < p).astype(int)
    return feature.reshape(-1, 1), up


# --- pure helpers ----------------------------------------------------------


def test_shrink_hand_computed():
    np.testing.assert_allclose(shrink([0.9, 0.5, 0.1], 0.5), [0.7, 0.5, 0.3])


def test_shrink_reuses_configured_gamma():
    # p_up = 1.0, gamma = 0.5 -> p_tilde = 0.75 (ADR-0009 default).
    cfg = get_config()
    assert shrink(np.array([1.0]), cfg.kelly.shrink_gamma)[0] == pytest.approx(0.75)


def test_brier_score_hand_computed():
    # ((0.8-1)^2 + (0.3-0)^2) / 2 = (0.04 + 0.09) / 2 = 0.065
    assert brier_score(np.array([0.8, 0.3]), np.array([1, 0])) == pytest.approx(0.065)


def test_calibration_curve_hand_computed():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    ece, rows = calibration_curve(p, y, n_bins=2)
    # bin 1 (0.0,0.5]: n=2, mean p=0.25, up-rate=0.0
    # bin 2 (0.5,1.0]: n=2, mean p=0.75, up-rate=1.0
    assert rows[0][2] == 2 and rows[0][3] == pytest.approx(0.25)
    assert rows[0][4] == pytest.approx(0.0)
    assert rows[1][2] == 2 and rows[1][3] == pytest.approx(0.75)
    assert rows[1][4] == pytest.approx(1.0)
    # ECE = 0.5*|0-0.25| + 0.5*|1-0.75| = 0.25
    assert ece == pytest.approx(0.25)


def test_calibration_curve_empty_bin_is_none():
    ece, rows = calibration_curve(
        np.array([0.1, 0.2]), np.array([1, 0]), n_bins=4
    )
    assert len(rows) == 4
    empty = [r for r in rows if r[2] == 0]
    assert empty and empty[0][3] is None and empty[0][4] is None
    # Both points fall in the first bin [0, 0.25]: mean p = 0.15, up-rate = 0.5.
    assert ece == pytest.approx(abs(0.5 - 0.15))


# --- walk-forward mechanics ------------------------------------------------


def test_hand_computed_tiny_walk_forward_with_stub():
    # 5 rows, min_warmup=2 -> first two are uncalibrated (NaN), rest = 0.7.
    X = np.array([[0.0], [1.0], [0.0], [1.0], [0.0]])
    y = np.array([0, 1, 0, 1, 1])
    out = walk_forward_p_up(
        X, y, min_warmup=2, model_factory=lambda: _ConstModel(0.7)
    )
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2:], [0.7, 0.7, 0.7])


def test_missing_feature_row_is_uncalibrated_and_skipped_in_fit():
    X = np.array([[0.0], [np.nan], [1.0], [0.0], [1.0]])
    y = np.array([0, 0, 1, 0, 1])
    out = walk_forward_p_up(
        X, y, min_warmup=2, model_factory=lambda: _ConstModel(0.6)
    )
    # row 0: no past -> NaN; row 1: feature missing -> NaN;
    # row 2: only row 0 usable before it -> only 1 < min_warmup -> NaN;
    # rows 3,4: >= 2 usable past rows -> 0.6.
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    np.testing.assert_allclose(out[3:], [0.6, 0.6])


def test_p_up_in_unit_interval():
    X, y = _synthetic_signals()
    out = walk_forward_p_up(X, y, min_warmup=50)
    scored = out[~np.isnan(out)]
    assert scored.size > 0
    assert (scored >= 0.0).all() and (scored <= 1.0).all()


def test_no_lookahead_ignores_future_rows():
    """Changing day t or later must not change any prediction up to and
    including t — the expanding window only ever looks backwards."""
    X, y = _synthetic_signals(n=300, seed=11)
    cutoff = 200
    before = walk_forward_p_up(X, y, min_warmup=40)

    X_future = X.copy()
    y_future = y.copy()
    # Drastically rewrite everything strictly after the cutoff.
    X_future[cutoff + 1 :] = -X[cutoff + 1 :]
    y_future[cutoff + 1 :] = 1 - y[cutoff + 1 :]
    after = walk_forward_p_up(X_future, y_future, min_warmup=40)

    np.testing.assert_allclose(
        before[: cutoff + 1], after[: cutoff + 1], equal_nan=True, rtol=1e-12
    )
    # Sanity: the future rewrite did move at least one later prediction.
    assert not np.allclose(before[cutoff + 1 :], after[cutoff + 1 :])


def test_no_lookahead_ignores_the_label_at_row_t():
    """Mutation guard for a ``valid[:i+1]`` (fits row ``t`` itself) bug.

    ``test_no_lookahead_ignores_future_rows`` only catches leakage of rows
    *after* ``t``; it would still pass if the model were fit on row ``t``
    itself. Flipping row ``t``'s own ``up`` label must therefore leave the
    prediction for row ``t`` bit-for-bit unchanged. With the strict
    ``valid[:i]`` window the fit set excludes ``t``, so it does; with
    ``valid[:i+1]`` the flipped label enters the fit and the prediction moves.
    """
    X, y = _synthetic_signals(n=300, seed=11)
    t = 200
    before = walk_forward_p_up(X, y, min_warmup=40)

    y_flipped = y.copy()
    y_flipped[t] = 1 - y[t]
    after = walk_forward_p_up(X, y_flipped, min_warmup=40)

    assert before[t] == after[t]
    assert not np.isnan(before[t])

    # Sanity: ``t+1`` is fit on row ``t``, so flipping that label *must* move
    # its prediction — otherwise the guard above would be vacuous.
    assert before[t + 1] != after[t + 1]


def test_monotonic_increasing_in_signal():
    X, y = _synthetic_signals(seed=3)
    out = walk_forward_p_up(X, y, min_warmup=50)
    mask = ~np.isnan(out)
    assert pd.Series(X[mask, 0]).corr(pd.Series(out[mask]), method="spearman") > 0.9
    assert out[mask][X[mask, 0] > 0.5].mean() > out[mask][X[mask, 0] < -0.5].mean()


# --- panel calibration -----------------------------------------------------


def _panel(n: int = 400, seed: int = 7) -> pd.DataFrame:
    feature, up = _synthetic_signals(n, seed)
    rng = np.random.default_rng(seed + 1)
    return pd.DataFrame(
        {
            DATE: [DAY0 + pd.Timedelta(days=i) for i in range(n)],
            "signal_date": [DAY0 + pd.Timedelta(days=i - 1) for i in range(n)],
            "score_wmean": feature[:, 0],
            "up": up.astype(bool),
            "ret_simple": rng.normal(0.0, 0.02, size=n),
        }
    )


def test_calibrate_panel_output_shape_and_shrink_relation():
    cfg = get_config()
    out = calibrate_panel(_panel(), cfg)
    assert {P_UP, P_UP_SHRUNK}.issubset(out.columns)
    assert (out[P_UP] >= 0.0).all() and (out[P_UP] <= 1.0).all()
    # Warm-up rows carry the neutral 0.5 (-> cash), so their shrunk value is 0.5.
    warmup = out[DATE] < DAY0 + pd.Timedelta(days=cfg.calibration.min_warmup - 1)
    assert (out.loc[warmup, P_UP] == NEUTRAL_P).all()
    expected = NEUTRAL_P + cfg.kelly.shrink_gamma * (out[P_UP] - NEUTRAL_P)
    np.testing.assert_allclose(out[P_UP_SHRUNK], expected)
    # p_up is increasing in the signal (monotonicity vs `score_wmean`).
    fitted = out[out[P_UP_FITTED]]
    assert fitted["score_wmean"].corr(fitted[P_UP], method="spearman") > 0.9


def test_calibrate_panel_is_date_sorted():
    shuffled = _panel().sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = calibrate_panel(shuffled, get_config())
    assert out[DATE].is_monotonic_increasing


def test_calibrate_panel_missing_feature_raises():
    with pytest.raises(KeyError, match="panel missing columns"):
        calibrate_panel(_panel().drop(columns=["score_wmean"]), get_config())


# --- integration with the backtest (double-shrink avoidance) ---------------


def test_backtest_consumes_raw_p_up_not_pre_shrunk():
    cfg = get_config()
    # p_up = 0.9 raw -> backtest p_tilde = 0.7 -> position = 0.25*(2*0.7-1) = 0.1.
    # If the pre-shrunk 0.7 were fed in, position would be 0.05.
    panel = pd.DataFrame(
        {
            DATE: [DAY0],
            "ret_simple": [0.0],
            P_UP: [0.9],
            P_UP_SHRUNK: [0.7],
        }
    )
    out = run_backtest(panel, cfg)
    assert out[POSITION].iloc[0] == pytest.approx(0.1)

    # calibrate_panel must emit the *raw* walk-forward probability as `p_up`:
    # pre-shrinking it there would double-shrink when the backtest shrinks again.
    calibrated = calibrate_panel(_panel(), cfg)
    raw = walk_forward_p_up(
        calibrated[list(cfg.calibration.features)].to_numpy(dtype="float64"),
        calibrated[UP].to_numpy(),
        min_warmup=cfg.calibration.min_warmup,
    )
    expected = np.where(np.isnan(raw), NEUTRAL_P, raw)
    np.testing.assert_allclose(calibrated[P_UP].to_numpy(), expected)
    # The stored shrink is a different column — `p_up` is not pre-shrunk.
    fitted = calibrated[calibrated[P_UP_FITTED]]
    assert not np.allclose(fitted[P_UP], fitted[P_UP_SHRUNK])


def test_report_mentions_key_artifacts_and_no_lookahead():
    cfg = get_config()
    calibrated = calibrate_panel(_panel(), cfg)
    text = render_report(calibrated, cfg)
    assert "Calibration curve" in text
    assert "Brier" in text
    assert "no lookahead" in text.lower()
    assert P_UP in text and P_UP_SHRUNK in text
    # A clean synthetic panel has no signal_date >= date violations.
    assert "**0**" in text


def test_report_states_brier_vs_climatology_honestly():
    cfg = get_config()
    calibrated = calibrate_panel(_panel(), cfg)
    text = render_report(calibrated, cfg)
    # The report must state the Brier-vs-climatology verdict plainly ...
    assert "climatology" in text
    assert (
        "beats the constant-base-rate climatology" in text
        or "is not better than the constant-base-rate climatology" in text
    )
    # ... and report how narrow the prediction span is.
    assert "spanning" in text

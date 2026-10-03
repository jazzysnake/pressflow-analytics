"""Tests for T4.1 — daily signal aggregation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pressflow.aggregate import (
    PROCESSED_FILENAME,
    aggregate_daily,
    save_signals,
    signals_path,
)
from pressflow.config import Aggregation, load_config, get_config, REPO_ROOT
from pressflow.schema import (
    BEAR_WMEAN,
    BUDAPEST_TZ,
    COUNT,
    DATE,
    DISAGREEMENT_STD,
    ENTROPY,
    HSZ_ID,
    NET,
    P_BEAR,
    P_BULL,
    P_NEUTRAL,
    SCORE_WMEAN,
    SIGNALS_COLUMNS,
    SIGNALS_SCHEMA,
    TS,
    UPVOTES,
    Z_SCORE,
)

DAY0 = pd.Timestamp("2024-01-02", tz=BUDAPEST_TZ)


def _cfg(**overrides):
    base = get_config()
    agg = replace(base.aggregation, **overrides)
    return replace(base, aggregation=agg)


def _post_row(hsz_id: int, day: pd.Timestamp, upvotes: int = 0) -> dict:
    return {HSZ_ID: hsz_id, TS: day + pd.Timedelta(hours=12), UPVOTES: upvotes}


def _score_row(
    hsz_id: int,
    p_bull: float,
    p_bear: float,
    p_neutral: float,
    net: float | None = None,
) -> dict:
    if net is None:
        net = p_bull - p_bear
    return {
        HSZ_ID: hsz_id,
        P_BULL: p_bull,
        P_BEAR: p_bear,
        P_NEUTRAL: p_neutral,
        NET: net,
    }


def _net_frames(nets: list[float], upvotes: list[int] | None = None):
    """One synthetic post per day; ``net`` fully determines the probabilities."""
    scores = []
    posts = []
    ups = upvotes or [0] * len(nets)
    for i, net in enumerate(nets):
        day = DAY0 + pd.Timedelta(days=i)
        if net >= 0:
            p_bull, p_bear, p_neutral = net, 0.0, 1.0 - net
        else:
            p_bull, p_bear, p_neutral = 0.0, -net, 1.0 + net
        scores.append(_score_row(i, p_bull, p_bear, p_neutral, net))
        posts.append(_post_row(i, day, ups[i]))
    return pd.DataFrame(scores), pd.DataFrame(posts)


# --- output contract -------------------------------------------------------


def test_output_columns_match_signals_schema():
    scores, posts = _net_frames([0.2, 0.4, 0.6, -0.3])
    out = aggregate_daily(scores, posts, _cfg(z_window=3, z_min_periods=2))
    assert list(out.columns) == list(SIGNALS_COLUMNS)


def test_date_is_tz_aware_budapest_midnight_and_count_int64():
    scores, posts = _net_frames([0.2, 0.4, 0.6, -0.3])
    out = aggregate_daily(scores, posts, _cfg(z_window=3, z_min_periods=2))
    dates = pd.DatetimeIndex(out[DATE])
    assert str(dates.tz) == BUDAPEST_TZ
    assert dates.dtype == "datetime64[us, Europe/Budapest]"
    assert (dates == dates.normalize()).all()
    assert out[COUNT].dtype == "int64"


# --- hand-computed aggregation --------------------------------------------


def _two_day_panel():
    """Day 1: pure bull post. Day 2: bull (w=3) + a (0.0, 0.5, 0.5) mix. Day 3: pure bear."""
    day1 = DAY0
    day2 = DAY0 + pd.Timedelta(days=1)
    day3 = DAY0 + pd.Timedelta(days=2)
    scores = pd.DataFrame(
        [
            _score_row(1, 1.0, 0.0, 0.0),  # day 1, net=1
            _score_row(2, 1.0, 0.0, 0.0),  # day 2, net=1, upvotes=2 -> w=3
            _score_row(3, 0.0, 0.5, 0.5),  # day 2, net=-0.5, w=1
            _score_row(4, 0.0, 1.0, 0.0),  # day 3, net=-1
        ]
    )
    posts = pd.DataFrame(
        [
            _post_row(1, day1),
            _post_row(2, day2, upvotes=2),
            _post_row(3, day2),
            _post_row(4, day3),
        ]
    )
    return scores, posts


def test_engagement_weighted_means_hand_computed():
    scores, posts = _two_day_panel()
    out = aggregate_daily(scores, posts, _cfg(z_window=2, z_min_periods=2))
    # day 1 is warm-up (only one observation); day 2 is the first retained row.
    row = out.iloc[0]
    assert row[DATE] == DAY0 + pd.Timedelta(days=1)
    # (3*1.0 + 1*(-0.5)) / (3 + 1) = 0.625
    assert row[SCORE_WMEAN] == pytest.approx(0.625)
    # (3*0.0 + 1*0.5) / 4 = 0.125
    assert row[BEAR_WMEAN] == pytest.approx(0.125)
    assert row[COUNT] == 2
    # population std of [1.0, -0.5] = 0.75
    assert row[DISAGREEMENT_STD] == pytest.approx(0.75)
    # mean of H(1,0,0) ~= 0 and H(0,0.5,0.5) = ln 2
    assert row[ENTROPY] == pytest.approx(np.log(2.0) / 2.0)


def test_pure_bull_entropy_is_zero():
    # Day 1 is a pure bull post; it is the first retained row (window=2,
    # min_periods=2) because day 0 differs from it.
    scores, posts = _net_frames([-0.5, 1.0, 0.3])
    out = aggregate_daily(scores, posts, _cfg(z_window=2, z_min_periods=2))
    assert out.iloc[0][DATE] == DAY0 + pd.Timedelta(days=1)
    assert out.iloc[0][ENTROPY] == pytest.approx(0.0, abs=1e-9)


def test_single_post_day_disagreement_is_zero():
    scores, posts = _net_frames([0.3, -0.6, 0.9])
    out = aggregate_daily(scores, posts, _cfg(z_window=2, z_min_periods=2))
    assert (out[DISAGREEMENT_STD] == 0.0).all()


def test_all_missing_probability_column_raises():
    scores, posts = _net_frames([0.2, 0.4])
    with pytest.raises(KeyError, match="scores missing columns"):
        aggregate_daily(scores.drop(columns=[NET]), posts, _cfg())


def test_unsupported_weight_scheme_raises():
    scores, posts = _net_frames([0.2, 0.4])
    cfg = replace(
        get_config(),
        aggregation=Aggregation(weight_scheme="uniform", z_window=2, z_min_periods=2),
    )
    with pytest.raises(ValueError, match="unsupported aggregation.weight_scheme"):
        aggregate_daily(scores, posts, cfg)


# --- z-score ---------------------------------------------------------------


def test_warmup_rows_are_dropped():
    scores, posts = _net_frames([0.01 * (i % 5 - 2) for i in range(35)])
    cfg = _cfg(z_window=60, z_min_periods=30)
    out = aggregate_daily(scores, posts, cfg)
    # First 29 rows have no z_score (min_periods=30) -> 35 - 29 = 6 rows remain.
    assert len(out) == 6
    assert out[DATE].iloc[0] == DAY0 + pd.Timedelta(days=29)
    assert out[Z_SCORE].notna().all()


def test_hand_computed_z_score():
    scores, posts = _net_frames([0.1, 0.2, 0.3, 0.4, 0.5])
    out = aggregate_daily(scores, posts, _cfg(z_window=3, z_min_periods=3))
    # First retained row is day index 2: window [0.1, 0.2, 0.3].
    expected = (0.3 - 0.2) / np.std([0.1, 0.2, 0.3], ddof=0)
    assert out.iloc[0][Z_SCORE] == pytest.approx(expected)


def test_z_score_uses_only_trailing_data():
    nets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7]
    scores, posts = _net_frames(nets)
    base = aggregate_daily(scores, posts, _cfg(z_window=3, z_min_periods=3))

    spiked = list(nets)
    spiked[-1] = 5.0  # change the *last* day only
    scores2, posts2 = _net_frames(spiked)
    changed = aggregate_daily(scores2, posts2, _cfg(z_window=3, z_min_periods=3))

    # A future spike must not retroactively change any earlier z_score.
    merged = base.merge(changed, on=DATE, suffixes=("_base", "_spiked"))
    overlap = merged[merged[DATE] < base[DATE].iloc[-1]]
    assert len(overlap) >= 3
    np.testing.assert_allclose(
        overlap[f"{Z_SCORE}_base"], overlap[f"{Z_SCORE}_spiked"]
    )
    # ... and the spiked day itself does change.
    assert changed[Z_SCORE].iloc[-1] != pytest.approx(base[Z_SCORE].iloc[-1])


def test_output_sorted_by_date_ascending():
    nets = [0.2, -0.4, 0.6, 0.1, -0.2, 0.3]
    # Shuffle input rows; output must still be date-sorted.
    scores, posts = _net_frames(nets)
    shuffled_scores = scores.sample(frac=1.0, random_state=0).reset_index(drop=True)
    shuffled_posts = posts.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = aggregate_daily(shuffled_scores, shuffled_posts, _cfg(z_window=3, z_min_periods=2))
    assert out[DATE].is_monotonic_increasing


# --- persistence -----------------------------------------------------------


def test_save_signals_writes_canonical_schema(tmp_path):
    scores, posts = _net_frames([0.2, 0.4, 0.6, -0.3])
    out = aggregate_daily(scores, posts, _cfg(z_window=3, z_min_periods=2))
    path = tmp_path / "signals.parquet"
    save_signals(out, path)
    assert pq.read_schema(path) == SIGNALS_SCHEMA
    reloaded = pd.read_parquet(path)
    assert list(reloaded.columns) == list(SIGNALS_COLUMNS)


def test_signals_path_uses_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        (REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert signals_path(cfg) == REPO_ROOT / cfg.paths.processed / PROCESSED_FILENAME

"""T4.1 — daily signal aggregation (ADR-0003, ADR-0007).

Collapses the per-post sentiment ``scores`` into one ``signals`` row per
Europe/Budapest trading day:

* ``score_wmean`` — engagement-weighted mean of ``net`` (weight ``1 + upvotes``);
* ``bear_wmean`` — engagement-weighted mean of ``p_bear``;
* ``count`` — number of scored posts that day;
* ``disagreement_std`` — *unweighted* population std (``ddof=0``) of ``net``,
  ``0.0`` for a single-post day;
* ``entropy`` — mean over posts of the Shannon entropy of
  ``(p_bull, p_bear, p_neutral)`` (natural log, probabilities floored at 1e-12);
* ``z_score`` — trailing rolling z-score of ``score_wmean`` (window/min periods
  from config). Warm-up rows without a ``z_score`` are dropped so the output
  stays non-null, matching the ``signals`` schema (ADR-0015).

Usage::

    uv run python -m pressflow.aggregate
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pressflow.config import REPO_ROOT, Config, get_config
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

logger = logging.getLogger(__name__)

PROCESSED_FILENAME = "signals.parquet"
SCORES_FILENAME = "scores.parquet"
POSTS_FILENAME = "posts.parquet"

#: Probability floor for the entropy logarithm (avoids ``log(0)``).
PROB_FLOOR = 1e-12

#: Columns the aggregation depends on.
REQUIRED_SCORE_COLUMNS: tuple[str, ...] = (HSZ_ID, P_BULL, P_BEAR, P_NEUTRAL, NET)
REQUIRED_POST_COLUMNS: tuple[str, ...] = (HSZ_ID, TS, UPVOTES)

#: Only supported engagement weighting for now (T4.1).
WEIGHT_SCHEME_ENGAGEMENT = "engagement"


def to_budapest_day(values: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalise timestamps to tz-aware Europe/Budapest midnight."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is None:
        idx = idx.tz_localize(BUDAPEST_TZ)
    else:
        idx = idx.tz_convert(BUDAPEST_TZ)
    return idx.normalize().as_unit("us")


def _shannon_entropy(frame: pd.DataFrame) -> pd.Series:
    """Per-post Shannon entropy of ``(p_bull, p_bear, p_neutral)`` (natural log)."""
    probs = frame[[P_BULL, P_BEAR, P_NEUTRAL]].clip(lower=PROB_FLOOR, upper=1.0)
    return -(probs * np.log(probs)).sum(axis=1)


def aggregate_daily(
    scores: pd.DataFrame,
    posts: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Aggregate per-post scores into one row per Budapest day.

    ``scores`` must carry ``hsz_id`` and the three probabilities plus ``net``;
    ``posts`` must carry ``hsz_id``, ``ts`` and ``upvotes``. They are joined on
    ``hsz_id`` (inner). Returns a frame with exactly
    :data:`pressflow.schema.SIGNALS_COLUMNS`, sorted by date, warm-up rows
    (null ``z_score``) removed.
    """
    missing_scores = [c for c in REQUIRED_SCORE_COLUMNS if c not in scores.columns]
    if missing_scores:
        raise KeyError(f"scores missing columns: {missing_scores}")
    missing_posts = [c for c in REQUIRED_POST_COLUMNS if c not in posts.columns]
    if missing_posts:
        raise KeyError(f"posts missing columns: {missing_posts}")

    if cfg.aggregation.weight_scheme != WEIGHT_SCHEME_ENGAGEMENT:
        raise ValueError(
            "unsupported aggregation.weight_scheme: "
            f"{cfg.aggregation.weight_scheme!r} (expected "
            f"{WEIGHT_SCHEME_ENGAGEMENT!r})"
        )

    merged = scores[list(REQUIRED_SCORE_COLUMNS)].merge(
        posts[list(REQUIRED_POST_COLUMNS)], on=HSZ_ID, how="inner"
    )
    merged = merged.assign(
        **{
            DATE: to_budapest_day(merged[TS]),
            "w": 1.0 + merged[UPVOTES].astype("float64"),
            "_entropy": _shannon_entropy(merged),
        }
    )
    merged["_wnet"] = merged["w"] * merged[NET]
    merged["_wbear"] = merged["w"] * merged[P_BEAR]

    grouped = merged.groupby(DATE, sort=True)
    agg = grouped.agg(
        wnet=("_wnet", "sum"),
        wsum=("w", "sum"),
        wbear=("_wbear", "sum"),
        count=(NET, "size"),
        std=(NET, lambda s: s.std(ddof=0)),
        entropy=("_entropy", "mean"),
    )

    out = pd.DataFrame(
        {
            SCORE_WMEAN: agg["wnet"] / agg["wsum"],
            BEAR_WMEAN: agg["wbear"] / agg["wsum"],
            COUNT: agg["count"].astype("int64"),
            # ddof=0 yields NaN for a single-post day; the spec wants 0.0.
            DISAGREEMENT_STD: agg["std"].fillna(0.0),
            ENTROPY: agg["entropy"],
        }
    )
    out.index.name = DATE

    z_window = cfg.aggregation.z_window
    z_min_periods = cfg.aggregation.z_min_periods
    rolling = out[SCORE_WMEAN].rolling(window=z_window, min_periods=z_min_periods)
    out[Z_SCORE] = (out[SCORE_WMEAN] - rolling.mean()) / rolling.std(ddof=0)

    out = out.sort_values(DATE).reset_index()
    out = out[out[Z_SCORE].notna()].reset_index(drop=True)
    return out[list(SIGNALS_COLUMNS)]


def signals_path(config: Config | None = None) -> Path:
    """Absolute path of the processed signals artifact."""
    config = config or get_config()
    return REPO_ROOT / config.paths.processed / PROCESSED_FILENAME


def save_signals(signals: pd.DataFrame, path: Path) -> None:
    """Write the signal frame with the canonical ``SIGNALS_SCHEMA``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(signals, schema=SIGNALS_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


def build_and_save(config: Config | None = None) -> pd.DataFrame:
    """Read scores + posts, aggregate and write ``data/processed/signals.parquet``."""
    config = config or get_config()
    processed = REPO_ROOT / config.paths.processed
    scores_file = processed / SCORES_FILENAME
    posts_file = processed / POSTS_FILENAME
    if not scores_file.is_file():
        raise FileNotFoundError(f"scores not found: {scores_file} (run scoring first)")
    if not posts_file.is_file():
        raise FileNotFoundError(f"posts not found: {posts_file} (run T1 first)")

    scores = pd.read_parquet(scores_file)
    posts = pd.read_parquet(posts_file, columns=list(REQUIRED_POST_COLUMNS))
    signals = aggregate_daily(scores, posts, config)
    out_file = signals_path(config)
    save_signals(signals, out_file)
    print(f"wrote {out_file} ({len(signals)} rows, {len(signals.columns)} columns)")
    return signals


def main() -> None:  # pragma: no cover - thin CLI wrapper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_and_save()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "PROCESSED_FILENAME",
    "PROB_FLOOR",
    "REQUIRED_SCORE_COLUMNS",
    "REQUIRED_POST_COLUMNS",
    "WEIGHT_SCHEME_ENGAGEMENT",
    "to_budapest_day",
    "aggregate_daily",
    "signals_path",
    "save_signals",
    "build_and_save",
]

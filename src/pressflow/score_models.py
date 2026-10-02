"""T3.2 — candidate sentiment model runner (ADR-0001).

Scores the T3.1 two-model consensus eval set (``eval/eval_set.csv``) with every
candidate off-the-shelf sentiment model and normalises each model's *native*
label space onto the pipeline's ``p_bull / p_bear / p_neutral`` probabilities
(with ``net = p_bull - p_bear``), writing
``data/processed/eval_scores.parquet``.

No fine-tuning (ADR-0001): the candidates are used exactly as published.

Label mapping — by MEANING, not position
----------------------------------------
Each model ships a different label space, and the two NYTK models expose only
generic ``LABEL_n`` names, so the semantic meaning was verified against the
model cards *and* empirically with known-polarity Hungarian probes:

============  ======================  ==================================
model         native labels           mapping
============  ======================  ==================================
hts5 (5-cls)  0 very neg · 1 neg ·    {0,1}→bear · {2}→neutral · {3,4}→bull
              2 neutral · 3 pos ·
              4 very pos
ohb3 (3-cls)  0 neg · 1 neutral ·     {0}→bear · {1}→neutral · {2}→bull
              2 pos
cardiffnlp    negative · neutral ·    negative→bear · neutral→neutral ·
               positive                positive→bull
============  ======================  ==================================

Named labels are parsed semantically (:func:`group_from_label_name`); for the
generic ``LABEL_n`` models the documented order above is used
(:data:`GENERIC_LABEL_GROUPS`).  Probabilities are summed within each group and
renormalised so they sum to 1.

CLI::

    uv run python -m pressflow.score_models
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pressflow.config import REPO_ROOT, Config, get_config

logger = logging.getLogger("pressflow.score_models")

# --- label groups ----------------------------------------------------------

BULL = "bull"
BEAR = "bear"
NEUTRAL = "neutral"

#: The three normalised classes, in a stable order.
GROUPS: tuple[str, ...] = (BULL, BEAR, NEUTRAL)

#: Documented native label ordering for models whose config only exposes
#: generic ``LABEL_n`` names.  Verified against the model cards and with
#: known-polarity probes (see module docstring).
GENERIC_LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "NYTK/sentiment-hts5-xlm-roberta-hungarian": (
        BEAR,  # 0 — very negative
        BEAR,  # 1 — negative
        NEUTRAL,  # 2 — neutral
        BULL,  # 3 — positive
        BULL,  # 4 — very positive
    ),
    "NYTK/sentiment-ohb3-xlm-roberta-hungarian": (
        BEAR,  # 0 — negative
        NEUTRAL,  # 1 — neutral
        BULL,  # 2 — positive
    ),
}

# --- artifact --------------------------------------------------------------

PROCESSED_FILENAME = "eval_scores.parquet"
EVAL_SET_CSV = REPO_ROOT / "eval" / "eval_set.csv"

#: Column order of the eval-scores artifact.
EVAL_SCORES_COLUMNS: tuple[str, ...] = (
    "hsz_id",
    "post_id",
    "model_id",
    "raw_label",
    "raw_score",
    "p_bull",
    "p_bear",
    "p_neutral",
    "net",
)

EVAL_SCORES_SCHEMA = pa.schema(
    [
        pa.field("hsz_id", pa.int64(), nullable=False),
        pa.field("post_id", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("raw_label", pa.string(), nullable=False),
        pa.field("raw_score", pa.float64(), nullable=False),
        pa.field("p_bull", pa.float64(), nullable=False),
        pa.field("p_bear", pa.float64(), nullable=False),
        pa.field("p_neutral", pa.float64(), nullable=False),
        pa.field("net", pa.float64(), nullable=False),
    ]
)

__all__ = [
    "BULL",
    "BEAR",
    "NEUTRAL",
    "GROUPS",
    "GENERIC_LABEL_GROUPS",
    "EVAL_SCORES_COLUMNS",
    "EVAL_SCORES_SCHEMA",
    "Probs",
    "group_from_label_name",
    "label_group_map",
    "normalize_distribution",
    "top_prediction",
    "score_rows_for_texts",
    "eval_scores_path",
    "save_eval_scores",
    "build_eval_scores",
    "load_eval_set",
    "main",
]


# --- pure helpers ----------------------------------------------------------


@dataclass(frozen=True)
class Probs:
    """Normalised sentiment probabilities for one post."""

    p_bull: float
    p_bear: float
    p_neutral: float
    net: float


def group_from_label_name(name: str) -> str | None:
    """Map a *semantic* native label (e.g. ``"very negative"``) to a group.

    Returns ``None`` for non-semantic names such as ``"LABEL_3"`` so callers can
    fall back to a documented ordering.  Matching is on word stems, so
    ``"very negative"`` and ``"negative"`` both map to :data:`BEAR`.
    """
    tokens = re.findall(r"[a-z]+", name.lower())
    if "neutral" in tokens:
        return NEUTRAL
    if any(token.startswith("neg") for token in tokens):
        return BEAR
    if any(token.startswith("pos") for token in tokens):
        return BULL
    return None


def label_group_map(
    id2label: Mapping[int, str], model_id: str
) -> dict[str, str]:
    """Return ``{native_label_name: group}`` for a model.

    Semantic names are parsed by meaning.  If every name is generic (``LABEL_n``)
    the documented :data:`GENERIC_LABEL_GROUPS` ordering for ``model_id`` is
    used.  Raises :class:`ValueError` if the semantics cannot be established.
    """
    semantic: dict[str, str] = {}
    for name in id2label.values():
        group = group_from_label_name(name)
        if group is None:
            semantic = {}
            break
        semantic[name] = group
    if semantic:
        return semantic

    scheme = GENERIC_LABEL_GROUPS.get(model_id)
    if scheme is None:
        raise ValueError(
            f"cannot determine label semantics for {model_id!r} from "
            f"id2label={dict(id2label)!r}; add a documented mapping"
        )
    ordered = [name for _, name in sorted(id2label.items())]
    if len(ordered) != len(scheme):
        raise ValueError(
            f"{model_id!r}: {len(ordered)} labels but scheme has {len(scheme)}"
        )
    return dict(zip(ordered, scheme))


def normalize_distribution(
    scores: Sequence[Mapping[str, Any]], name_to_group: Mapping[str, str]
) -> Probs:
    """Sum a native score distribution into normalised bull/bear/neutral probs.

    ``scores`` is the pipeline output for one text: a sequence of
    ``{"label": str, "score": float}`` mappings covering the full native label
    space.  Group probabilities are renormalised to sum to 1 and ``net`` is
    asserted to lie in ``[-1, 1]``.
    """
    if not scores:
        raise ValueError("empty score distribution")
    totals = {BULL: 0.0, BEAR: 0.0, NEUTRAL: 0.0}
    for item in scores:
        label = str(item["label"])
        if label not in name_to_group:
            raise KeyError(f"unknown native label {label!r}")
        totals[name_to_group[label]] += float(item["score"])

    total = sum(totals.values())
    if total <= 0.0:
        raise ValueError(f"non-positive probability mass: {total}")
    p_bull = totals[BULL] / total
    p_bear = totals[BEAR] / total
    p_neutral = totals[NEUTRAL] / total
    net = p_bull - p_bear
    if not -1.0 <= net <= 1.0:
        raise AssertionError(f"net out of [-1, 1]: {net}")
    return Probs(p_bull=p_bull, p_bear=p_bear, p_neutral=p_neutral, net=net)


def top_prediction(scores: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    """Return the highest-scoring native ``(label, score)``."""
    if not scores:
        raise ValueError("empty score distribution")
    best = max(scores, key=lambda item: float(item["score"]))
    return str(best["label"]), float(best["score"])


def _as_full_distribution(output: Any) -> list[dict[str, Any]]:
    """Normalise a pipeline output into a list of label/score dicts."""
    if isinstance(output, Mapping):  # single-label mode (top_k=1)
        return [dict(output)]
    return [dict(item) for item in output]


def score_rows_for_texts(
    texts: Sequence[str],
    outputs: Sequence[Any],
    id2label: Mapping[int, str],
    model_id: str,
) -> list[dict[str, Any]]:
    """Turn raw pipeline ``outputs`` into per-post score rows.

    ``outputs[i]`` must be the pipeline result for ``texts[i]``.  Pure
    (no model access) so it can be unit-tested with synthetic outputs.
    """
    if len(texts) != len(outputs):
        raise ValueError("texts and outputs must be the same length")
    name_to_group = label_group_map(id2label, model_id)
    rows: list[dict[str, Any]] = []
    for output in outputs:
        probs = normalize_distribution(_as_full_distribution(output), name_to_group)
        raw_label, raw_score = top_prediction(_as_full_distribution(output))
        rows.append(
            {
                "raw_label": raw_label,
                "raw_score": raw_score,
                "p_bull": probs.p_bull,
                "p_bear": probs.p_bear,
                "p_neutral": probs.p_neutral,
                "net": probs.net,
            }
        )
    return rows


# --- I/O -------------------------------------------------------------------


def eval_scores_path(config: Config | None = None) -> Path:
    """Absolute path of the eval-scores artifact."""
    config = config or get_config()
    return REPO_ROOT / config.paths.processed / PROCESSED_FILENAME


def load_eval_set(path: Path = EVAL_SET_CSV) -> pd.DataFrame:
    """Load ``eval/eval_set.csv`` with the expected columns validated."""
    frame = pd.read_csv(path, dtype={"hsz_id": "int64", "post_id": "int64"})
    required = {"hsz_id", "post_id", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def save_eval_scores(rows: pd.DataFrame, path: Path) -> None:
    """Write score rows with the canonical schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        rows[list(EVAL_SCORES_COLUMNS)], schema=EVAL_SCORES_SCHEMA, preserve_index=False
    )
    pq.write_table(table, path)


# --- runner ----------------------------------------------------------------


def _load_pipeline(model_id: str) -> Any:
    """Load a CPU text-classification pipeline for ``model_id``."""
    from transformers import pipeline  # heavy import, kept local

    logger.info("loading model %s (CPU)", model_id)
    return pipeline("sentiment-analysis", model=model_id, device=-1)


def build_eval_scores(
    config: Config | None = None,
    *,
    model_ids: Sequence[str] | None = None,
    eval_set: pd.DataFrame | None = None,
    pipe_factory: Any = _load_pipeline,
    batch_size: int = 16,
    max_length: int = 512,
) -> pd.DataFrame:
    """Score the eval set with every candidate model; return the score frame.

    ``pipe_factory`` maps a model id to a loaded pipeline (injectable for
    tests).  The eval set defaults to ``eval/eval_set.csv``.  Posts longer than
    ``max_length`` tokens are truncated (512 = XLM-R's positional limit minus
    the two special tokens; ``cardiffnlp`` ships no ``model_max_length``).
    """
    config = config or get_config()
    model_ids = list(model_ids or config.models)
    frame = eval_set if eval_set is not None else load_eval_set()
    texts = frame["text"].fillna("").astype(str).tolist()

    all_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        pipe = pipe_factory(model_id)
        id2label = dict(pipe.model.config.id2label)
        logger.info(
            "scoring %d posts with %s (id2label=%s)",
            len(texts),
            model_id,
            id2label,
        )
        outputs = pipe(
            texts,
            top_k=None,
            truncation=True,
            max_length=max_length,
            batch_size=batch_size,
        )
        scored = score_rows_for_texts(texts, outputs, id2label, model_id)
        for (_, post), row in zip(frame.iterrows(), scored):
            all_rows.append(
                {
                    "hsz_id": int(post["hsz_id"]),
                    "post_id": int(post["post_id"]),
                    "model_id": model_id,
                    **row,
                }
            )
        logger.info("done %s", model_id)

    result = pd.DataFrame(all_rows, columns=list(EVAL_SCORES_COLUMNS))
    return result


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the T3.1 eval set with every candidate sentiment model."
    )
    parser.add_argument(
        "--eval-set", type=Path, default=EVAL_SET_CSV, help="eval_set.csv path"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output parquet (default: <processed>/eval_scores.parquet)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="inference batch size"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="override candidate model ids (default: config candidates)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    config = get_config()
    frame = load_eval_set(args.eval_set)
    scores = build_eval_scores(
        config,
        model_ids=args.models,
        eval_set=frame,
        batch_size=args.batch_size,
    )
    out_path = args.out or eval_scores_path(config)
    save_eval_scores(scores, out_path)
    print(
        f"wrote {len(scores)} rows "
        f"({scores['model_id'].nunique()} models x {len(frame)} posts) -> {out_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

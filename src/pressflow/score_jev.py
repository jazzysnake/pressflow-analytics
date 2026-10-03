"""T3.6 — Jev (TypeSafe API) candidate sentiment scorer.

Scores the T3.1 two-model consensus eval set (``eval/eval_set.csv``) with
TypeSafe's API-only decision model *Jev* using its ``choice`` primitive, and
normalises its calibrated answer into the pipeline's
``p_bull / p_bear / p_neutral`` probabilities (``net = p_bull - p_bear``),
writing ``data/processed/jev_scores.parquet``.

Why Jev
-------
The off-the-shelf HF classifiers are near-chance on directional sentiment
(ADR-0001's "mostly right" bar is not met). Jev returns *calibrated*
probabilities over a user-defined option set, so it can be asked the exact
directional question instead of relying on a generic polarity label space.

Secret handling
---------------
The TypeSafe API key lives in ``.env.secrets`` (gitignored) under
``TYPESAFE_API_KEY``. It is loaded into the process environment by
:func:`load_secrets` and used only in the ``Authorization: Bearer`` header.
The key is never logged, printed or committed.

CLI::

    uv run python -m pressflow.score_jev
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.score_models import BEAR, BULL, NEUTRAL, load_eval_set

logger = logging.getLogger("pressflow.score_jev")

#: TypeSafe SystemOne endpoint (https://docs.typesafe.ai/introduction/quickstart).
SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"

#: Requested Jev model tag (the response reports the concrete version).
JEV_MODEL = "jev-latest"

#: Model id recorded in the artifact / report.
MODEL_ID = JEV_MODEL

#: Question id used in the request / response payload.
QUESTION_ID = "sentiment"

#: The directional choice criteria sent to Jev.
CRITERIA: dict[str, str] = {
    BULL: "expects the price to rise (buy, positive)",
    BEAR: "expects the price to fall (sell, negative)",
    NEUTRAL: "no directional view (question, quote, off-topic)",
}

#: Full ``questions`` block for the request body.
QUESTIONS: dict[str, Any] = {
    QUESTION_ID: {
        "type": "choice",
        "instructions": (
            "What is the author's directional view on the OTP share price?"
        ),
        "criteria": CRITERIA,
    }
}

#: Politeness delay between API calls (seconds).
REQUEST_DELAY_SEC = 0.2

#: Maximum attempts per post (1 initial + retries).
MAX_ATTEMPTS = 5

#: Backoff base in seconds for retries (exponential: base * 2**(attempt-1)).
BACKOFF_BASE_SEC = 1.0

#: HTTP statuses worth retrying (rate limit / transient server errors).
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Default location of the gitignored secrets file.
DEFAULT_SECRETS_PATH = REPO_ROOT / ".env.secrets"

#: Artifact filename under the processed data directory.
PROCESSED_FILENAME = "jev_scores.parquet"

#: Column order of the Jev-scores artifact.
JEV_SCORES_COLUMNS: tuple[str, ...] = (
    "hsz_id",
    "post_id",
    "model_id",
    "raw_label",
    "confidence",
    "p_bull",
    "p_bear",
    "p_neutral",
    "net",
)

JEV_SCORES_SCHEMA = pa.schema(
    [
        pa.field("hsz_id", pa.int64(), nullable=False),
        pa.field("post_id", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("raw_label", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("p_bull", pa.float64(), nullable=False),
        pa.field("p_bear", pa.float64(), nullable=False),
        pa.field("p_neutral", pa.float64(), nullable=False),
        pa.field("net", pa.float64(), nullable=False),
    ]
)

__all__ = [
    "SYSTEMONE_URL",
    "JEV_MODEL",
    "MODEL_ID",
    "CRITERIA",
    "QUESTIONS",
    "JEV_SCORES_COLUMNS",
    "JEV_SCORES_SCHEMA",
    "JevScore",
    "load_secrets",
    "get_api_key",
    "build_request_body",
    "parse_answer",
    "score_rows_for_answers",
    "score_post",
    "score_post_with_usage",
    "score_eval_set",
    "jev_scores_path",
    "save_jev_scores",
    "main",
]


# --- secrets ---------------------------------------------------------------


def load_secrets(path: str | os.PathLike[str] = DEFAULT_SECRETS_PATH) -> int:
    """Load ``KEY=value`` lines from *path* into ``os.environ``.

    Minimal, dependency-free loader: each non-empty line is split on the
    **first** ``=`` and only set when not already present
    (:func:`os.environ.setdefault`). Returns the number of variables set from
    the file. Values are never logged.
    """
    secrets_path = Path(path)
    if not secrets_path.is_file():
        return 0
    loaded = 0
    for raw_line in secrets_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if not name:
            continue
        os.environ.setdefault(name, value.strip())
        loaded += 1
    return loaded


def get_api_key(path: str | os.PathLike[str] = DEFAULT_SECRETS_PATH) -> str:
    """Return the TypeSafe API key, loading ``.env.secrets`` if needed.

    Raises :class:`RuntimeError` if the variable is absent. The value is never
    logged or echoed.
    """
    load_secrets(path)
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set (expected in .env.secrets or the "
            "environment)"
        )
    return key


# --- pure mapping ----------------------------------------------------------


@dataclass(frozen=True)
class JevScore:
    """Normalised directional score for one post."""

    raw_label: str
    confidence: float
    p_bull: float
    p_bear: float
    p_neutral: float
    net: float


def build_request_body(text: str) -> dict[str, Any]:
    """Build the SystemOne request body for one post's text."""
    return {"state": text, "model": JEV_MODEL, "questions": QUESTIONS}


def parse_answer(answer: Mapping[str, Any]) -> JevScore:
    """Map a Jev ``choice`` answer to normalised directional probabilities.

    ``answer`` is the ``answers.<qid>`` object from the SystemOne response:
    ``{"type", "choice", "confidence", "probabilities"}``. Probabilities are
    renormalised so they sum to 1 and ``net = p_bull - p_bear`` is asserted to
    lie in ``[-1, 1]``. The hard ``choice`` is used as ``raw_label`` when
    present, otherwise the argmax probability decides. Pure (no network) so it
    can be unit-tested directly.
    """
    if not isinstance(answer, Mapping):
        raise ValueError(f"expected an answer mapping, got {type(answer).__name__}")
    probs = answer.get("probabilities") or {}
    if not isinstance(probs, Mapping):
        raise ValueError("answer.probabilities must be a mapping")

    totals = {BULL: 0.0, BEAR: 0.0, NEUTRAL: 0.0}
    for label in (BULL, BEAR, NEUTRAL):
        totals[label] = float(probs.get(label, 0.0))

    total = sum(totals.values())
    if total <= 0.0:
        raise ValueError(f"non-positive probability mass: {total}")

    p_bull = totals[BULL] / total
    p_bear = totals[BEAR] / total
    p_neutral = totals[NEUTRAL] / total
    net = p_bull - p_bear
    if not -1.0 <= net <= 1.0:
        raise AssertionError(f"net out of [-1, 1]: {net}")

    choice = answer.get("choice")
    if choice in (BULL, BEAR, NEUTRAL):
        raw_label = str(choice)
    else:
        raw_label = max(
            ((BULL, p_bull), (BEAR, p_bear), (NEUTRAL, p_neutral)),
            key=lambda item: item[1],
        )[0]

    confidence = answer.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = max(p_bull, p_bear, p_neutral)

    return JevScore(
        raw_label=raw_label,
        confidence=confidence,
        p_bull=p_bull,
        p_bear=p_bear,
        p_neutral=p_neutral,
        net=net,
    )


def score_rows_for_answers(
    posts: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Turn a list of Jev answers into per-post score rows (pure).

    ``posts[i]`` must carry ``hsz_id`` / ``post_id`` for ``answers[i]``.
    """
    if len(posts) != len(answers):
        raise ValueError("posts and answers must be the same length")
    rows: list[dict[str, Any]] = []
    for post, answer in zip(posts, answers):
        score = parse_answer(answer)
        rows.append(
            {
                "hsz_id": int(post["hsz_id"]),
                "post_id": int(post["post_id"]),
                "model_id": MODEL_ID,
                "raw_label": score.raw_label,
                "confidence": score.confidence,
                "p_bull": score.p_bull,
                "p_bear": score.p_bear,
                "p_neutral": score.p_neutral,
                "net": score.net,
            }
        )
    return rows


# --- API -------------------------------------------------------------------


def _post_once(
    session: requests.Session, text: str, api_key: str, timeout: float
) -> requests.Response:
    return session.post(
        SYSTEMONE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=build_request_body(text),
        timeout=timeout,
    )


def score_post_with_usage(
    text: str,
    *,
    session: requests.Session | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_base: float = BACKOFF_BASE_SEC,
    sleep: Any = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one post with Jev, returning ``(answer, usage)``.

    Identical retry behaviour to :func:`score_post` (429/5xx and network
    errors with exponential backoff) but also returns the response's ``usage``
    mapping (``input_tokens`` / ``output_tokens``) so callers can track cost.
    Raises the last :class:`requests.HTTPError` /
    :class:`requests.RequestException` when every attempt fails.
    """
    own_session = session is None
    session = session or requests.Session()
    key = api_key or get_api_key()
    last_exc: Exception | None = None
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                response = _post_once(session, text, key, timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    raise
                sleep(backoff_base * 2 ** (attempt - 1))
                continue

            if response.status_code in RETRY_STATUSES:
                last_exc = requests.HTTPError(
                    f"retryable HTTP {response.status_code}", response=response
                )
                if attempt >= max_attempts:
                    raise last_exc
                sleep(backoff_base * 2 ** (attempt - 1))
                continue

            response.raise_for_status()
            payload = response.json()
            answers = payload.get("answers")
            if not isinstance(answers, Mapping) or QUESTION_ID not in answers:
                raise ValueError(
                    f"unexpected SystemOne response: missing answers.{QUESTION_ID}"
                )
            usage = payload.get("usage")
            return (
                dict(answers[QUESTION_ID]),
                dict(usage) if isinstance(usage, Mapping) else {},
            )
        raise last_exc if last_exc else RuntimeError("unreachable")
    finally:
        if own_session:
            session.close()


def score_post(
    text: str,
    *,
    session: requests.Session | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_base: float = BACKOFF_BASE_SEC,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Score one post with Jev, retrying 429/5xx with exponential backoff.

    Returns the raw ``answers[QUESTION_ID]`` mapping. Raises the last
    :class:`requests.HTTPError` / :class:`requests.RequestException` when every
    attempt fails.
    """
    answer, _ = score_post_with_usage(
        text,
        session=session,
        api_key=api_key,
        timeout=timeout,
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        sleep=sleep,
    )
    return answer


def score_eval_set(
    eval_set: pd.DataFrame | None = None,
    *,
    session: requests.Session | None = None,
    api_key: str | None = None,
    delay: float = REQUEST_DELAY_SEC,
    timeout: float = 120.0,
    sleep: Any = time.sleep,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Score every eval post with Jev and return the score frame."""
    frame = eval_set if eval_set is not None else load_eval_set()
    own_session = session is None
    session = session or requests.Session()
    key = api_key or get_api_key()
    posts: list[Mapping[str, Any]] = []
    answers: list[Mapping[str, Any]] = []
    n = len(frame)
    try:
        for i, (_, post) in enumerate(frame.iterrows()):
            text = "" if pd.isna(post["text"]) else str(post["text"])
            answers.append(
                score_post(text, session=session, api_key=key, timeout=timeout)
            )
            posts.append({"hsz_id": post["hsz_id"], "post_id": post["post_id"]})
            if (i + 1) % progress_every == 0 or (i + 1) == n:
                logger.info("scored %d/%d posts", i + 1, n)
            if delay and (i + 1) < n:
                sleep(delay)
    finally:
        if own_session:
            session.close()

    rows = score_rows_for_answers(posts, answers)
    return pd.DataFrame(rows, columns=list(JEV_SCORES_COLUMNS))


# --- I/O -------------------------------------------------------------------


def jev_scores_path(config: Config | None = None) -> Path:
    """Absolute path of the Jev-scores artifact."""
    config = config or get_config()
    return REPO_ROOT / config.paths.processed / PROCESSED_FILENAME


def save_jev_scores(rows: pd.DataFrame, path: Path) -> None:
    """Write Jev score rows with the canonical schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        rows[list(JEV_SCORES_COLUMNS)],
        schema=JEV_SCORES_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, path)


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the T3.1 eval set with Jev (TypeSafe SystemOne)."
    )
    parser.add_argument(
        "--eval-set", type=Path, default=None, help="eval_set.csv path"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output parquet (default: <processed>/jev_scores.parquet)",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY_SEC, help="seconds between calls"
    )
    parser.add_argument(
        "--secrets", type=Path, default=DEFAULT_SECRETS_PATH, help=".env.secrets path"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    load_secrets(args.secrets)
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY is not set"

    config = get_config()
    frame = load_eval_set(args.eval_set) if args.eval_set else load_eval_set()
    scores = score_eval_set(frame, delay=args.delay)
    out_path = args.out or jev_scores_path(config)
    save_jev_scores(scores, out_path)
    print(f"wrote {len(scores)} rows -> {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

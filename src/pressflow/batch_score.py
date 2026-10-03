"""T3.4 — concurrent, resumable batch scoring with Jev (TypeSafe API).

Scores every post in ``data/processed/posts.parquet`` (newest → oldest) with
the chosen Jev ``choice`` model and writes ``data/processed/scores.parquet``
keyed by the ``hsz_id`` natural key (T0.3), plus two sidecar files:

    data/processed/scores.parquet            # append/rewrite, dedupe on hsz_id
    data/processed/scores_checkpoint.json    # completed hsz_ids (resume state)
    data/processed/scores_failed.json        # failed hsz_ids + error messages

Design
------
* **Concurrency** — a :class:`ThreadPoolExecutor` with
  ``scoring.concurrency`` workers. Each task reuses the request/retry logic in
  :mod:`pressflow.score_jev` (:func:`score_post_with_usage`) and sleeps
  ``scoring.request_delay_sec`` between calls per worker.
* **Resumable** — on start, ``hsz_id``s present in ``scores.parquet`` *or* the
  checkpoint are skipped, so an interrupted run never pays twice. The
  checkpoint is flushed every ``scoring.flush_every`` completions and on
  shutdown / ``KeyboardInterrupt``.
* **Failure tolerant** — each post is retried with exponential backoff
  (``scoring.retries`` attempts); a post that still fails is recorded in
  ``scores_failed.json`` and the run continues. The whole run aborts only after
  ``scoring.max_consecutive_failures`` failures *in a row* (any success resets
  the counter).
* **Cost capped** — cumulative ``usage`` tokens are converted to an estimated
  USD figure ($0.042 / 1M tokens, input and output assumed equal) and the run
  aborts before exceeding ``scoring.cost_cap_usd``. Throughput and ETA are
  logged alongside cost.

The API key is loaded by :func:`pressflow.score_jev.load_secrets` from the
gitignored ``.env.secrets`` and never logged.

CLI::

    uv run python -m pressflow.batch_score --limit 50      # smoke slice
    uv run python -m pressflow.batch_score --resume        # full resumable run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.score_jev import (
    DEFAULT_SECRETS_PATH,
    JEV_MODEL,
    load_secrets,
    parse_answer,
    score_post_with_usage,
)

logger = logging.getLogger("pressflow.batch_score")

#: Input posts artifact.
INPUT_FILENAME = "posts.parquet"
#: Output scores artifact.
SCORES_FILENAME = "scores.parquet"
#: Resume checkpoint sidecar.
CHECKPOINT_FILENAME = "scores_checkpoint.json"
#: Permanently failed posts sidecar.
FAILED_FILENAME = "scores_failed.json"

#: TypeSafe Jev pricing used for the running estimate (USD per 1M tokens),
#: applied to input and output tokens alike.
PRICE_PER_MILLION_TOKENS = 0.042

#: Number of in-flight futures kept per worker (bounded submission window).
IN_FLIGHT_PER_WORKER = 4

#: Column order of the batch scores artifact: the T0.3 minimum plus the extras
#: requested by T3.4 (``confidence`` / ``raw_label``).
SCORES_OUT_COLUMNS: tuple[str, ...] = (
    "hsz_id",
    "model",
    "p_bull",
    "p_bear",
    "p_neutral",
    "net",
    "confidence",
    "raw_label",
)

SCORES_OUT_SCHEMA = pa.schema(
    [
        pa.field("hsz_id", pa.int64(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("p_bull", pa.float64(), nullable=False),
        pa.field("p_bear", pa.float64(), nullable=False),
        pa.field("p_neutral", pa.float64(), nullable=False),
        pa.field("net", pa.float64(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("raw_label", pa.string(), nullable=False),
    ]
)

#: Score function contract: post text -> ``(answer, usage)``.
ScoreFn = Callable[[str], tuple[Mapping[str, Any], Mapping[str, Any]]]

__all__ = [
    "INPUT_FILENAME",
    "SCORES_FILENAME",
    "CHECKPOINT_FILENAME",
    "FAILED_FILENAME",
    "PRICE_PER_MILLION_TOKENS",
    "SCORES_OUT_COLUMNS",
    "SCORES_OUT_SCHEMA",
    "Post",
    "BatchPaths",
    "BatchResult",
    "CostTracker",
    "ConsecutiveFailureTracker",
    "batch_paths",
    "estimate_cost_usd",
    "usage_tokens",
    "build_score_row",
    "load_posts",
    "load_completed_hsz_ids",
    "save_checkpoint",
    "load_failed_errors",
    "save_failed",
    "write_scores",
    "run_batch",
    "main",
]


# --- paths -----------------------------------------------------------------


@dataclass(frozen=True)
class BatchPaths:
    """Absolute paths of the four batch artifacts."""

    input: Path
    scores: Path
    checkpoint: Path
    failed: Path


def batch_paths(config: Config | None = None) -> BatchPaths:
    """Resolve the batch artifact paths from ``paths.processed``."""
    config = config or get_config()
    processed = REPO_ROOT / config.paths.processed
    return BatchPaths(
        input=processed / INPUT_FILENAME,
        scores=processed / SCORES_FILENAME,
        checkpoint=processed / CHECKPOINT_FILENAME,
        failed=processed / FAILED_FILENAME,
    )


# --- posts -----------------------------------------------------------------


@dataclass(frozen=True)
class Post:
    """One post to score (natural key + text)."""

    hsz_id: int
    text: str


def load_posts(path: Path, limit: int | None = None) -> list[Post]:
    """Load posts, **newest → oldest**, optionally keeping only the first *limit*."""
    frame = pd.read_parquet(path, columns=["hsz_id", "text", "ts"])
    frame = frame.sort_values("ts", ascending=False, kind="stable")
    if limit is not None:
        frame = frame.head(limit)
    return [
        Post(int(hsz_id), "" if pd.isna(text) else str(text))
        for hsz_id, text in zip(frame["hsz_id"], frame["text"])
    ]


# --- cost ------------------------------------------------------------------


def usage_tokens(usage: Mapping[str, Any] | None) -> tuple[int, int]:
    """Extract ``(input_tokens, output_tokens)`` from a response ``usage`` dict.

    Accepts both the TypeSafe ``input_tokens`` / ``output_tokens`` names and the
    OpenAI-style ``prompt_tokens`` / ``completion_tokens`` aliases. Missing or
    unparseable values count as zero.
    """
    if not isinstance(usage, Mapping):
        return 0, 0

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return _int(input_tokens), _int(output_tokens)


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    price_per_million: float = PRICE_PER_MILLION_TOKENS,
) -> float:
    """Estimated USD cost for *input_tokens* + *output_tokens* at *price_per_million*."""
    total = int(input_tokens) + int(output_tokens)
    return total / 1_000_000 * price_per_million


@dataclass
class CostTracker:
    """Accumulate token usage and report an estimated USD cost."""

    price_per_million: float = PRICE_PER_MILLION_TOKENS
    input_tokens: int = 0
    output_tokens: int = 0

    def add_usage(self, usage: Mapping[str, Any] | None) -> tuple[int, int]:
        """Add one response's usage; returns the ``(input, output)`` delta."""
        delta = usage_tokens(usage)
        self.input_tokens += delta[0]
        self.output_tokens += delta[1]
        return delta

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_cost_usd(
            self.input_tokens, self.output_tokens, self.price_per_million
        )

    def exceeds_cap(self, cap: float) -> bool:
        return self.estimated_cost_usd > cap


# --- failure tolerance -----------------------------------------------------


@dataclass
class ConsecutiveFailureTracker:
    """Track consecutive failures; a success resets the counter."""

    max_consecutive: int
    consecutive: int = 0
    total_failures: int = 0

    def record_success(self) -> None:
        self.consecutive = 0

    def record_failure(self) -> bool:
        """Record a failure; returns ``True`` when the abort threshold is hit."""
        self.consecutive += 1
        self.total_failures += 1
        return self.should_abort

    @property
    def should_abort(self) -> bool:
        return self.consecutive >= self.max_consecutive


# --- score rows ------------------------------------------------------------


def build_score_row(
    hsz_id: int, answer: Mapping[str, Any], model: str = JEV_MODEL
) -> dict[str, Any]:
    """Normalise one Jev answer into a ``scores`` row via ``parse_answer``."""
    score = parse_answer(answer)
    return {
        "hsz_id": int(hsz_id),
        "model": model,
        "p_bull": score.p_bull,
        "p_bear": score.p_bear,
        "p_neutral": score.p_neutral,
        "net": score.net,
        "confidence": score.confidence,
        "raw_label": score.raw_label,
    }


# --- checkpoint / sidecars -------------------------------------------------


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("could not read %s; ignoring", path)
        return None


def load_checkpoint_ids(path: Path) -> set[int]:
    """Read the completed ``hsz_id`` set from the checkpoint sidecar."""
    data = _read_json(path)
    if data is None:
        return set()
    if isinstance(data, dict):
        raw = data.get("completed_hsz_ids", [])
    elif isinstance(data, list):
        raw = data
    else:
        return set()
    if not isinstance(raw, list):
        return set()
    return {int(item) for item in raw}


def save_checkpoint(path: Path, completed_hsz_ids: Any) -> None:
    """Atomically write ``{"completed_hsz_ids": [...]}`` to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"completed_hsz_ids": sorted(int(i) for i in completed_hsz_ids)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_completed_hsz_ids(scores_path: Path, checkpoint_path: Path) -> set[int]:
    """Union of ``hsz_id``s already written to scores.parquet and the checkpoint."""
    done: set[int] = set()
    if scores_path.is_file():
        table = pq.read_table(scores_path, columns=["hsz_id"])
        done.update(v for v in table.column("hsz_id").to_pylist() if v is not None)
    done.update(load_checkpoint_ids(checkpoint_path))
    return done


def load_failed_errors(path: Path) -> dict[int, str]:
    """Read the ``{hsz_id: error}`` map from ``scores_failed.json``."""
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}
    errors = data.get("errors", {})
    if not isinstance(errors, dict):
        return {}
    result: dict[int, str] = {}
    for key, value in errors.items():
        try:
            result[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return result


def save_failed(path: Path, errors: Mapping[int, str]) -> None:
    """Atomically write the failed ``hsz_id`` list + error messages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(int(i) for i in errors)
    payload = {
        "failed_hsz_ids": ordered,
        "errors": {str(i): str(errors[i]) for i in ordered},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --- scores I/O ------------------------------------------------------------


def write_scores(rows: Sequence[Mapping[str, Any]], path: Path) -> int:
    """Merge *rows* into *path* (dedupe on ``hsz_id``, keep last) atomically.

    Existing rows are read back so a flushed run keeps every previously scored
    post. Returns the total number of rows in the rewritten file.
    """
    if path.is_file():
        existing = pd.read_parquet(path)
        frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["hsz_id"], keep="last")
    frame = frame[list(SCORES_OUT_COLUMNS)].reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(frame, schema=SCORES_OUT_SCHEMA, preserve_index=False)
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return len(frame)


# --- run -------------------------------------------------------------------


@dataclass
class BatchResult:
    """Summary of one batch-scoring run."""

    scored: int
    skipped: int
    failed: int
    aborted: bool
    abort_reason: str | None
    estimated_cost_usd: float
    input_tokens: int
    output_tokens: int
    elapsed_sec: float


def run_batch(
    posts: Sequence[Post],
    *,
    config: Config,
    paths: BatchPaths,
    score_fn: ScoreFn | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    session_factory: Callable[[], requests.Session] | None = None,
    max_in_flight: int | None = None,
) -> BatchResult:
    """Score *posts* concurrently, resuming and checkpointing as configured.

    *score_fn* (``text -> (answer, usage)``) defaults to Jev via
    :func:`pressflow.score_jev.score_post_with_usage`; tests inject a fake to
    exercise the loop without network access. *max_in_flight* bounds the number
    of queued futures (defaults to ``concurrency * IN_FLIGHT_PER_WORKER``).
    """
    scoring = config.scoring
    done = load_completed_hsz_ids(paths.scores, paths.checkpoint)
    pending = [post for post in posts if post.hsz_id not in done]
    skipped = len(posts) - len(pending)

    errors = load_failed_errors(paths.failed)
    completed_ids: set[int] = set(done)
    new_rows: list[dict[str, Any]] = []
    tracker = ConsecutiveFailureTracker(scoring.max_consecutive_failures)
    cost = CostTracker()
    start = clock()
    scored = 0
    aborted = False
    abort_reason: str | None = None

    if not pending:
        save_checkpoint(paths.checkpoint, completed_ids)
        save_failed(paths.failed, errors)
        logger.info(
            "nothing to do: %d post(s) already scored, %d pending",
            skipped,
            len(pending),
        )
        return BatchResult(
            scored=0,
            skipped=skipped,
            failed=len(errors),
            aborted=False,
            abort_reason=None,
            estimated_cost_usd=cost.estimated_cost_usd,
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
            elapsed_sec=clock() - start,
        )

    def flush() -> None:
        if new_rows:
            write_scores(new_rows, paths.scores)
            new_rows.clear()
        save_checkpoint(paths.checkpoint, completed_ids)
        save_failed(paths.failed, errors)

    def log_progress() -> None:
        elapsed = clock() - start
        rate = scored / elapsed if elapsed > 0 else 0.0
        remaining = len(pending) - scored
        eta = remaining / rate if rate > 0 else float("inf")
        logger.info(
            "scored=%d/%d failed=%d consecutive_failures=%d/%d "
            "cost=$%.4f throughput=%.1f posts/s eta=%.0fs",
            scored,
            len(pending),
            tracker.total_failures,
            tracker.consecutive,
            scoring.max_consecutive_failures,
            cost.estimated_cost_usd,
            rate,
            eta,
        )

    # One requests.Session per worker thread (requests.Session is not safe to
    # share across threads for concurrent writes).
    thread_local = threading.local() if score_fn is None else None

    def new_session() -> requests.Session:
        assert thread_local is not None
        session = getattr(thread_local, "session", None)
        if session is None:
            session = (
                session_factory() if session_factory is not None else requests.Session()
            )
            thread_local.session = session
        return session

    def task(text: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if scoring.request_delay_sec:
            sleep(scoring.request_delay_sec)
        if score_fn is not None:
            return score_fn(text)
        return score_post_with_usage(
            text,
            session=new_session(),
            max_attempts=max(1, scoring.retries),
        )

    executor = ThreadPoolExecutor(max_workers=scoring.concurrency)
    pending_iter = iter(pending)
    in_flight: dict[Future, int] = {}

    def pump() -> None:
        window = (
            max_in_flight
            if max_in_flight is not None
            else max(scoring.concurrency * IN_FLIGHT_PER_WORKER, 1)
        )
        window = max(1, window)
        while len(in_flight) < window:
            try:
                post = next(pending_iter)
            except StopIteration:
                return
            in_flight[executor.submit(task, post.text)] = post.hsz_id

    progress_every = max(1, min(scoring.flush_every, 100))

    try:
        pump()
        while in_flight and not aborted:
            finished, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for future in finished:
                hsz_id = in_flight.pop(future)
                try:
                    answer, usage = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad post must not kill the run
                    errors[hsz_id] = f"{type(exc).__name__}: {exc}"
                    hit_max = tracker.record_failure()
                    logger.warning(
                        "FAILED hsz_id=%s (%s); consecutive_failures=%d/%d",
                        hsz_id,
                        errors[hsz_id],
                        tracker.consecutive,
                        scoring.max_consecutive_failures,
                    )
                    if hit_max:
                        aborted = True
                        abort_reason = (
                            f"{tracker.consecutive} consecutive failures "
                            f"(>= {scoring.max_consecutive_failures})"
                        )
                        logger.error("aborting after %s", abort_reason)
                        break
                    continue

                tracker.record_success()
                new_rows.append(build_score_row(hsz_id, answer, model=scoring.model))
                completed_ids.add(hsz_id)
                errors.pop(hsz_id, None)
                scored += 1
                cost.add_usage(usage)

                if scored % scoring.flush_every == 0:
                    flush()
                if scored % progress_every == 0:
                    log_progress()

                if cost.exceeds_cap(scoring.cost_cap_usd):
                    aborted = True
                    abort_reason = (
                        f"estimated cost ${cost.estimated_cost_usd:.4f} exceeds "
                        f"cap ${scoring.cost_cap_usd:.2f}"
                    )
                    logger.error("aborting: %s", abort_reason)
                    break
            pump()
    finally:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        flush()
        log_progress()

    elapsed = clock() - start
    logger.info(
        "batch done: scored=%d skipped=%d failed=%d aborted=%s cost=$%.4f "
        "in=%d out=%d elapsed=%.1fs",
        scored,
        skipped,
        len(errors),
        aborted,
        cost.estimated_cost_usd,
        cost.input_tokens,
        cost.output_tokens,
        elapsed,
    )
    return BatchResult(
        scored=scored,
        skipped=skipped,
        failed=len(errors),
        aborted=aborted,
        abort_reason=abort_reason,
        estimated_cost_usd=cost.estimated_cost_usd,
        input_tokens=cost.input_tokens,
        output_tokens=cost.output_tokens,
        elapsed_sec=elapsed,
    )


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent, resumable Jev batch scoring (T3.4)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N newest posts (smoke tests)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip already-scored posts (default; kept for explicitness)",
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="override posts.parquet path"
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
    paths = batch_paths(config)
    if args.input is not None:
        paths = BatchPaths(
            input=args.input,
            scores=paths.scores,
            checkpoint=paths.checkpoint,
            failed=paths.failed,
        )

    posts = load_posts(paths.input, limit=args.limit)
    logger.info(
        "loaded %d post(s) from %s (limit=%s); scoring backend=%s model=%s "
        "concurrency=%d",
        len(posts),
        paths.input,
        args.limit,
        config.scoring.backend,
        config.scoring.model,
        config.scoring.concurrency,
    )
    result = run_batch(posts, config=config, paths=paths)
    print(
        f"scored={result.scored} skipped={result.skipped} failed={result.failed} "
        f"aborted={result.aborted} cost=${result.estimated_cost_usd:.4f} "
        f"elapsed={result.elapsed_sec:.1f}s"
    )
    return 1 if result.aborted else 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

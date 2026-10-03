"""Tests for T3.4 — concurrent, resumable batch scoring (pure, NO network).

Covers the consecutive-failure abort rule, the checkpoint merge/skip logic, cost
estimation from ``usage`` dicts, the scoring-config parse, and the run loop
itself with an injected fake scorer.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
import requests

from pressflow.batch_score import (
    SCORES_OUT_COLUMNS,
    BatchPaths,
    ConsecutiveFailureTracker,
    CostTracker,
    Post,
    build_score_row,
    estimate_cost_usd,
    load_checkpoint_ids,
    load_completed_hsz_ids,
    run_batch,
    save_checkpoint,
    usage_tokens,
    write_scores,
)
from pressflow.config import get_config

BULL_ANSWER = {
    "choice": "bull",
    "confidence": 0.9,
    "probabilities": {"bull": 0.9, "bear": 0.05, "neutral": 0.05},
}
BEAR_ANSWER = {
    "choice": "bear",
    "confidence": 0.8,
    "probabilities": {"bull": 0.1, "bear": 0.8, "neutral": 0.1},
}


def _config(**scoring_overrides):
    base = get_config()
    return replace(base, scoring=replace(base.scoring, **scoring_overrides))


def _paths(tmp_path) -> BatchPaths:
    return BatchPaths(
        input=tmp_path / "posts.parquet",
        scores=tmp_path / "scores.parquet",
        checkpoint=tmp_path / "scores_checkpoint.json",
        failed=tmp_path / "scores_failed.json",
    )


def _ok_score_fn(answer=BULL_ANSWER):
    def fn(text: str):
        return answer, {"input_tokens": 100, "output_tokens": 10}

    return fn


# --- consecutive-failure abort rule ----------------------------------------


def test_success_resets_consecutive_counter():
    tracker = ConsecutiveFailureTracker(max_consecutive=10)
    for _ in range(5):
        assert tracker.record_failure() is False
    assert tracker.consecutive == 5

    tracker.record_success()
    assert tracker.consecutive == 0

    for _ in range(9):
        assert tracker.record_failure() is False
    assert tracker.should_abort is False
    # total failures are never reset by a success
    assert tracker.total_failures == 14


def test_ten_consecutive_failures_abort():
    tracker = ConsecutiveFailureTracker(max_consecutive=10)
    for _ in range(9):
        assert tracker.record_failure() is False
    assert tracker.record_failure() is True
    assert tracker.should_abort is True


def test_nine_consecutive_failures_do_not_abort():
    tracker = ConsecutiveFailureTracker(max_consecutive=10)
    for _ in range(9):
        assert tracker.record_failure() is False
    assert tracker.should_abort is False


# --- cost estimation -------------------------------------------------------


def test_estimate_cost_usd():
    # $0.042 per 1M tokens, input and output priced the same.
    assert estimate_cost_usd(1_000_000, 0) == pytest.approx(0.042)
    assert estimate_cost_usd(0, 1_000_000) == pytest.approx(0.042)
    assert estimate_cost_usd(500_000, 500_000) == pytest.approx(0.042)
    assert estimate_cost_usd(0, 0) == 0.0


def test_usage_tokens_reads_typesafe_keys():
    assert usage_tokens({"input_tokens": 377, "output_tokens": 39}) == (377, 39)


def test_usage_tokens_accepts_aliases_and_bad_values():
    assert usage_tokens({"prompt_tokens": "7", "completion_tokens": None}) == (7, 0)
    assert usage_tokens({"input_tokens": 3}) == (3, 0)
    assert usage_tokens(None) == (0, 0)
    assert usage_tokens({"input_tokens": "n/a", "output_tokens": 2}) == (0, 2)


def test_cost_tracker_accumulates_and_caps():
    tracker = CostTracker()
    tracker.add_usage({"input_tokens": 1_000_000, "output_tokens": 0})
    tracker.add_usage({"input_tokens": 0, "output_tokens": 1_000_000})
    assert tracker.input_tokens == 1_000_000
    assert tracker.output_tokens == 1_000_000
    assert tracker.estimated_cost_usd == pytest.approx(0.084)
    assert tracker.exceeds_cap(0.10) is False
    assert tracker.exceeds_cap(0.05) is True


# --- checkpoint merge / skip logic -----------------------------------------


def test_checkpoint_roundtrip_sorts_ids(tmp_path):
    path = tmp_path / "scores_checkpoint.json"
    save_checkpoint(path, {3, 1, 2})
    assert load_checkpoint_ids(path) == {1, 2, 3}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["completed_hsz_ids"] == [1, 2, 3]


def test_load_checkpoint_missing_or_corrupt_is_empty(tmp_path):
    assert load_checkpoint_ids(tmp_path / "nope.json") == set()
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not json", encoding="utf-8")
    assert load_checkpoint_ids(corrupt) == set()


def test_load_completed_merges_scores_parquet_and_checkpoint(tmp_path):
    paths = _paths(tmp_path)
    write_scores(
        [build_score_row(1, BULL_ANSWER), build_score_row(2, BULL_ANSWER)],
        paths.scores,
    )
    save_checkpoint(paths.checkpoint, {3})
    assert load_completed_hsz_ids(paths.scores, paths.checkpoint) == {1, 2, 3}


def test_load_completed_missing_files_is_empty(tmp_path):
    paths = _paths(tmp_path)
    assert load_completed_hsz_ids(paths.scores, paths.checkpoint) == set()


def test_write_scores_dedupes_keep_last(tmp_path):
    paths = _paths(tmp_path)
    write_scores([build_score_row(1, BULL_ANSWER)], paths.scores)
    write_scores(
        [build_score_row(1, BEAR_ANSWER), build_score_row(2, BULL_ANSWER)],
        paths.scores,
    )
    frame = pd.read_parquet(paths.scores)
    assert list(frame.columns) == list(SCORES_OUT_COLUMNS)
    assert list(frame["hsz_id"]) == [1, 2]
    assert frame.loc[frame["hsz_id"] == 1, "raw_label"].iloc[0] == "bear"


# --- scoring config --------------------------------------------------------


def test_scoring_config_defaults():
    scoring = get_config().scoring
    assert scoring.backend == "jev"
    assert scoring.model == "jev-latest"
    assert scoring.concurrency == 16
    assert scoring.max_consecutive_failures == 10
    assert scoring.cost_cap_usd == 9.0
    assert scoring.request_delay_sec == 0.2
    assert scoring.retries == 3
    assert scoring.flush_every == 500


# --- run loop --------------------------------------------------------------


def _posts(n: int) -> list[Post]:
    return [Post(i, f"post-{i}") for i in range(1, n + 1)]


def test_run_batch_scores_writes_and_resumes(tmp_path):
    paths = _paths(tmp_path)
    cfg = _config(concurrency=2, request_delay_sec=0.0, flush_every=2, retries=1)
    calls: list[str] = []

    def fn(text: str):
        calls.append(text)
        return BULL_ANSWER, {"input_tokens": 100, "output_tokens": 10}

    result = run_batch(
        _posts(3), config=cfg, paths=paths, score_fn=fn, sleep=lambda _s: None
    )
    assert (result.scored, result.skipped, result.failed) == (3, 0, 0)
    assert not result.aborted
    assert result.input_tokens == 300 and result.output_tokens == 30
    assert result.estimated_cost_usd == pytest.approx(330 / 1_000_000 * 0.042)
    assert set(pd.read_parquet(paths.scores)["hsz_id"]) == {1, 2, 3}
    assert load_checkpoint_ids(paths.checkpoint) == {1, 2, 3}

    # Re-running resumes: nothing is pending, no extra calls / spend.
    calls.clear()
    resumed = run_batch(
        _posts(3), config=cfg, paths=paths, score_fn=fn, sleep=lambda _s: None
    )
    assert (resumed.scored, resumed.skipped) == (0, 3)
    assert calls == []


def test_run_batch_aborts_after_ten_consecutive_failures(tmp_path):
    paths = _paths(tmp_path)
    cfg = _config(
        concurrency=1,
        request_delay_sec=0.0,
        flush_every=100,
        retries=1,
        max_consecutive_failures=10,
    )

    def fn(text: str):
        raise requests.RequestException("boom")

    result = run_batch(
        _posts(30), config=cfg, paths=paths, score_fn=fn, sleep=lambda _s: None
    )
    assert result.aborted is True
    assert result.scored == 0
    assert result.failed == 10
    assert "consecutive failures" in (result.abort_reason or "")
    failed = json.loads(paths.failed.read_text(encoding="utf-8"))
    assert len(failed["failed_hsz_ids"]) == 10


def test_run_batch_success_resets_consecutive_failures(tmp_path):
    paths = _paths(tmp_path)
    cfg = _config(
        concurrency=1,
        request_delay_sec=0.0,
        flush_every=1000,
        retries=1,
        max_consecutive_failures=10,
    )

    def fn(text: str):
        number = int(text.rsplit("-", 1)[1])
        if number in (10, 20):
            return BULL_ANSWER, {"input_tokens": 1, "output_tokens": 1}
        raise requests.RequestException("boom")

    result = run_batch(
        _posts(24),
        config=cfg,
        paths=paths,
        score_fn=fn,
        sleep=lambda _s: None,
        max_in_flight=1,
    )
    # 22 failures across the run, but never 10 in a row → no abort.
    assert result.aborted is False
    assert result.scored == 2
    assert result.failed == 22


def test_run_batch_aborts_when_cost_cap_exceeded(tmp_path):
    paths = _paths(tmp_path)
    cfg = _config(
        concurrency=1,
        request_delay_sec=0.0,
        flush_every=1000,
        retries=1,
        cost_cap_usd=0.0002,
    )

    def fn(text: str):
        # Each post costs 1M + 0 tokens = $0.042, immediately over the cap.
        return BULL_ANSWER, {"input_tokens": 1_000_000, "output_tokens": 0}

    result = run_batch(
        _posts(10), config=cfg, paths=paths, score_fn=fn, sleep=lambda _s: None
    )
    assert result.aborted is True
    assert "cost" in (result.abort_reason or "")
    assert result.estimated_cost_usd > cfg.scoring.cost_cap_usd

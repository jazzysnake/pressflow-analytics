"""Tests for T3.6 — Jev score mapping and adopt rule (pure, NO network)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pressflow.eval_jev import (
    ADOPT_MARGIN,
    HTS5_DIRECTIONAL_MACRO_F1,
    NEUTRAL_BASELINE_ACC,
    JevReport,
    decide,
)
from pressflow.score_jev import (
    BEAR,
    BULL,
    JEV_MODEL,
    NEUTRAL,
    build_request_body,
    load_secrets,
    parse_answer,
    score_rows_for_answers,
)


# --- parse_answer ----------------------------------------------------------


def test_parse_answer_clear_bull():
    answer = {
        "type": "choice",
        "choice": "bull",
        "confidence": 1.0,
        "probabilities": {"bull": 1.0, "neutral": 0.0, "bear": 0.0},
    }
    score = parse_answer(answer)
    assert score.raw_label == BULL
    assert score.confidence == pytest.approx(1.0)
    assert score.p_bull == pytest.approx(1.0)
    assert score.p_bear == pytest.approx(0.0)
    assert score.p_neutral == pytest.approx(0.0)
    assert score.net == pytest.approx(1.0)


def test_parse_answer_mixed_probabilities_net():
    answer = {
        "choice": "bull",
        "confidence": 0.5,
        "probabilities": {"bull": 0.5, "bear": 0.3, "neutral": 0.2},
    }
    score = parse_answer(answer)
    assert score.p_bull == pytest.approx(0.5)
    assert score.p_bear == pytest.approx(0.3)
    assert score.p_neutral == pytest.approx(0.2)
    assert score.net == pytest.approx(0.2)
    assert score.p_bull + score.p_bear + score.p_neutral == pytest.approx(1.0)


def test_parse_answer_renormalizes_non_unit_mass():
    answer = {
        "choice": "bull",
        "confidence": 0.5,
        "probabilities": {"bull": 2.0, "bear": 1.0, "neutral": 1.0},
    }
    score = parse_answer(answer)
    assert score.p_bull == pytest.approx(0.5)
    assert score.p_bear == pytest.approx(0.25)
    assert score.p_neutral == pytest.approx(0.25)
    assert score.net == pytest.approx(0.25)


def test_parse_answer_missing_probability_keys_default_zero():
    answer = {"choice": "bear", "confidence": 0.9, "probabilities": {"bear": 1.0}}
    score = parse_answer(answer)
    assert score.p_bull == pytest.approx(0.0)
    assert score.p_bear == pytest.approx(1.0)
    assert score.p_neutral == pytest.approx(0.0)
    assert score.net == pytest.approx(-1.0)


def test_parse_answer_falls_back_to_argmax_when_choice_missing():
    answer = {
        "confidence": 0.7,
        "probabilities": {"bull": 0.1, "bear": 0.2, "neutral": 0.7},
    }
    score = parse_answer(answer)
    assert score.raw_label == NEUTRAL


def test_parse_answer_confidence_missing_falls_back_to_max_prob():
    answer = {
        "choice": "bull",
        "probabilities": {"bull": 0.8, "bear": 0.1, "neutral": 0.1},
    }
    score = parse_answer(answer)
    assert score.confidence == pytest.approx(0.8)


def test_parse_answer_net_always_in_range_and_probs_sum_to_one():
    for probs in (
        {"bull": 0.1, "bear": 0.8, "neutral": 0.1},
        {"bull": 0.34, "bear": 0.33, "neutral": 0.33},
        {"bull": 0.0, "bear": 0.0, "neutral": 1.0},
    ):
        score = parse_answer({"choice": "neutral", "probabilities": probs})
        assert -1.0 <= score.net <= 1.0
        assert score.p_bull + score.p_bear + score.p_neutral == pytest.approx(1.0)


def test_parse_answer_zero_mass_raises():
    with pytest.raises(ValueError, match="non-positive probability mass"):
        parse_answer(
            {"choice": "bull", "probabilities": {"bull": 0.0, "bear": 0.0, "neutral": 0.0}}
        )


def test_parse_answer_missing_probabilities_raises():
    with pytest.raises(ValueError, match="non-positive probability mass"):
        parse_answer({"choice": "bull", "confidence": 1.0})


def test_parse_answer_non_mapping_raises():
    with pytest.raises(ValueError, match="expected an answer mapping"):
        parse_answer(None)  # type: ignore[arg-type]


# --- build_request_body ----------------------------------------------------


def test_build_request_body_shape():
    body = build_request_body("some hungarian text")
    assert body["state"] == "some hungarian text"
    assert body["model"] == JEV_MODEL
    question = body["questions"]["sentiment"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {BULL, BEAR, NEUTRAL}


# --- score_rows_for_answers ------------------------------------------------


def test_score_rows_for_answers_builds_columns():
    posts = [
        {"hsz_id": 1, "post_id": 10},
        {"hsz_id": 2, "post_id": 20},
    ]
    answers = [
        {"choice": "bull", "confidence": 0.9, "probabilities": {"bull": 0.9, "bear": 0.05, "neutral": 0.05}},
        {"choice": "bear", "confidence": 0.8, "probabilities": {"bull": 0.05, "bear": 0.8, "neutral": 0.15}},
    ]
    rows = score_rows_for_answers(posts, answers)
    assert len(rows) == 2
    assert rows[0]["hsz_id"] == 1 and rows[0]["post_id"] == 10
    assert rows[0]["model_id"] == JEV_MODEL
    assert rows[0]["raw_label"] == BULL and rows[0]["net"] > 0
    assert rows[1]["raw_label"] == BEAR and rows[1]["net"] < 0
    for row in rows:
        assert row["p_bull"] + row["p_bear"] + row["p_neutral"] == pytest.approx(1.0)


def test_score_rows_for_answers_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        score_rows_for_answers([{"hsz_id": 1, "post_id": 1}], [])


# --- load_secrets ----------------------------------------------------------


def test_load_secrets_splits_on_first_equals(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_KEY", raising=False)
    secrets_file = tmp_path / ".env.secrets"
    # A base64-ish value containing '=' padding must be preserved intact.
    secrets_file.write_text(
        "# comment\nOTHER_KEY=abc=def==\nTYPESAFE_API_KEY=sk-test-123==\n",
        encoding="utf-8",
    )
    load_secrets(secrets_file)
    assert os.environ["TYPESAFE_API_KEY"] == "sk-test-123=="
    assert os.environ["OTHER_KEY"] == "abc=def=="


def test_load_secrets_does_not_override_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "already-set")
    secrets_file = tmp_path / ".env.secrets"
    secrets_file.write_text("TYPESAFE_API_KEY=from-file\n", encoding="utf-8")
    load_secrets(secrets_file)
    assert os.environ["TYPESAFE_API_KEY"] == "already-set"


def test_load_secrets_missing_file_returns_zero(tmp_path):
    assert load_secrets(tmp_path / "nope.env") == 0


# --- decide ----------------------------------------------------------------


def _report(acc, macro, dirf1):
    return JevReport(
        model_id="jev-latest",
        n=201,
        accuracy=acc,
        macro_f1=macro,
        directional_macro_f1=dirf1,
        confusion=np.zeros((3, 3), dtype=int),
    )


def test_decide_adopts_when_both_conditions_hold():
    report = _report(
        acc=NEUTRAL_BASELINE_ACC + 0.05,
        macro=0.5,
        dirf1=HTS5_DIRECTIONAL_MACRO_F1 + ADOPT_MARGIN + 0.01,
    )
    assert decide(report).adopt is True


def test_decide_rejects_when_accuracy_below_neutral_baseline():
    report = _report(
        acc=NEUTRAL_BASELINE_ACC - 0.10,
        macro=0.5,
        dirf1=HTS5_DIRECTIONAL_MACRO_F1 + ADOPT_MARGIN + 0.01,
    )
    assert decide(report).adopt is False


def test_decide_rejects_when_directional_f1_not_clearly_better():
    # beats 0.230 but not by the required margin
    report = _report(
        acc=NEUTRAL_BASELINE_ACC + 0.05,
        macro=0.5,
        dirf1=HTS5_DIRECTIONAL_MACRO_F1 + 0.01,
    )
    assert decide(report).adopt is False


def test_decide_reason_mentions_both_conditions():
    report = _report(acc=0.5, macro=0.4, dirf1=0.3)
    decision = decide(report)
    assert "directional macro-F1" in decision.reason
    assert "accuracy" in decision.reason

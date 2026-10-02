"""Tests for T3.3 — model selection metrics (pure, no model download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressflow.select_model import (
    BEAR,
    BULL,
    NEUTRAL,
    ModelReport,
    choose_model,
    confusion_matrix,
    directional_macro_f1,
    expected_calibration_error,
    evaluate_model,
    macro_f1,
    predicted_classes,
    render_report,
    sarcasm_panic_errors,
)


def _scored(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal scored frame (probs + consensus label + text)."""
    frame = pd.DataFrame(rows)
    frame["text"] = frame.get("text", [""] * len(frame))
    return frame


# --- predicted_classes / confusion -----------------------------------------


def test_predicted_classes_argmax():
    frame = _scored(
        [
            {"p_bull": 0.7, "p_bear": 0.2, "p_neutral": 0.1, "label": BULL},
            {"p_bull": 0.1, "p_bear": 0.6, "p_neutral": 0.3, "label": BEAR},
            {"p_bull": 0.2, "p_bear": 0.1, "p_neutral": 0.7, "label": NEUTRAL},
        ]
    )
    assert list(predicted_classes(frame)) == [BULL, BEAR, NEUTRAL]


def test_confusion_matrix_rows_true_cols_pred():
    y_true = [BULL, BEAR, NEUTRAL]
    y_pred = [BULL, NEUTRAL, BEAR]
    cm = confusion_matrix(y_true, y_pred)
    # order of CLASSES is (bull, bear, neutral)
    assert cm.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]


# --- F1 --------------------------------------------------------------------


def test_macro_f1_known_example():
    y_true = [BULL, BULL, BEAR, BEAR, NEUTRAL, NEUTRAL]
    y_pred = [BULL, BULL, BEAR, NEUTRAL, NEUTRAL, BEAR]
    assert macro_f1(y_true, y_pred) == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert directional_macro_f1(y_true, y_pred) == pytest.approx((1.0 + 0.5) / 2)


def test_macro_f1_perfect_is_one():
    labels = [BULL, BEAR, NEUTRAL] * 3
    assert macro_f1(labels, labels) == pytest.approx(1.0)
    assert directional_macro_f1(labels, labels) == pytest.approx(1.0)


# --- calibration -----------------------------------------------------------


def test_ece_when_always_confident_but_wrong():
    conf = np.full(10, 0.8)
    correct = np.zeros(10, dtype=bool)
    ece, rows = expected_calibration_error(conf, correct)
    assert ece == pytest.approx(0.8)
    non_empty = [r for r in rows if r[2] > 0]
    assert len(non_empty) == 1 and non_empty[0][2] == 10


def test_ece_when_perfectly_calibrated():
    # 10 posts at conf 1.0 all correct -> ECE 0
    ece, _ = expected_calibration_error(np.full(10, 1.0), np.ones(10, dtype=bool))
    assert ece == pytest.approx(0.0)


# --- evaluate_model --------------------------------------------------------


def test_evaluate_model_fields():
    frame = _scored(
        [
            {"p_bull": 0.9, "p_bear": 0.05, "p_neutral": 0.05, "label": BULL},
            {"p_bull": 0.05, "p_bear": 0.9, "p_neutral": 0.05, "label": BEAR},
            {"p_bull": 0.05, "p_bear": 0.05, "p_neutral": 0.9, "label": NEUTRAL},
        ]
    )
    report = evaluate_model("m", frame)
    assert report.accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.confusion.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# --- sarcasm / panic errors ------------------------------------------------


def test_sarcasm_panic_errors_only_flagged_and_wrong():
    merged = pd.DataFrame(
        {
            "model_id": ["m", "m", "m"],
            "hsz_id": [1, 2, 3],
            "label": [BULL, BEAR, NEUTRAL],
            "net": [0.5, -0.5, 0.0],
            "sarcasm": ["true", "", ""],
            "panic": ["", "true", "true"],
            "text": ["a", "b", "c"],
        }
    )
    pred = pd.Series([BEAR, BEAR, BULL])  # #1 wrong+flagged, #2 right, #3 wrong+flagged
    errors = sarcasm_panic_errors(merged, pred)
    assert list(errors["hsz_id"]) == [1, 3]


# --- choose_model ----------------------------------------------------------


def _report(model_id, dir_f1, dir_ece, acc=0.5, macro=0.5):
    return ModelReport(
        model_id=model_id,
        n=10,
        accuracy=acc,
        macro_f1=macro,
        directional_macro_f1=dir_f1,
        ece=dir_ece,
        directional_ece=dir_ece,
        confusion=np.zeros((3, 3), dtype=int),
        reliability_rows=[],
        directional_rows=[],
    )


def test_choose_model_prefers_directional_f1_among_calibrated():
    # both within the calibration margin -> higher directional macro-F1 wins
    reports = [_report("strong", 0.80, 0.20), _report("weak", 0.50, 0.10)]
    assert choose_model(reports)[0] == "strong"


def test_choose_model_sets_aside_overconfident_model():
    # "sharp" has much higher dir-F1 but is far too over-confident -> excluded
    reports = [_report("sharp", 0.90, 0.70), _report("calm", 0.40, 0.10)]
    assert choose_model(reports)[0] == "calm"


def test_choose_model_breaks_equal_f1_tie_by_lower_ece():
    reports = [_report("a", 0.80, 0.20), _report("b", 0.80, 0.12)]
    assert choose_model(reports)[0] == "b"


def test_choose_model_empty_raises():
    with pytest.raises(ValueError, match="no model reports"):
        choose_model([])


# --- render_report ---------------------------------------------------------


def test_render_report_contains_chosen_and_tables():
    reports = [_report("m1", 0.8, 0.1), _report("m2", 0.5, 0.2)]
    merged = pd.DataFrame(
        {
            "model_id": ["m1"],
            "hsz_id": [1],
            "label": [BULL],
            "net": [0.5],
            "p_bull": [0.6],
            "p_bear": [0.1],
            "p_neutral": [0.3],
            "sarcasm": ["true"],
            "panic": [""],
            "text": ["some text"],
        }
    )
    text = render_report(reports, merged, "m1", "because reasons")
    assert "# Sentiment model selection" in text
    assert "m1" in text and "m2" in text
    assert "because reasons" in text
    assert "## Confusion matrices" in text
    assert "## Chosen model" in text

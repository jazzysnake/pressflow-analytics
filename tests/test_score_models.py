"""Tests for T3.2 — candidate model runner (pure label/normalisation logic).

No model is downloaded: all pipeline outputs are synthetic.
"""

from __future__ import annotations

import pytest

from pressflow.score_models import (
    BEAR,
    BULL,
    NEUTRAL,
    GENERIC_LABEL_GROUPS,
    Probs,
    group_from_label_name,
    label_group_map,
    normalize_distribution,
    score_rows_for_texts,
    top_prediction,
)

HTS5_ID = "NYTK/sentiment-hts5-xlm-roberta-hungarian"
OHB3_ID = "NYTK/sentiment-ohb3-xlm-roberta-hungarian"
CARDIFF_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

HTS5_LABELS = {i: f"LABEL_{i}" for i in range(5)}
OHB3_LABELS = {i: f"LABEL_{i}" for i in range(3)}
CARDIFF_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


# --- group_from_label_name -------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("positive", BULL),
        ("very positive", BULL),
        ("POSITIVE", BULL),
        ("negative", BEAR),
        ("very negative", BEAR),
        ("Very-Negative", BEAR),
        ("neutral", NEUTRAL),
        ("LABEL_0", None),
        ("LABEL_4", None),
        ("", None),
        ("star_3", None),
    ],
)
def test_group_from_label_name(name, expected):
    assert group_from_label_name(name) == expected


# --- label_group_map -------------------------------------------------------


def test_label_group_map_cardiff_semantic_names():
    mapping = label_group_map(CARDIFF_LABELS, CARDIFF_ID)
    assert mapping == {"negative": BEAR, "neutral": NEUTRAL, "positive": BULL}


def test_label_group_map_hts5_uses_documented_order():
    assert label_group_map(HTS5_LABELS, HTS5_ID) == {
        "LABEL_0": BEAR,
        "LABEL_1": BEAR,
        "LABEL_2": NEUTRAL,
        "LABEL_3": BULL,
        "LABEL_4": BULL,
    }


def test_label_group_map_ohb3_uses_documented_order():
    assert label_group_map(OHB3_LABELS, OHB3_ID) == {
        "LABEL_0": BEAR,
        "LABEL_1": NEUTRAL,
        "LABEL_2": BULL,
    }


def test_generic_schemes_have_expected_lengths():
    # Guard so a model-card change can't silently shift the mapping.
    assert len(GENERIC_LABEL_GROUPS[HTS5_ID]) == 5
    assert len(GENERIC_LABEL_GROUPS[OHB3_ID]) == 3


def test_label_group_map_unknown_generic_raises():
    with pytest.raises(ValueError, match="cannot determine label semantics"):
        label_group_map({0: "LABEL_0", 1: "LABEL_1"}, "some/unknown-model")


# --- normalize_distribution ------------------------------------------------


def _dist(labels, scores):
    return [{"label": lab, "score": sc} for lab, sc in zip(labels, scores)]


def _assert_probs(got: Probs, p_bull, p_bear, p_neutral, net):
    assert got.p_bull == pytest.approx(p_bull)
    assert got.p_bear == pytest.approx(p_bear)
    assert got.p_neutral == pytest.approx(p_neutral)
    assert got.net == pytest.approx(net)


def test_normalize_hts5_groups_and_renormalizes():
    # very-negative .1, negative .2 -> bear .3 ; neutral .2 ; pos .3, very-pos .2
    # -> bull .5 ; sum == 1
    scores = _dist(
        ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3", "LABEL_4"],
        [0.1, 0.2, 0.2, 0.3, 0.2],
    )
    got = normalize_distribution(scores, label_group_map(HTS5_LABELS, HTS5_ID))
    _assert_probs(got, 0.5, 0.3, 0.2, 0.2)


def test_normalize_renormalizes_non_unit_mass():
    scores = _dist(CARDIFF_LABELS.values(), [0.0, 3.0, 1.0])
    got = normalize_distribution(scores, label_group_map(CARDIFF_LABELS, CARDIFF_ID))
    _assert_probs(got, 0.25, 0.0, 0.75, 0.25)


def test_normalize_probs_sum_to_one_and_net_in_range():
    scores = _dist(CARDIFF_LABELS.values(), [0.7, 0.1, 0.2])
    got = normalize_distribution(scores, label_group_map(CARDIFF_LABELS, CARDIFF_ID))
    assert got.p_bull + got.p_bear + got.p_neutral == pytest.approx(1.0)
    assert -1.0 <= got.net <= 1.0


def test_normalize_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        normalize_distribution([], label_group_map(CARDIFF_LABELS, CARDIFF_ID))


def test_normalize_unknown_label_raises():
    with pytest.raises(KeyError, match="unknown native label"):
        normalize_distribution(
            [{"label": "weird", "score": 1.0}],
            label_group_map(CARDIFF_LABELS, CARDIFF_ID),
        )


def test_normalize_zero_mass_raises():
    scores = _dist(CARDIFF_LABELS.values(), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-positive probability mass"):
        normalize_distribution(scores, label_group_map(CARDIFF_LABELS, CARDIFF_ID))


# --- top_prediction --------------------------------------------------------


def test_top_prediction_picks_max():
    scores = _dist(CARDIFF_LABELS.values(), [0.1, 0.7, 0.2])
    assert top_prediction(scores) == ("neutral", 0.7)


# --- score_rows_for_texts --------------------------------------------------


def test_score_rows_for_texts_multiple_texts():
    # Pipeline output for two posts (top_k=None -> list of lists).
    outputs = [
        _dist(CARDIFF_LABELS.values(), [0.9, 0.05, 0.05]),  # negative
        _dist(CARDIFF_LABELS.values(), [0.05, 0.05, 0.9]),  # positive
    ]
    rows = score_rows_for_texts(
        ["bearish text", "bullish text"], outputs, CARDIFF_LABELS, CARDIFF_ID
    )
    assert len(rows) == 2
    assert rows[0]["raw_label"] == "negative"
    assert rows[0]["p_bear"] == pytest.approx(0.9)
    assert rows[0]["net"] < 0
    assert rows[1]["raw_label"] == "positive"
    assert rows[1]["net"] > 0
    for row in rows:
        assert row["p_bull"] + row["p_bear"] + row["p_neutral"] == pytest.approx(1.0)


def test_score_rows_for_texts_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        score_rows_for_texts(
            ["a"],
            [],
            CARDIFF_LABELS,
            CARDIFF_ID,
        )

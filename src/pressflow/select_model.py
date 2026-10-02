"""T3.3 — sentiment model selection (ADR-0001, ADR-0003).

Compares the candidate sentiment models scored by T3.2
(``data/processed/eval_scores.parquet``) against the T3.1 consensus labels
(``eval/eval_set.csv``) and writes ``docs/reports/model-selection.md``:

* accuracy, macro-F1 (bull/bear/neutral) and the confusion matrix per model;
* misclassified examples flagged ``sarcasm`` or ``panic``;
* a calibration check — binned reliability / ECE of the directional call
  (aggregation uses probabilities, so calibration matters, ADR-0003).

The chosen model is picked with an explicit rule
(:func:`choose_model`) and recorded as ``models.default`` in the config.

CLI::

    uv run python -m pressflow.select_model
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.score_models import (
    BEAR,
    BULL,
    NEUTRAL,
    EVAL_SET_CSV,
    eval_scores_path,
    load_eval_set,
)

logger = logging.getLogger("pressflow.select_model")

#: Class order used everywhere in the report.
CLASSES: tuple[str, ...] = (BULL, BEAR, NEUTRAL)

#: Report location.
REPORT_PATH = REPO_ROOT / "docs" / "reports" / "model-selection.md"

#: Number of bins for the calibration curves.
ECE_BINS = 10

#: A model whose directional ECE is more than this above the best-calibrated
#: candidate is treated as too over-confident for probability aggregation
#: (ADR-0003).  Among the remaining models the strongest directional macro-F1
#: wins, so calibration gates the choice and discrimination breaks the tie.
CALIBRATION_MARGIN = 0.15

#: Human-readable label mapping per known candidate (regenerated into the doc).
LABEL_MAPPING_NOTES: dict[str, str] = {
    "NYTK/sentiment-hts5-xlm-roberta-hungarian": (
        "5-class (0 very-neg, 1 neg, 2 neutral, 3 pos, 4 very-pos) → "
        "{0,1}→bear, {2}→neutral, {3,4}→bull"
    ),
    "NYTK/sentiment-ohb3-xlm-roberta-hungarian": (
        "3-class (0 neg, 1 neutral, 2 pos) → {0}→bear, {1}→neutral, {2}→bull"
    ),
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": (
        "3-class named (negative, neutral, positive) → bear, neutral, bull"
    ),
}

__all__ = [
    "CLASSES",
    "REPORT_PATH",
    "ModelReport",
    "predicted_classes",
    "confusion_matrix",
    "macro_f1",
    "expected_calibration_error",
    "directional_reliability",
    "evaluate_model",
    "choose_model",
    "render_report",
    "write_report",
    "build_report",
    "main",
]


# --- metrics ---------------------------------------------------------------


def predicted_classes(frame: pd.DataFrame) -> np.ndarray:
    """Hard class = argmax over ``p_bull / p_bear / p_neutral``."""
    probs = frame[["p_bull", "p_bear", "p_neutral"]].to_numpy(dtype=float)
    return np.array(CLASSES, dtype=object)[probs.argmax(axis=1)]


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    """Confusion matrix with rows = true, cols = predicted, order ``CLASSES``."""
    index = {label: i for i, label in enumerate(CLASSES)}
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[index[true], index[pred]] += 1
    return cm


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-averaged F1 over ``CLASSES`` (zero-division → 0, like sklearn)."""
    cm = confusion_matrix(y_true, y_pred)
    f1s: list[float] = []
    for i in range(len(CLASSES)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1s.append(f1)
    return float(np.mean(f1s))


def directional_macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-F1 restricted to the directional classes (bull, bear)."""
    cm = confusion_matrix(y_true, y_pred)
    index = {label: i for i, label in enumerate(CLASSES)}
    f1s: list[float] = []
    for label in (BULL, BEAR):
        i = index[label]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1s.append(f1)
    return float(np.mean(f1s))


def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = ECE_BINS,
) -> tuple[float, list[tuple[float, float, int, float | None, float | None]]]:
    """Return ``(ece, rows)`` for the confidence→accuracy reliability curve.

    ``rows`` are ``(lo, hi, n, mean_confidence, accuracy)`` per bin; empty bins
    carry ``None`` for the two means.  The first bin is left-closed at 0.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = _bin_edges(n_bins)
    ece = 0.0
    rows: list[tuple[float, float, int, float | None, float | None]] = []
    n_total = len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        mean_conf = float(confidence[mask].mean())
        acc = float(correct[mask].mean())
        ece += (n / n_total) * abs(acc - mean_conf)
        rows.append((lo, hi, n, mean_conf, acc))
    return ece, rows


def directional_reliability(
    frame: pd.DataFrame, *, n_bins: int = ECE_BINS
) -> tuple[float, list[tuple[float, float, int, float | None, float | None]]]:
    """Calibration of the *directional* call (bull/bear) vs its confidence.

    Only posts the model calls bullish or bearish are considered; confidence is
    the probability of that directional class and "correct" means the consensus
    label matches it.  Returns ``(ece, rows)`` in the same format as
    :func:`expected_calibration_error`.
    """
    pred = predicted_classes(frame)
    probs = frame[["p_bull", "p_bear", "p_neutral"]].to_numpy(dtype=float)
    conf = probs.max(axis=1)
    directional = np.isin(pred, [BULL, BEAR])
    correct = np.asarray(frame["label"], dtype=object)[directional] == pred[directional]
    return expected_calibration_error(
        conf[directional], correct, n_bins=n_bins
    )


# --- evaluation ------------------------------------------------------------


@dataclass(frozen=True)
class ModelReport:
    model_id: str
    n: int
    accuracy: float
    macro_f1: float
    directional_macro_f1: float
    ece: float
    directional_ece: float
    confusion: np.ndarray
    reliability_rows: list[tuple[float, float, int, float | None, float | None]]
    directional_rows: list[tuple[float, float, int, float | None, float | None]]


def evaluate_model(model_id: str, scored: pd.DataFrame) -> ModelReport:
    """Build a :class:`ModelReport` for one model's merged predictions."""
    y_true = list(scored["label"])
    pred = list(predicted_classes(scored))
    probs = scored[["p_bull", "p_bear", "p_neutral"]].to_numpy(dtype=float)
    confidence = probs.max(axis=1)
    correct = np.asarray(y_true, dtype=object) == np.asarray(pred, dtype=object)
    ece, rows = expected_calibration_error(confidence, correct)
    dir_ece, dir_rows = directional_reliability(scored)
    return ModelReport(
        model_id=model_id,
        n=len(scored),
        accuracy=float(correct.mean()),
        macro_f1=macro_f1(y_true, pred),
        directional_macro_f1=directional_macro_f1(y_true, pred),
        ece=ece,
        directional_ece=dir_ece,
        confusion=confusion_matrix(y_true, pred),
        reliability_rows=rows,
        directional_rows=dir_rows,
    )


def choose_model(reports: Sequence[ModelReport]) -> tuple[str, str]:
    """Pick the preferred model and return ``(model_id, reasoning)``.

    Calibration gates the choice (ADR-0003: aggregation consumes
    probabilities): any candidate whose directional ECE is more than
    :data:`CALIBRATION_MARGIN` above the best-calibrated model is set aside as
    too over-confident.  Among the survivors the model with the strongest
    directional macro-F1 (the signal-bearing classes) wins; ties break on
    lower directional ECE.
    """
    if not reports:
        raise ValueError("no model reports to choose from")
    best_ece = min(r.directional_ece for r in reports)
    eligible = [
        r for r in reports if r.directional_ece <= best_ece + CALIBRATION_MARGIN
    ]
    winner = max(
        eligible, key=lambda r: (r.directional_macro_f1, -r.directional_ece)
    )
    reason = (
        f"best directional ECE is {best_ece:.3f}; "
        f"{len(eligible)}/{len(reports)} model(s) within "
        f"{CALIBRATION_MARGIN:.2f} of it; among those the highest directional "
        f"macro-F1 is {winner.directional_macro_f1:.3f} ({winner.model_id})."
    )
    return winner.model_id, reason


def sarcasm_panic_errors(merged: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Misclassified posts flagged ``sarcasm`` or ``panic`` (all models)."""
    flags = merged.get("sarcasm", pd.Series([None] * len(merged))).fillna("")
    panic = merged.get("panic", pd.Series([None] * len(merged))).fillna("")
    flagged = (flags.astype(str).str.lower() == "true") | (
        panic.astype(str).str.lower() == "true"
    )
    wrong = pred.to_numpy() != merged["label"].to_numpy()
    cols = ["model_id", "hsz_id", "label", "prediction", "net", "sarcasm", "panic", "text"]
    errors = merged.loc[flagged & wrong].copy()
    errors["prediction"] = pred[flagged & wrong]
    errors["text"] = errors["text"].astype(str).str.replace(r"\s+", " ", regex=True)
    return errors[cols]


# --- merging ---------------------------------------------------------------


def merge_scores(
    config: Config | None = None,
    *,
    scores_path: Path | None = None,
    eval_set_path: Path = EVAL_SET_CSV,
) -> pd.DataFrame:
    """Merge eval scores with labels; attach the predicted class column."""
    config = config or get_config()
    scores_path = scores_path or eval_scores_path(config)
    scores = pd.read_parquet(scores_path)
    labels = load_eval_set(eval_set_path)
    missing = set(scores["model_id"]) ^ set(config.models.candidates)
    if missing:
        logger.warning("model id mismatch vs config candidates: %s", sorted(missing))
    merged = scores.merge(
        labels[["hsz_id", "text", "label", "sarcasm", "panic"]],
        on="hsz_id",
        how="left",
        validate="many_to_one",
    )
    merged["prediction"] = predicted_classes(merged)
    return merged


# --- report rendering ------------------------------------------------------


def _fmt_pct(value: float) -> str:
    return f"{value:.3f}"


def majority_baseline(merged: pd.DataFrame) -> tuple[str, float, float]:
    """Return ``(majority_label, accuracy, macro_f1)`` for a constant predictor."""
    labels = list(merged["label"])
    majority = max(set(labels), key=labels.count)
    accuracy = labels.count(majority) / len(labels)
    return majority, accuracy, macro_f1(labels, [majority] * len(labels))


def _reliability_table(
    rows: list[tuple[float, float, int, float | None, float | None]]
) -> str:
    lines = ["| bin | n | mean conf | empirical acc |", "|---|---|---|---|"]
    for lo, hi, n, conf, acc in rows:
        if n == 0:
            lines.append(f"| {lo:.1f}–{hi:.1f} | 0 | – | – |")
        else:
            lines.append(f"| {lo:.1f}–{hi:.1f} | {n} | {conf:.3f} | {acc:.3f} |")
    return "\n".join(lines)


def _confusion_table(cm: np.ndarray) -> str:
    header = "| true \\ pred | " + " | ".join(CLASSES) + " |"
    sep = "|---|" + "---|" * len(CLASSES)
    lines = [header, sep]
    for i, label in enumerate(CLASSES):
        cells = " | ".join(str(int(cm[i, j])) for j in range(len(CLASSES)))
        lines.append(f"| **{label}** | {cells} |")
    return "\n".join(lines)


def render_report(
    reports: Sequence[ModelReport],
    merged: pd.DataFrame,
    chosen_id: str,
    reasoning: str,
    label_notes: dict[str, str] | None = None,
) -> str:
    """Render the full markdown report."""
    label_notes = label_notes if label_notes is not None else LABEL_MAPPING_NOTES
    n_posts = merged["hsz_id"].nunique()
    parts: list[str] = []
    parts.append("# Sentiment model selection (T3.3)\n")
    parts.append(
        "Generated by `pressflow.select_model` (T3.3) from "
        "`data/processed/eval_scores.parquet` and `eval/eval_set.csv` "
        f"({n_posts} consensus posts; ADR-0001, ADR-0003).\n"
    )

    parts.append("## Candidates & label mapping\n")
    parts.append("| model | native label mapping |\n|---|---|")
    for report in reports:
        parts.append(
            f"| `{report.model_id}` | {label_notes.get(report.model_id, '—')} |"
        )
    parts.append("")

    parts.append("## Metrics\n")
    majority, base_acc, base_f1 = majority_baseline(merged)
    parts.append(
        f"_Baseline: always predicting the majority class `{majority}` gives "
        f"accuracy {base_acc:.3f} and macro-F1 {base_f1:.3f}. Every candidate's "
        "hard-label accuracy is below that constant baseline, so the useful "
        "comparison is the directional (bull/bear) macro-F1 and the calibration "
        "of the soft probabilities._\n"
    )
    parts.append(
        "| model | accuracy | macro-F1 (3-class) | directional macro-F1 (bull/bear) "
        "| ECE | directional ECE |\n|---|---|---|---|---|---|"
    )
    for report in reports:
        parts.append(
            f"| `{report.model_id}` | {_fmt_pct(report.accuracy)} "
            f"| {_fmt_pct(report.macro_f1)} "
            f"| {_fmt_pct(report.directional_macro_f1)} "
            f"| {_fmt_pct(report.ece)} "
            f"| {_fmt_pct(report.directional_ece)} |"
        )
    parts.append("")

    parts.append("## Confusion matrices\n")
    for report in reports:
        parts.append(f"### `{report.model_id}`\n")
        parts.append(_confusion_table(report.confusion))
        parts.append("")

    parts.append("## Calibration\n")
    parts.append(
        "Aggregation consumes probabilities, so the *confidence* should match the "
        "observed accuracy. ECE is computed over the 3-class max probability; the "
        "directional ECE restricts to posts called bull/bear and uses that "
        "direction's probability as confidence.\n"
    )
    for report in reports:
        parts.append(f"### `{report.model_id}`\n")
        parts.append(
            f"- 3-class ECE: **{report.ece:.3f}**; "
            f"directional ECE: **{report.directional_ece:.3f}**\n"
        )
        parts.append("Directional reliability:\n")
        parts.append(_reliability_table(report.directional_rows))
        parts.append("")

    parts.append("## Sarcasm / panic error examples\n")
    pred = merged["prediction"] if "prediction" in merged.columns else pd.Series(
        predicted_classes(merged), index=merged.index
    )
    errors = sarcasm_panic_errors(merged, pred)
    if errors.empty:
        parts.append("_No misclassified sarcasm/panic examples._\n")
    else:
        parts.append(
            "| model | hsz_id | true | pred | net | sarcasm | panic | text |\n"
            "|---|---|---|---|---|---|---|---|"
        )
        for _, row in errors.iterrows():
            text = str(row["text"])[:120]
            parts.append(
                f"| `{row['model_id'].split('/')[-1]}` | {int(row['hsz_id'])} "
                f"| {row['label']} | {row['prediction']} | {row['net']:.2f} "
                f"| {row['sarcasm'] or ''} | {row['panic'] or ''} | {text} |"
            )
        parts.append("")

    parts.append("## Chosen model\n")
    parts.append(f"**`{chosen_id}`**\n")
    parts.append(f"Selection rule: {reasoning}\n")
    parts.append(_justification(reports, chosen_id) + "\n")
    return "\n".join(parts)


def _justification(reports: Sequence[ModelReport], chosen_id: str) -> str:
    chosen = next(r for r in reports if r.model_id == chosen_id)
    best_ece = min(r.directional_ece for r in reports)
    eligible = {
        r.model_id
        for r in reports
        if r.directional_ece <= best_ece + CALIBRATION_MARGIN
    }
    excluded = [r for r in reports if r.model_id not in eligible]
    others = [
        r for r in reports if r.model_id != chosen_id and r.model_id in eligible
    ]
    text = (
        f"`{chosen_id}` has the strongest directional discrimination "
        f"(directional macro-F1 {chosen.directional_macro_f1:.3f}) among the "
        f"candidates that are not pathologically over-confident (directional ECE "
        f"{chosen.directional_ece:.3f}, versus a best of {best_ece:.3f}). Because "
        f"the pipeline aggregates soft probabilities rather than hard labels "
        f"(ADR-0001/ADR-0003), a model that emits near-certain *wrong* directional "
        f"scores is more damaging to the aggregate than one with slightly weaker "
        f"per-post discrimination, so calibration gates the decision."
    )
    if others:
        text += (
            " Also calibrated enough but weaker on directional macro-F1: "
            + "; ".join(
                f"`{r.model_id}` ({r.directional_macro_f1:.3f})" for r in others
            )
            + "."
        )
    if excluded:
        text += (
            " Set aside for severe over-confidence (directional ECE more than "
            f"{CALIBRATION_MARGIN:.2f} above the best): "
            + "; ".join(
                f"`{r.model_id}` (directional macro-F1 "
                f"{r.directional_macro_f1:.3f}, directional ECE "
                f"{r.directional_ece:.3f})"
                for r in excluded
            )
            + "."
        )
    return text


def write_report(text: str, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_report(
    config: Config | None = None,
    *,
    scores_path: Path | None = None,
    eval_set_path: Path = EVAL_SET_CSV,
    report_path: Path = REPORT_PATH,
) -> tuple[str, list[ModelReport]]:
    """Evaluate every candidate, render and write the report."""
    config = config or get_config()
    merged = merge_scores(
        config, scores_path=scores_path, eval_set_path=eval_set_path
    )
    reports: list[ModelReport] = []
    for model_id in config.models.candidates:
        scored = merged[merged["model_id"] == model_id]
        if scored.empty:
            logger.warning("no scores for %s; skipping", model_id)
            continue
        reports.append(evaluate_model(model_id, scored))
    chosen_id, reasoning = choose_model(reports)
    text = render_report(reports, merged, chosen_id, reasoning)
    write_report(text, report_path)
    logger.info("chosen model: %s", chosen_id)
    return text, reports


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate sentiment models and pick a default."
    )
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_CSV)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    _, reports = build_report(
        scores_path=args.scores,
        eval_set_path=args.eval_set,
        report_path=args.report,
    )
    for report in reports:
        print(
            f"{report.model_id}: acc={report.accuracy:.3f} "
            f"macroF1={report.macro_f1:.3f} "
            f"dirF1={report.directional_macro_f1:.3f} ece={report.ece:.3f} "
            f"dirECE={report.directional_ece:.3f}"
        )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

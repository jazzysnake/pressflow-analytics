"""T3.6 — evaluate Jev (TypeSafe API) against the T3.3 baselines.

Reads ``data/processed/jev_scores.parquet`` (produced by
:mod:`pressflow.score_jev`) and the T3.1 consensus labels
(``eval/eval_set.csv``), computes the same metrics as T3.3
(:mod:`pressflow.select_model`: accuracy, macro-F1, directional macro-F1 and
the confusion matrix) and writes ``docs/reports/jev-eval.md`` with a side-by-side
comparison against the off-the-shelf candidates plus an explicit adopt/reject
verdict.

Adopt rule (T3.6 DoD): adopt Jev only when its directional macro-F1 *clearly*
beats hts5's 0.230 (by at least :data:`ADOPT_MARGIN`) **and** its accuracy beats
the 0.741 neutral baseline; otherwise keep hts5.

CLI::

    uv run python -m pressflow.eval_jev
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
from pressflow.score_jev import jev_scores_path
from pressflow.score_models import EVAL_SET_CSV, load_eval_set
from pressflow.select_model import (
    CLASSES,
    confusion_matrix,
    directional_macro_f1,
    macro_f1,
    predicted_classes,
)

logger = logging.getLogger("pressflow.eval_jev")

#: Report location.
REPORT_PATH = REPO_ROOT / "docs" / "reports" / "jev-eval.md"

#: hts5 baseline directional macro-F1 (T3.3, ``docs/reports/model-selection.md``).
HTS5_DIRECTIONAL_MACRO_F1 = 0.230

#: Majority-class (always ``neutral``) accuracy baseline.
NEUTRAL_BASELINE_ACC = 0.741

#: How much Jev must beat hts5's directional macro-F1 by to count as "clearly
#: better". The directional classes are small (40 bull / 25 bear), so a margin
#: guards against noise.
ADOPT_MARGIN = 0.05

#: T3.3 baseline metrics (accuracy, macro-F1, directional macro-F1), copied
#: from ``docs/reports/model-selection.md`` (source of truth for the comparison).
BASELINES: tuple[tuple[str, float, float, float], ...] = (
    ("NYTK/sentiment-hts5-xlm-roberta-hungarian", 0.453, 0.354, 0.230),
    ("NYTK/sentiment-ohb3-xlm-roberta-hungarian", 0.587, 0.418, 0.264),
    ("cardiffnlp/twitter-xlm-roberta-base-sentiment", 0.463, 0.319, 0.170),
)

__all__ = [
    "REPORT_PATH",
    "HTS5_DIRECTIONAL_MACRO_F1",
    "NEUTRAL_BASELINE_ACC",
    "ADOPT_MARGIN",
    "BASELINES",
    "JevReport",
    "Decision",
    "evaluate_jev",
    "decide",
    "render_report",
    "write_report",
    "build_report",
    "main",
]


@dataclass(frozen=True)
class JevReport:
    model_id: str
    n: int
    accuracy: float
    macro_f1: float
    directional_macro_f1: float
    confusion: np.ndarray


@dataclass(frozen=True)
class Decision:
    adopt: bool
    reason: str


def evaluate_jev(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
) -> JevReport:
    """Merge Jev scores with consensus labels and compute T3.3 metrics."""
    merged = scores.merge(
        labels[["hsz_id", "label"]],
        on="hsz_id",
        how="left",
        validate="one_to_one",
    )
    missing = int(merged["label"].isna().sum())
    if missing:
        raise ValueError(f"{missing} Jev score(s) have no matching eval label")
    y_true = list(merged["label"])
    pred = list(predicted_classes(merged))
    model_id = str(scores["model_id"].iloc[0]) if len(scores) else "jev"
    return JevReport(
        model_id=model_id,
        n=len(merged),
        accuracy=float(
            np.mean(
                np.asarray(y_true, dtype=object) == np.asarray(pred, dtype=object)
            )
        ),
        macro_f1=macro_f1(y_true, pred),
        directional_macro_f1=directional_macro_f1(y_true, pred),
        confusion=confusion_matrix(y_true, pred),
    )


def decide(
    report: JevReport,
    *,
    hts5_directional_macro_f1: float = HTS5_DIRECTIONAL_MACRO_F1,
    neutral_baseline_acc: float = NEUTRAL_BASELINE_ACC,
    margin: float = ADOPT_MARGIN,
) -> Decision:
    """Apply the T3.6 adopt rule and return a :class:`Decision`."""
    dir_bar = hts5_directional_macro_f1 + margin
    beats_dir = report.directional_macro_f1 > dir_bar
    beats_acc = report.accuracy > neutral_baseline_acc
    adopt = beats_dir and beats_acc
    reason = (
        f"Jev directional macro-F1 {report.directional_macro_f1:.3f} vs hts5 "
        f"{hts5_directional_macro_f1:.3f} + margin {margin:.2f} = {dir_bar:.3f} "
        f"({'pass' if beats_dir else 'fail'}); accuracy {report.accuracy:.3f} "
        f"vs neutral baseline {neutral_baseline_acc:.3f} "
        f"({'pass' if beats_acc else 'fail'}). "
    )
    if adopt:
        reason += "Both conditions hold → adopt Jev."
    else:
        reason += "Condition not met → keep hts5."
    return Decision(adopt=adopt, reason=reason)


def _confusion_table(cm: np.ndarray) -> str:
    header = "| true \\ pred | " + " | ".join(CLASSES) + " |"
    sep = "|---|" + "---|" * len(CLASSES)
    lines = [header, sep]
    for i, label in enumerate(CLASSES):
        cells = " | ".join(str(int(cm[i, j])) for j in range(len(CLASSES)))
        lines.append(f"| **{label}** | {cells} |")
    return "\n".join(lines)


def _per_class_table(cm: np.ndarray) -> str:
    lines = [
        "| class | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for i, label in enumerate(CLASSES):
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
        lines.append(
            f"| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} "
            f"| {int(cm[i, :].sum())} |"
        )
    return "\n".join(lines)


def render_report(report: JevReport, decision: Decision) -> str:
    """Render the markdown report for the Jev candidate."""
    parts: list[str] = []
    parts.append("# Jev (TypeSafe API) sentiment evaluation (T3.6)\n")
    parts.append(
        "Generated by `pressflow.eval_jev` from `data/processed/jev_scores.parquet` "
        f"and `eval/eval_set.csv` ({report.n} consensus posts). Metrics computed with "
        "the T3.3 functions in `pressflow.select_model`; baselines copied from "
        "`docs/reports/model-selection.md`.\n"
    )

    parts.append("## Method\n")
    parts.append(
        "Each post's text was sent to `POST https://api.typesafe.ai/v1/systemone` "
        'with `model=\"jev-latest\"` and a single `choice` question answered over the '
        "options `bull` (expects rise / buy), `bear` (expects fall / sell) and "
        "`neutral` (no directional view). Jev's `probabilities` were normalised to "
        "`p_bull / p_bear / p_neutral` (sum = 1) with `net = p_bull - p_bear`; the "
        "hard class is the argmax, matching the off-the-shelf pipeline. Requests "
        "were sent with a ~0.2 s delay and retried on 429/5xx with exponential "
        "backoff.\n"
    )

    parts.append("## Metrics vs off-the-shelf candidates\n")
    parts.append(
        f"_Neutral (always predict `neutral`) baseline: accuracy "
        f"{NEUTRAL_BASELINE_ACC:.3f}. Adopt requires directional macro-F1 > "
        f"{HTS5_DIRECTIONAL_MACRO_F1:.3f} + {ADOPT_MARGIN:.2f} = "
        f"{HTS5_DIRECTIONAL_MACRO_F1 + ADOPT_MARGIN:.3f} **and** accuracy > "
        f"{NEUTRAL_BASELINE_ACC:.3f}._\n"
    )
    parts.append(
        "| model | accuracy | macro-F1 (3-class) | directional macro-F1 (bull/bear) |\n"
        "|---|---|---|---|"
    )
    parts.append(
        f"| **`{report.model_id}`** | **{report.accuracy:.3f}** "
        f"| **{report.macro_f1:.3f}** | **{report.directional_macro_f1:.3f}** |"
    )
    for model_id, acc, mf1, dirf1 in BASELINES:
        parts.append(f"| `{model_id}` | {acc:.3f} | {mf1:.3f} | {dirf1:.3f} |")
    parts.append("")

    parts.append("## Jev confusion matrix\n")
    parts.append("Rows = consensus label, columns = Jev hard (argmax) prediction.\n")
    parts.append(_confusion_table(report.confusion))
    parts.append("")

    parts.append("### Per-class precision / recall / F1\n")
    parts.append(_per_class_table(report.confusion))
    parts.append("")

    parts.append("## Verdict\n")
    parts.append(f"**{'ADOPT Jev' if decision.adopt else 'REJECT Jev — keep hts5'}**\n")
    parts.append(decision.reason + "\n")
    if not decision.adopt and report.directional_macro_f1 > (
        HTS5_DIRECTIONAL_MACRO_F1 + ADOPT_MARGIN
    ):
        parts.append(
            "_Note: Jev dominates every off-the-shelf candidate on both macro-F1 "
            f"({report.macro_f1:.3f}) and directional macro-F1 "
            f"({report.directional_macro_f1:.3f}), and its confusion matrix is far "
            "more balanced. It is held back only by hard-label accuracy "
            f"({report.accuracy:.3f}) narrowly trailing the constant-neutral "
            f"baseline ({NEUTRAL_BASELINE_ACC:.3f}): its extra `bull`/`bear` calls "
            "trade some neutral accuracy for directional recall. Under the T3.6 "
            "rule both gates must pass, so hts5 is retained; Jev's directional "
            "signal is nonetheless worth revisiting at T3.4._\n"
        )
    return "\n".join(parts)


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
) -> tuple[str, JevReport, Decision]:
    """Evaluate Jev, render and write the report."""
    config = config or get_config()
    scores_path = scores_path or jev_scores_path(config)
    scores = pd.read_parquet(scores_path)
    labels = load_eval_set(eval_set_path)
    report = evaluate_jev(scores, labels)
    decision = decide(report)
    text = render_report(report, decision)
    write_report(text, report_path)
    logger.info(
        "Jev: acc=%.3f macroF1=%.3f dirF1=%.3f -> %s",
        report.accuracy,
        report.macro_f1,
        report.directional_macro_f1,
        "adopt" if decision.adopt else "reject (keep hts5)",
    )
    return text, report, decision


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Jev scores against the T3.3 baselines."
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
    _, report, decision = build_report(
        scores_path=args.scores,
        eval_set_path=args.eval_set,
        report_path=args.report,
    )
    print(
        f"{report.model_id}: acc={report.accuracy:.3f} "
        f"macroF1={report.macro_f1:.3f} "
        f"dirF1={report.directional_macro_f1:.3f} "
        f"-> {'ADOPT' if decision.adopt else 'REJECT (keep hts5)'}"
    )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

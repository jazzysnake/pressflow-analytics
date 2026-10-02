"""T1.5 — scrape validation report over the ``posts`` Parquet.

Computes completeness/cleanliness metrics and enforces the T1.5 thresholds:

- ≥ 95 % of posts have non-null ``text`` and ``ts``
- 0 duplicate ``hsz_id``

A violation flags the dataset for review and blocks Epic 03 (sentiment).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq

from pressflow import schema as s
from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.forum.crawler import CrawlerPaths, load_checkpoint

#: Enforced thresholds (T1.5 DoD).
MIN_TEXT_RATE = 0.95
MIN_TS_RATE = 0.95
MAX_DUPLICATES = 0

#: A page with fewer/more posts than this is flagged as an outlier.
PAGE_OUTLIER_LOW = 15
PAGE_OUTLIER_HIGH = 30

#: Calendar gap (days) that is reported as suspicious.
GAP_DAYS = 3

#: Checkpoint counters surfaced in the report (drop counts happen pre-write).
DROP_COUNT_KEYS: tuple[str, ...] = (
    "comments_seen",
    "comments_skipped_no_date",
    "comments_skipped_no_key",
    "records_dropped",
    "pages_skipped",
)

#: Human-readable labels for the drop-count table.
_DROP_COUNT_LABELS: dict[str, str] = {
    "comments_seen": "Comments seen",
    "comments_skipped_no_date": "Skipped: no date (topic opener/widget)",
    "comments_skipped_no_key": "Skipped: no vote span / hsz_id",
    "records_dropped": "Dropped: missing required field",
    "pages_skipped": "Pages skipped (fetch failure)",
}

__all__ = [
    "MIN_TEXT_RATE",
    "MIN_TS_RATE",
    "MAX_DUPLICATES",
    "ValidationReport",
    "validate_table",
    "write_report",
]


@dataclass
class ValidationReport:
    """Result of validating a ``posts`` table."""

    total_posts: int
    pages: int
    page_count_min: int
    page_count_max: int
    page_count_median: float
    outlier_pages: dict[int, int]
    first_ts: datetime | None
    last_ts: datetime | None
    coverage_days: int
    gaps_over_threshold: list[tuple[date, date, int]]
    missing_rates: dict[str, float]
    duplicate_hsz: int
    posts_per_day_mean: float
    posts_per_day_median: float
    posts_per_day_max: int
    ts_parse_failures: int | None
    drop_counts: dict[str, int] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_markdown(self, *, label: str, posts_path: Path) -> str:
        status = "PASS" if self.ok else "FAIL"
        try:
            display_path = posts_path.resolve().relative_to(REPO_ROOT)
        except ValueError:  # pragma: no cover - path outside the repo
            display_path = posts_path
        lines: list[str] = []
        lines.append("# Scrape validation report")
        lines.append("")
        lines.append(f"**Scope:** {label}")
        lines.append("")
        lines.append(f"**Dataset:** `{display_path.as_posix()}`")
        lines.append("")
        lines.append(f"**Verdict:** {status}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Posts | {self.total_posts} |")
        lines.append(f"| Distinct pages | {self.pages} |")
        lines.append(
            f"| Posts/page (min / median / max) | {self.page_count_min} / "
            f"{self.page_count_median:.0f} / {self.page_count_max} |"
        )
        lines.append(f"| First post (Budapest) | {self._fmt(self.first_ts)} |")
        lines.append(f"| Last post (Budapest) | {self._fmt(self.last_ts)} |")
        lines.append(f"| Coverage | {self.coverage_days} days |")
        lines.append(f"| Gaps > {GAP_DAYS} days | {len(self.gaps_over_threshold)} |")
        lines.append(f"| Duplicate `hsz_id` | {self.duplicate_hsz} |")
        lines.append(f"| Posts/day (mean / median / max) | {self.posts_per_day_mean:.1f} / "
                     f"{self.posts_per_day_median:.0f} / {self.posts_per_day_max} |")
        if self.ts_parse_failures is not None:
            lines.append(f"| `ts` parse failures | {self.ts_parse_failures} |")
        lines.append("")

        if self.drop_counts or self.ts_parse_failures is not None:
            lines.append("## Dropped / skipped comments (pre-write)")
            lines.append("")
            lines.append("| Count | Value |")
            lines.append("|---|---|")
            for key in DROP_COUNT_KEYS:
                if key in self.drop_counts:
                    lines.append(
                        f"| {_DROP_COUNT_LABELS[key]} | {self.drop_counts[key]} |"
                    )
            if self.ts_parse_failures is not None:
                lines.append(f"| `ts` parse failures | {self.ts_parse_failures} |")
            lines.append("")
            lines.append(
                "> Dropped/skipped comments never reach the Parquet write, so they "
                "cannot fail the written-row `text`/`ts` thresholds below."
            )
            lines.append("")

        lines.append("## Missing-field rates")
        lines.append("")
        lines.append("| Column | Missing / null rate |")
        lines.append("|---|---|")
        for column, rate in self.missing_rates.items():
            lines.append(f"| `{column}` | {rate * 100:.2f}% |")
        lines.append("")

        lines.append("## Thresholds (T1.5 DoD)")
        lines.append("")
        lines.append("| Check | Threshold | Result |")
        lines.append("|---|---|---|")
        text_rate = 1.0 - self.missing_rates.get(s.TEXT, 0.0)
        ts_rate = 1.0 - self.missing_rates.get(s.TS, 0.0)
        lines.append(
            f"| non-null `text` | ≥ {MIN_TEXT_RATE * 100:.0f}% | "
            f"{text_rate * 100:.2f}% ({self._mark(text_rate >= MIN_TEXT_RATE)}) |"
        )
        lines.append(
            f"| non-null `ts` | ≥ {MIN_TS_RATE * 100:.0f}% | "
            f"{ts_rate * 100:.2f}% ({self._mark(ts_rate >= MIN_TS_RATE)}) |"
        )
        lines.append(
            f"| duplicate `hsz_id` | = {MAX_DUPLICATES} | "
            f"{self.duplicate_hsz} ({self._mark(self.duplicate_hsz <= MAX_DUPLICATES)}) |"
        )
        lines.append("")

        lines.append("## Outliers & gaps")
        lines.append("")
        if self.outlier_pages:
            lines.append(
                f"Pages outside [{PAGE_OUTLIER_LOW}, {PAGE_OUTLIER_HIGH}] posts "
                "(expect ~20–24):"
            )
            lines.append("")
            for page, count in sorted(self.outlier_pages.items()):
                lines.append(f"- `oldal={page}`: {count} posts")
        else:
            lines.append("No posts-per-page outliers.")
        lines.append("")
        if self.gaps_over_threshold:
            lines.append(f"Calendar gaps > {GAP_DAYS} days:")
            lines.append("")
            for start, end, days in self.gaps_over_threshold:
                lines.append(f"- {start.isoformat()} → {end.isoformat()} ({days} days)")
        else:
            lines.append(f"No calendar gaps > {GAP_DAYS} days.")
        lines.append("")

        if self.violations:
            lines.append("## ⚠️ Violations (blocks Epic 03 — review required)")
            lines.append("")
            for violation in self.violations:
                lines.append(f"- {violation}")
            lines.append("")
        else:
            lines.append("No threshold violations.")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _fmt(value: datetime | None) -> str:
        return value.isoformat() if value is not None else "—"

    @staticmethod
    def _mark(ok: bool) -> str:
        return "ok" if ok else "**VIOLATION**"


def validate_table(
    table,
    *,
    ts_parse_failures: int | None = None,
    drop_counts: Mapping[str, int] | None = None,
) -> ValidationReport:
    """Compute the T1.5 validation report for a ``posts`` pyarrow table.

    ``ts_parse_failures`` and ``drop_counts`` come from the crawl checkpoint and
    make the pre-write drop/skip counts visible (T1.6 issue 5).
    """
    total = table.num_rows
    hsz_values = [v for v in table.column(s.HSZ_ID).to_pylist() if v is not None]
    duplicate_hsz = len(hsz_values) - len(set(hsz_values))

    missing_rates: dict[str, float] = {}
    for column in table.column_names:
        column_data = table.column(column)
        nulls = column_data.null_count
        # Treat empty strings as missing for the text column.
        if s.TEXT in table.column_names and column == s.TEXT:
            nulls += sum(
                1 for v in column_data.to_pylist() if isinstance(v, str) and not v.strip()
            )
        missing_rates[column] = (nulls / total) if total else 0.0

    # Posts per page.
    page_counts = Counter(
        v for v in table.column(s.PAGE).to_pylist() if v is not None
    )
    counts = list(page_counts.values())
    outlier_pages = {
        page: count
        for page, count in page_counts.items()
        if count < PAGE_OUTLIER_LOW or count > PAGE_OUTLIER_HIGH
    }

    # Date coverage.
    timestamps = [ts for ts in table.column(s.TS).to_pylist() if ts is not None]
    first_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None
    day_counts: Counter[date] = Counter(ts.date() for ts in timestamps)
    days = sorted(day_counts)
    gaps: list[tuple[date, date, int]] = []
    for previous, current in zip(days, days[1:]):
        delta = (current - previous).days
        if delta > GAP_DAYS:
            gaps.append((previous, current, delta))
    coverage_days = (days[-1] - days[0]).days + 1 if days else 0

    per_day = list(day_counts.values())
    violations: list[str] = []
    text_rate = 1.0 - missing_rates.get(s.TEXT, 0.0)
    ts_rate = 1.0 - missing_rates.get(s.TS, 0.0)
    if text_rate < MIN_TEXT_RATE:
        violations.append(
            f"non-null text rate {text_rate:.2%} < {MIN_TEXT_RATE:.0%}"
        )
    if ts_rate < MIN_TS_RATE:
        violations.append(f"non-null ts rate {ts_rate:.2%} < {MIN_TS_RATE:.0%}")
    if duplicate_hsz > MAX_DUPLICATES:
        violations.append(f"duplicate hsz_id count {duplicate_hsz} > {MAX_DUPLICATES}")

    return ValidationReport(
        total_posts=total,
        pages=len(page_counts),
        page_count_min=min(counts) if counts else 0,
        page_count_max=max(counts) if counts else 0,
        page_count_median=statistics.median(counts) if counts else 0.0,
        outlier_pages=outlier_pages,
        first_ts=first_ts,
        last_ts=last_ts,
        coverage_days=coverage_days,
        gaps_over_threshold=gaps,
        missing_rates=missing_rates,
        duplicate_hsz=duplicate_hsz,
        posts_per_day_mean=statistics.fmean(per_day) if per_day else 0.0,
        posts_per_day_median=statistics.median(per_day) if per_day else 0.0,
        posts_per_day_max=max(per_day) if per_day else 0,
        ts_parse_failures=ts_parse_failures,
        drop_counts=dict(drop_counts or {}),
        violations=violations,
    )


def _read_posts(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"posts Parquet not found: {path}")
    return pq.read_table(path)


def write_report(
    posts_path: Path,
    out_path: Path,
    *,
    label: str,
    ts_parse_failures: int | None = None,
    drop_counts: Mapping[str, int] | None = None,
) -> ValidationReport:
    """Validate ``posts_path`` and write the markdown report to ``out_path``."""
    table = _read_posts(posts_path)
    report = validate_table(
        table, ts_parse_failures=ts_parse_failures, drop_counts=drop_counts
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        report.to_markdown(label=label, posts_path=posts_path), encoding="utf-8"
    )
    return report


def _checkpoint_failures(cfg: Config) -> int | None:
    checkpoint = load_checkpoint(CrawlerPaths.from_config(cfg).checkpoint)
    value = checkpoint.get("ts_parse_failures")
    return int(value) if value is not None else None


def _checkpoint_drop_counts(cfg: Config) -> dict[str, int]:
    checkpoint = load_checkpoint(CrawlerPaths.from_config(cfg).checkpoint)
    return {
        key: int(checkpoint[key])
        for key in DROP_COUNT_KEYS
        if checkpoint.get(key) is not None
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the scraped posts Parquet.")
    parser.add_argument("--posts", type=str, default=None, help="posts.parquet path")
    parser.add_argument("--out", type=str, default=None, help="markdown output path")
    parser.add_argument("--label", type=str, default="full crawl")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = get_config()
    paths = CrawlerPaths.from_config(cfg)
    posts_path = Path(args.posts) if args.posts else paths.posts
    out_path = Path(args.out) if args.out else REPO_ROOT / "docs" / "reports" / "scrape-validation.md"

    report = write_report(
        posts_path,
        out_path,
        label=args.label,
        ts_parse_failures=_checkpoint_failures(cfg),
        drop_counts=_checkpoint_drop_counts(cfg),
    )
    print(
        json.dumps(
            {
                "posts": report.total_posts,
                "pages": report.pages,
                "duplicate_hsz": report.duplicate_hsz,
                "drop_counts": report.drop_counts,
                "violations": report.violations,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

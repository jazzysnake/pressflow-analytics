"""T1.5 — validation report tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from pressflow import schema as s
from pressflow.forum.validation import validate_table, write_report

BUD = ZoneInfo("Europe/Budapest")


def _row(
    hsz: int,
    *,
    page: int = 1,
    author: str = "user",
    text: str = "hello",
    ts: datetime | None = None,
) -> dict:
    return {
        s.POST_ID: hsz,
        s.HSZ_ID: hsz,
        s.AUTHOR: author,
        s.USER_ID: 1,
        s.TS: ts or datetime(2026, 10, 1, 12, 0, tzinfo=BUD),
        s.REPLY_TO_POST_ID: None,
        s.REPLY_TO_HSZ_ID: None,
        s.UPVOTES: 1,
        s.DOWNVOTES: 0,
        s.TEXT: text,
        s.PAGE: page,
    }


def _table(rows: list[dict]) -> pa.Table:
    columns = {name: [r[name] for r in rows] for name in s.POSTS_COLUMNS}
    return pa.Table.from_pydict(columns, schema=s.POSTS_SCHEMA)


def test_clean_table_passes():
    rows = [_row(i, page=1 + i // 20) for i in range(60)]
    report = validate_table(_table(rows))

    assert report.ok
    assert report.violations == []
    assert report.total_posts == 60
    assert report.pages == 3
    assert report.duplicate_hsz == 0
    assert report.missing_rates[s.TEXT] == 0.0
    assert report.missing_rates[s.TS] == 0.0


def test_duplicate_hsz_is_a_violation():
    rows = [_row(100), _row(100), _row(101)]
    report = validate_table(_table(rows))
    assert report.duplicate_hsz == 1
    assert not report.ok
    assert any("duplicate" in v for v in report.violations)


def test_low_text_rate_is_a_violation():
    # 1 blank text out of 4 -> 75% < 95%.
    rows = [_row(1), _row(2), _row(3), _row(4, text="")]
    report = validate_table(_table(rows))
    assert not report.ok
    assert any("text rate" in v for v in report.violations)


def test_gap_detection():
    day0 = datetime(2026, 9, 1, 10, 0, tzinfo=BUD)
    rows = [
        _row(1, ts=day0),
        _row(2, ts=day0 + timedelta(days=1)),
        # 5-day gap
        _row(3, ts=day0 + timedelta(days=6)),
    ]
    report = validate_table(_table(rows))
    assert len(report.gaps_over_threshold) == 1
    start, end, days = report.gaps_over_threshold[0]
    assert days == 5
    assert report.coverage_days == 7


def test_page_outliers_flagged():
    rows = [_row(i, page=1) for i in range(5)]  # too few posts
    report = validate_table(_table(rows))
    assert report.outlier_pages == {1: 5}


def test_posts_per_day_stats():
    day0 = datetime(2026, 9, 1, 10, 0, tzinfo=BUD)
    rows = [
        _row(1, ts=day0),
        _row(2, ts=day0 + timedelta(hours=1)),
        _row(3, ts=day0 + timedelta(days=1)),
    ]
    report = validate_table(_table(rows))
    assert report.posts_per_day_max == 2
    assert report.posts_per_day_mean == pytest.approx(1.5)


def test_drop_counts_surface_in_report(tmp_path):
    """T1.6 issue 5: pre-write drop/skip counts appear in the report."""
    rows = [_row(i, page=1 + i // 20) for i in range(40)]
    posts = tmp_path / "posts.parquet"
    import pyarrow.parquet as pq

    pq.write_table(_table(rows), posts)

    out = tmp_path / "report.md"
    drop_counts = {
        "comments_seen": 500,
        "comments_skipped_no_date": 3,
        "comments_skipped_no_key": 7,
        "records_dropped": 2,
        "pages_skipped": 1,
    }
    report = write_report(
        posts, out, label="drop counts", drop_counts=drop_counts
    )
    text = out.read_text(encoding="utf-8")
    assert "Dropped / skipped comments (pre-write)" in text
    assert "Comments seen | 500" in text
    assert "Skipped: no vote span / hsz_id | 7" in text
    assert "Pages skipped (fetch failure) | 1" in text
    assert report.drop_counts == drop_counts


def test_write_report_creates_markdown(tmp_path):
    rows = [_row(i, page=1 + i // 20) for i in range(40)]
    posts = tmp_path / "posts.parquet"
    import pyarrow.parquet as pq

    pq.write_table(_table(rows), posts)

    out = tmp_path / "report.md"
    report = write_report(posts, out, label="unit test scope", ts_parse_failures=2)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "unit test scope" in text
    assert "PASS" in text
    assert report.ts_parse_failures == 2

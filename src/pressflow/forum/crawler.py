"""T1.4 — resumable, polite crawler for the Portfolio.hu OTP thread.

Walks pages **newest → oldest**, honouring the config delay (≥ 0.35 s) and
retrying with exponential backoff.  Posts are written to Parquet incrementally
(batch part files) and a checkpoint records the next page to fetch, so an
interrupted run resumes without gaps or duplicates.  Dedupe is on ``hsz_id``
(the ``posts`` natural key, ``schema.NATURAL_KEYS``).

Layout (paths relative to the repo root, from ``config/default.yaml``)::

    data/raw/crawl_checkpoint.json   # resume state
    data/raw/parts/part-NNNNN.parquet  # incremental batches
    data/processed/posts.parquet       # compacted, schema-validated output

CLI::

    uv run python -m pressflow.forum.crawler --pages 100          # smoke slice
    uv run python -m pressflow.forum.crawler --full               # 10-year crawl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pressflow import schema as s
from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.forum.http import FetchError, ForumClient, PageResponse
from pressflow.forum.pagination import (
    discover_max_oldal,
    find_start_oldal,
    newest_timestamp,
)
from pressflow.forum.parser import parse_page, parse_page_detailed
from pressflow.forum.timestamps import BUDAPEST, parse_http_date, resolve_timestamp

logger = logging.getLogger("pressflow.forum.crawler")

#: Pages per incremental Parquet flush.
DEFAULT_BATCH_PAGES = 50

__all__ = [
    "DEFAULT_BATCH_PAGES",
    "CrawlStats",
    "CrawlerPaths",
    "crawl",
    "compact_parts",
    "load_checkpoint",
    "save_checkpoint",
]


# --- paths -----------------------------------------------------------------


@dataclass(frozen=True)
class CrawlerPaths:
    """Resolved filesystem locations used by the crawler."""

    raw_dir: Path
    processed_dir: Path
    checkpoint: Path
    parts_dir: Path
    posts: Path

    @classmethod
    def from_config(cls, cfg: Config, root: Path = REPO_ROOT) -> "CrawlerPaths":
        raw_dir = root / cfg.paths.raw
        processed_dir = root / cfg.paths.processed
        return cls(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            checkpoint=raw_dir / "crawl_checkpoint.json",
            parts_dir=raw_dir / "parts",
            posts=processed_dir / "posts.parquet",
        )


# --- stats -----------------------------------------------------------------


@dataclass
class CrawlStats:
    """Counters for one crawler run (also persisted in the checkpoint)."""

    pages_fetched: int = 0
    pages_skipped: int = 0
    pages_reused: int = 0
    posts_seen: int = 0
    posts_written: int = 0
    duplicates: int = 0
    ts_parse_failures: int = 0
    records_dropped: int = 0
    missing_date_header: int = 0
    comments_seen: int = 0
    comments_skipped_no_date: int = 0
    comments_skipped_no_key: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pages_fetched": self.pages_fetched,
            "pages_skipped": self.pages_skipped,
            "pages_reused": self.pages_reused,
            "posts_seen": self.posts_seen,
            "posts_written": self.posts_written,
            "duplicates": self.duplicates,
            "ts_parse_failures": self.ts_parse_failures,
            "records_dropped": self.records_dropped,
            "missing_date_header": self.missing_date_header,
            "comments_seen": self.comments_seen,
            "comments_skipped_no_date": self.comments_skipped_no_date,
            "comments_skipped_no_key": self.comments_skipped_no_key,
            "failures": self.failures,
        }


# --- checkpoint helpers ----------------------------------------------------


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load the JSON checkpoint, returning ``{}`` when absent/corrupt."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        logger.warning("could not read checkpoint %s; starting fresh", path)
        return {}
    return data if isinstance(data, dict) else {}


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    """Atomically write the checkpoint (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --- Parquet helpers -------------------------------------------------------


def _records_to_table(records: Sequence[dict[str, Any]]) -> pa.Table:
    """Build a ``POSTS_SCHEMA`` table from resolved records."""
    columns: dict[str, list[Any]] = {name: [] for name in s.POSTS_COLUMNS}
    for record in records:
        for name in s.POSTS_COLUMNS:
            columns[name].append(record[name])
    return pa.Table.from_pydict(columns, schema=s.POSTS_SCHEMA)


def _write_part(parts_dir: Path, sequence: int, records: Sequence[dict[str, Any]]) -> Path:
    parts_dir.mkdir(parents=True, exist_ok=True)
    table = _records_to_table(records)
    path = parts_dir / f"part-{sequence:05d}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


def _clear_parts_dir(parts_dir: Path) -> None:
    """Remove stale part files so ``--fresh`` cannot dedupe against them (T1.6)."""
    if parts_dir.exists():
        shutil.rmtree(parts_dir, ignore_errors=True)


def _load_seen_hsz_ids(parts_dir: Path) -> set[int]:
    """Collect ``hsz_id`` values already written, for cross-resume dedupe."""
    seen: set[int] = set()
    if not parts_dir.is_dir():
        return seen
    for part in sorted(parts_dir.glob("part-*.parquet")):
        table = pq.read_table(part, columns=[s.HSZ_ID])
        seen.update(value for value in table.column(s.HSZ_ID).to_pylist() if value is not None)
    return seen


def compact_parts(
    parts_dir: Path,
    posts_path: Path,
    *,
    dedupe: bool = True,
) -> int:
    """Concatenate part files into the canonical ``posts.parquet``.

    Deduplicates on ``hsz_id`` (keep last) and sorts by timestamp.  Returns the
    number of rows written.
    """
    tables = [pq.read_table(part) for part in sorted(parts_dir.glob("part-*.parquet"))]
    posts_path.parent.mkdir(parents=True, exist_ok=True)

    if not tables:
        empty = s.POSTS_SCHEMA.empty_table()
        tmp = posts_path.with_suffix(".parquet.tmp")
        pq.write_table(empty, tmp)
        os.replace(tmp, posts_path)
        return 0

    table = pa.concat_tables(tables)
    if dedupe and table.num_rows:
        frame = table.to_pandas()
        frame = frame.drop_duplicates(subset=[s.HSZ_ID], keep="last")
        frame = frame.sort_values(s.TS, kind="stable")
        table = pa.Table.from_pandas(frame, schema=s.POSTS_SCHEMA, preserve_index=False)

    tmp = posts_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, posts_path)
    return table.num_rows


# --- page processing -------------------------------------------------------


def _resolve_page(
    response: PageResponse,
    page_number: int,
    stats: CrawlStats,
    fallback_now: datetime,
) -> list[dict[str, Any]]:
    """Parse + timestamp-resolve one page into schema-ready records."""
    records, parse_stats = parse_page_detailed(response.html, page_number)
    stats.comments_seen += parse_stats.comments_seen
    stats.comments_skipped_no_date += parse_stats.skipped_no_date
    stats.comments_skipped_no_key += parse_stats.skipped_no_key
    stats.posts_seen += len(records)

    now = response.server_date
    if now is None:
        stats.missing_date_header += 1
        now = fallback_now
        logger.warning("page %s had no HTTP Date header; using run clock", page_number)

    resolved: list[dict[str, Any]] = []
    for record in records:
        ts = resolve_timestamp(record.get(s.TS), now)
        if ts is None:
            stats.ts_parse_failures += 1
            continue
        record = dict(record)
        record[s.TS] = ts
        required = (
            record.get(s.POST_ID),
            record.get(s.HSZ_ID),
            record.get(s.AUTHOR),
            record.get(s.USER_ID),
            record.get(s.UPVOTES),
            record.get(s.DOWNVOTES),
            record.get(s.TEXT),
            record.get(s.PAGE),
        )
        if any(value is None for value in required):
            stats.records_dropped += 1
            continue
        resolved.append(record)
    return resolved


# --- main crawl ------------------------------------------------------------


def _determine_max_oldal(
    client: ForumClient,
    checkpoint: dict[str, Any],
    stats: CrawlStats,
    fallback_now: datetime,
    cache: dict[int, PageResponse],
) -> tuple[int, list[dict[str, Any]], datetime | None]:
    """Return ``(max_oldal, base_records, newest_ts)``.

    The newest page is fetched once and reused as the first page of the run, so
    it is never fetched twice.  It is cached under ``max_oldal`` so a later
    binary-search probe of that page is served from memory.  On resume the
    values come from the checkpoint.
    """
    if "max_oldal" in checkpoint:
        max_oldal = int(checkpoint["max_oldal"])
        stored = checkpoint.get("newest_ts")
        newest_ts = datetime.fromisoformat(stored) if stored else None
        return max_oldal, [], newest_ts

    response = client.fetch(None)  # bare thread URL == newest page
    stats.pages_fetched += 1
    max_oldal = discover_max_oldal(response.html)
    cache[max_oldal] = response
    logger.info("discovered max oldal = %d", max_oldal)
    records = _resolve_page(response, max_oldal, stats, fallback_now)
    newest_ts = records[0][s.TS] if records else None
    return max_oldal, records, newest_ts


def crawl(
    cfg: Config | None = None,
    *,
    target_date: datetime | None = None,
    end_oldal: int | None = None,
    max_pages: int | None = None,
    client: ForumClient | None = None,
    paths: CrawlerPaths | None = None,
    batch_pages: int = DEFAULT_BATCH_PAGES,
    compact: bool = True,
    fresh: bool = False,
) -> CrawlStats:
    """Run (or resume) the newest→oldest crawl.

    Parameters
    ----------
    target_date:
        Stop page (inclusive) is the smallest ``oldal`` whose newest post is
        ``>= target_date`` (T1.3 binary search).
    end_oldal:
        Explicit stop page, overriding ``target_date``.
    max_pages:
        Safety cap on pages attempted in *this* run (used by the smoke test).
    fresh:
        Ignore any existing checkpoint and start over.
    """
    cfg = cfg or get_config()
    paths = paths or CrawlerPaths.from_config(cfg)
    client = client or ForumClient(cfg)
    fallback_now = datetime.now(timezone.utc)
    stats = CrawlStats()

    if fresh:
        _clear_parts_dir(paths.parts_dir)
    checkpoint = {} if fresh else load_checkpoint(paths.checkpoint)
    seen = _load_seen_hsz_ids(paths.parts_dir)
    sequence = int(checkpoint.get("parts", 0))

    # Fetched pages are cached so binary-search probes are never re-fetched by
    # the crawl loop (T1.6 issue 2).
    cache: dict[int, PageResponse] = {}

    # 1. Newest page + max oldal (fetch once; reused as first page).
    max_oldal, base_records, newest_ts = _determine_max_oldal(
        client, checkpoint, stats, fallback_now, cache
    )

    # 2. Decide the stop page.
    if end_oldal is not None:
        start_oldal = end_oldal
    elif target_date is not None:
        probe = _make_newest_ts_probe(client, fallback_now, stats, cache)
        start_oldal = find_start_oldal(probe, target_date, max_oldal)
    else:
        start_oldal = 1
    logger.info("crawl oldal range: %d -> %d", max_oldal, start_oldal)

    # 3. Dedupe/queue the base page (if fetched this run).
    pending: list[dict[str, Any]] = []
    pages_in_batch = 0

    def absorb(records: list[dict[str, Any]]) -> None:
        for record in records:
            hsz = record[s.HSZ_ID]
            if hsz in seen:
                stats.duplicates += 1
                continue
            seen.add(hsz)
            pending.append(record)

    def flush(next_oldal: int) -> None:
        nonlocal sequence, pages_in_batch
        if pending:
            sequence += 1
            _write_part(paths.parts_dir, sequence, pending)
            stats.posts_written += len(pending)
            pending.clear()
        checkpoint.update(
            {
                "max_oldal": max_oldal,
                "newest_ts": (newest_ts.isoformat() if newest_ts else None),
                "next_oldal": next_oldal,
                "parts": sequence,
                "posts_written": stats.posts_written,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **{
                    key: value
                    for key, value in stats.as_dict().items()
                    if key != "failures"
                },
            }
        )
        save_checkpoint(paths.checkpoint, checkpoint)
        pages_in_batch = 0

    if base_records:
        absorb(base_records)
        pages_in_batch += 1

    if "next_oldal" in checkpoint:
        oldal = int(checkpoint["next_oldal"])
    else:
        oldal = max_oldal - 1  # base page already consumed
    pages_attempted = 0 if not base_records else 1
    last_oldal: int | None = None

    while oldal >= start_oldal:
        if max_pages is not None and pages_attempted >= max_pages:
            break
        last_oldal = oldal
        cached = cache.get(oldal)
        if cached is not None:
            stats.pages_reused += 1
            absorb(_resolve_page(cached, oldal, stats, fallback_now))
        else:
            try:
                response = client.fetch(oldal)
            except FetchError as exc:
                stats.pages_skipped += 1
                stats.failures.append({"oldal": oldal, "error": str(exc)})
                logger.warning("skipping oldal=%d: %s", oldal, exc)
            else:
                stats.pages_fetched += 1
                cache[oldal] = response
                absorb(_resolve_page(response, oldal, stats, fallback_now))
        pages_attempted += 1
        pages_in_batch += 1
        if pages_in_batch >= batch_pages:
            flush(oldal - 1)
        oldal -= 1

    if last_oldal is not None and pages_in_batch:
        flush(last_oldal - 1)

    if compact:
        rows = compact_parts(paths.parts_dir, paths.posts)
        logger.info("compacted %d rows into %s", rows, paths.posts)

    logger.info(
        "crawl done: fetched=%d skipped=%d posts_written=%d duplicates=%d",
        stats.pages_fetched,
        stats.pages_skipped,
        stats.posts_written,
        stats.duplicates,
    )
    return stats


def _make_newest_ts_probe(
    client: ForumClient,
    fallback_now: datetime,
    stats: CrawlStats,
    cache: dict[int, PageResponse],
):
    """Return an ``oldal -> newest timestamp`` callable for T1.3.

    Fetched pages are stored in the shared ``cache`` so the crawl loop reuses
    them instead of fetching the same page a second time (T1.6 issue 2).
    """

    def probe(oldal: int) -> datetime:
        response = cache.get(oldal)
        if response is None:
            response = client.fetch(oldal)
            stats.pages_fetched += 1
            cache[oldal] = response
        ts = newest_timestamp(
            parse_page(response.html, oldal),
            response.server_date or fallback_now,
        )
        if ts is None:
            raise ValueError(f"could not resolve newest timestamp for oldal={oldal}")
        return ts

    return probe


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable Portfolio.hu forum crawler.")
    parser.add_argument("--pages", type=int, default=None, help="max pages to attempt")
    parser.add_argument(
        "--end-oldal", type=int, default=None, help="stop at this oldal (inclusive)"
    )
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="stop at the page containing this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="crawl the full configured window (window_years back from newest post)",
    )
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoint")
    parser.add_argument(
        "--batch-pages", type=int, default=DEFAULT_BATCH_PAGES, help="pages per flush"
    )
    parser.add_argument(
        "--no-compact", action="store_true", help="skip final posts.parquet compaction"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = get_config()

    target_date = None
    if args.target_date:
        target_date = datetime.strptime(args.target_date, "%Y-%m-%d").replace(tzinfo=BUDAPEST)
    elif args.full:
        checkpoint = (
            {}
            if args.fresh
            else load_checkpoint(CrawlerPaths.from_config(cfg).checkpoint)
        )
        newest = checkpoint.get("newest_ts")
        if newest:
            newest_dt = datetime.fromisoformat(newest)
        else:
            newest_dt = datetime.now(BUDAPEST)
        target_date = newest_dt - timedelta(days=365 * cfg.window_years)

    stats = crawl(
        cfg,
        target_date=target_date,
        end_oldal=args.end_oldal,
        max_pages=args.pages,
        batch_pages=args.batch_pages,
        compact=not args.no_compact,
        fresh=args.fresh,
    )
    print(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

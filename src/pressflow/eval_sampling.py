"""T3.1 — build the ~200-post sentiment eval sample (ADR-0001).

The two-model consensus eval set (T3.1) needs a small, hand-checkable sample of
forum posts spread evenly across the 10-year analysis window
(2016-10 → 2026-10, ADR-0006).  This module picks ~20 evenly-spaced *dates*,
locates each date's page with the existing pagination binary search (T1.3) and
parser/HTTP helpers (T1.1/T1.4), and writes:

* ``eval/sample.csv``        — ``hsz_id, post_id, ts, text`` (~200 rows).
* ``eval/sampling_log.csv``  — ``target_date, oldal, n_posts`` (one row/date).

Date → page strategy
--------------------
A forum page's newest post is all we need to place a date, but the relationship
between ``oldal`` and time is strongly nonuniform (~1.2k–2.3k pages/year across
2016–2026), so a plain even-``oldal`` walk would leave year-long gaps.  We
therefore:

1. fetch the newest page once (``discover_max_oldal``), and
2. binary-search the first target date with the existing
   :func:`pressflow.forum.pagination.find_start_oldal` (which also caches
   ~16 pages spread across the whole thread), then
3. treat every cached ``(newest_ts, oldal)`` pair as a calibration anchor and
   pick each remaining target's page by piecewise-linear interpolation in the
   ``date → oldal`` plane, refining the anchors as we go.

This keeps the request budget at ~30–40 pages (politeness: the
:class:`~pressflow.forum.http.ForumClient` delay of 0.35 s is unchanged) while
still resolving each date to within days.

CLI::

    uv run python -m pressflow.eval_sampling
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pressflow import schema as s
from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.forum.http import ForumClient, PageResponse
from pressflow.forum.pagination import (
    discover_max_oldal,
    find_start_oldal,
    newest_timestamp,
)
from pressflow.forum.parser import parse_page
from pressflow.forum.timestamps import BUDAPEST, resolve_timestamp

logger = logging.getLogger("pressflow.eval_sampling")

#: Analysis window (ADR-0006): 2016-10 → 2026-10.
SAMPLING_START = datetime(2016, 10, 1, tzinfo=BUDAPEST)
SAMPLING_END = datetime(2026, 10, 1, tzinfo=BUDAPEST)

#: ~20 dates ≈ every 6 months.
DEFAULT_N_DATES = 20

#: ~10 posts/date → ~200 posts.
DEFAULT_POSTS_PER_DATE = 10

#: Refine any date gap wider than this (8 months, safely under the 9-month DoD).
MAX_GAP_DAYS = 240.0

#: Hard cap on refinement passes (keeps the request budget polite).
MAX_GAP_CORRECTIONS = 8

#: Tracked output directory (repo root; never the gitignored ``data/``).
EVAL_DIR = REPO_ROOT / "eval"
SAMPLE_CSV = EVAL_DIR / "sample.csv"
SAMPLING_LOG_CSV = EVAL_DIR / "sampling_log.csv"

SAMPLE_FIELDS: tuple[str, ...] = ("hsz_id", "post_id", "ts", "text")
LOG_FIELDS: tuple[str, ...] = ("target_date", "oldal", "n_posts")

__all__ = [
    "SAMPLING_START",
    "SAMPLING_END",
    "DEFAULT_N_DATES",
    "DEFAULT_POSTS_PER_DATE",
    "evenly_spaced_dates",
    "interpolate_oldal",
    "select_nearest_posts",
    "build_sample",
    "write_sample",
    "main",
]


# --- pure helpers ----------------------------------------------------------


def evenly_spaced_dates(
    start: datetime, end: datetime, n: int
) -> list[datetime]:
    """Return ``n`` dates evenly spaced over ``[start, end]`` (inclusive)."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if end <= start:
        raise ValueError("end must be after start")
    span = end - start
    return [start + span * (i / (n - 1)) for i in range(n)]


def interpolate_oldal(
    anchors: Sequence[tuple[datetime, int]], target: datetime, max_oldal: int
) -> int:
    """Estimate the ``oldal`` whose newest post is nearest ``target``.

    ``anchors`` are ``(newest_ts, oldal)`` pairs known to be monotone; they are
    sorted internally.  Outside the anchor range the nearest anchor is
    returned; inside, the estimate is piecewise-linear.  The result is
    clamped to ``[1, max_oldal]``.
    """
    ordered = sorted(anchors)
    if not ordered:
        raise ValueError("at least one anchor is required")
    if target <= ordered[0][0]:
        return max(1, min(ordered[0][1], max_oldal))
    if target >= ordered[-1][0]:
        return max(1, min(ordered[-1][1], max_oldal))
    for (t0, o0), (t1, o1) in zip(ordered, ordered[1:]):
        if t0 <= target <= t1:
            if t1 == t0:
                return max(1, min(o0, max_oldal))
            frac = (target - t0).total_seconds() / (t1 - t0).total_seconds()
            return max(1, min(round(o0 + frac * (o1 - o0)), max_oldal))
    return max(1, min(ordered[-1][1], max_oldal))  # pragma: no cover - defensive


def select_nearest_posts(
    posts: Sequence[dict[str, Any]],
    target: datetime,
    k: int,
    skip_hsz_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Pick the ``k`` posts closest in time to ``target`` (oldest→newest).

    ``posts`` must carry a ``ts`` key; ``skip_hsz_ids`` excludes already-sampled
    posts (a page is never reused, but this keeps the sample duplicate-free).
    """
    skip = skip_hsz_ids or set()
    available = [p for p in posts if p["hsz_id"] not in skip]
    available.sort(key=lambda p: abs((p["ts"] - target).total_seconds()))
    return sorted(available[:k], key=lambda p: p["ts"])


# --- sampler ---------------------------------------------------------------


class _Sampler:
    """Caches fetched pages and parsed records so probes are never re-fetched."""

    def __init__(self, client: ForumClient, *, fallback_now: datetime) -> None:
        self.client = client
        self.fallback_now = fallback_now
        self.pages: dict[int | None, PageResponse] = {}
        self.records: dict[int | None, list[dict[str, Any]]] = {}
        self.pages_fetched = 0

    def fetch(self, oldal: int | None) -> PageResponse:
        response = self.pages.get(oldal)
        if response is None:
            response = self.client.fetch(oldal)
            self.pages[oldal] = response
            self.pages_fetched += 1
        return response

    def _page_records(self, oldal: int | None) -> list[dict[str, Any]]:
        if oldal not in self.records:
            page_number = oldal if oldal is not None else 0
            response = self.fetch(oldal)
            self.records[oldal] = parse_page(response.html, page_number)
        return self.records[oldal]

    def newest_ts(self, oldal: int | None) -> datetime:
        response = self.fetch(oldal)
        ts = newest_timestamp(
            self._page_records(oldal), response.server_date or self.fallback_now
        )
        if ts is None:
            raise ValueError(f"no resolvable timestamp on oldal={oldal}")
        return ts

    def posts(self, oldal: int) -> list[dict[str, Any]]:
        """Resolved posts for ``oldal`` in document order (newest→oldest)."""
        response = self.fetch(oldal)
        now = response.server_date or self.fallback_now
        resolved: list[dict[str, Any]] = []
        for record in self._page_records(oldal):
            ts = resolve_timestamp(record.get(s.TS), now)
            if ts is None:
                continue
            resolved.append(
                {
                    "hsz_id": record[s.HSZ_ID],
                    "post_id": record[s.POST_ID],
                    "ts": ts,
                    "text": (record.get(s.TEXT) or "").strip(),
                    "oldal": oldal,
                }
            )
        return resolved


def _build_anchors(sampler: _Sampler) -> list[tuple[datetime, int]]:
    """Collect ``(newest_ts, oldal)`` anchors from every cached page."""
    anchors: list[tuple[datetime, int]] = []
    for oldal in [key for key in sampler.pages if key is not None]:
        try:
            anchors.append((sampler.newest_ts(oldal), int(oldal)))
        except ValueError as exc:  # pragma: no cover - defensive
            logger.warning("skipping anchor for oldal=%s: %s", oldal, exc)
    anchors.sort()
    return anchors


def _fill_gaps(
    sampler: _Sampler,
    sample_rows: list[dict[str, Any]],
    log_rows: list[dict[str, Any]],
    anchors: list[tuple[datetime, int]],
    used_hsz_ids: set[int],
    used_oldals: set[int],
    *,
    posts_per_correction: int,
) -> int:
    """Insert pages into any date gap wider than :data:`MAX_GAP_DAYS`.

    The 2016–2026 thread spans ~1.0k–5.0k pages/year, so a single interpolated
    page per target can still leave a >9-month hole.  Repeatedly pick the widest
    gap and fetch the *page-bracket midpoint* (one request); because
    ``newest_ts`` is monotone in ``oldal`` this always splits the gap.  Only a
    few posts are taken from a correction page so the total stays ~200.  Returns
    the number of corrections made.
    """
    tried: set[tuple[int, int]] = set()
    corrections = 0
    while corrections < MAX_GAP_CORRECTIONS:
        ordered = sorted(sample_rows, key=lambda row: row["ts"])
        worst: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        for left, right in zip(ordered, ordered[1:]):
            gap = (right["ts"] - left["ts"]).total_seconds() / 86400.0
            bracket = (int(left["oldal"]), int(right["oldal"]))
            if gap <= MAX_GAP_DAYS or bracket in tried or bracket[1] - bracket[0] < 2:
                continue
            if worst is None or gap > worst[0]:
                worst = (gap, left, right)
        if worst is None:
            break
        gap_days, left, right = worst
        lo, hi = int(left["oldal"]), int(right["oldal"])
        tried.add((lo, hi))
        oldal = (lo + hi) // 2
        if oldal in used_oldals:
            continue
        used_oldals.add(oldal)
        posts = sampler.posts(oldal)
        if posts:
            anchors.append((posts[0]["ts"], oldal))
            anchors.sort()
        mid = left["ts"] + (right["ts"] - left["ts"]) / 2
        chosen = select_nearest_posts(posts, mid, posts_per_correction, used_hsz_ids)
        if not chosen:
            continue
        used_hsz_ids.update(post["hsz_id"] for post in chosen)
        sample_rows.extend(chosen)
        log_rows.append(
            {
                "target_date": mid.date().isoformat(),
                "oldal": oldal,
                "n_posts": len(chosen),
            }
        )
        corrections += 1
        logger.info(
            "gap %.0fd before %s -> mid %s via oldal %d (%d posts)",
            gap_days,
            right["ts"].date(),
            mid.date(),
            oldal,
            len(chosen),
        )
    return corrections


# --- orchestration ---------------------------------------------------------


def build_sample(
    cfg: Config | None = None,
    *,
    n_dates: int = DEFAULT_N_DATES,
    posts_per_date: int = DEFAULT_POSTS_PER_DATE,
    start: datetime = SAMPLING_START,
    end: datetime = SAMPLING_END,
    client: ForumClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(sample_rows, log_rows)`` for the eval sample.

    ``sample_rows`` are dicts with ``hsz_id, post_id, ts, text`` (sorted by
    ``ts``); ``log_rows`` are ``target_date, oldal, n_posts`` (one row per
    target date).  Network access goes through ``client`` (injectable for
    tests).
    """
    cfg = cfg or get_config()
    client = client or ForumClient(cfg)
    sampler = _Sampler(client, fallback_now=datetime.now(timezone.utc))

    # 1. Newest page -> max oldal (reused as the highest anchor).
    newest = sampler.fetch(None)
    max_oldal = discover_max_oldal(newest.html)
    sampler.pages[max_oldal] = newest  # ?oldal=max redirects to the same page

    targets = evenly_spaced_dates(start, end, n_dates)

    # 2. Binary-search the first target; this caches ~16 pages across the
    #    whole thread that double as calibration anchors.
    start_oldal = find_start_oldal(sampler.newest_ts, targets[0], max_oldal)
    logger.info("first target %s -> oldal %d", targets[0].date(), start_oldal)

    anchors = _build_anchors(sampler)

    sample_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    used_hsz_ids: set[int] = set()
    used_oldals: set[int] = set()

    for target in targets:
        oldal = interpolate_oldal(anchors, target, max_oldal)
        while oldal in used_oldals and oldal < max_oldal:
            oldal += 1
        used_oldals.add(oldal)

        posts = sampler.posts(oldal)
        if posts:
            anchors.append((posts[0]["ts"], oldal))
            anchors.sort()

        chosen = select_nearest_posts(posts, target, posts_per_date, used_hsz_ids)
        used_hsz_ids.update(post["hsz_id"] for post in chosen)
        sample_rows.extend(chosen)
        log_rows.append(
            {
                "target_date": target.date().isoformat(),
                "oldal": oldal,
                "n_posts": len(chosen),
            }
        )
        logger.info(
            "target %s -> oldal %d (%d posts, newest %s)",
            target.date(),
            oldal,
            len(chosen),
            posts[0]["ts"].date() if posts else "?",
        )

    sample_rows.sort(key=lambda row: row["ts"])
    corrections = _fill_gaps(
        sampler,
        sample_rows,
        log_rows,
        anchors,
        used_hsz_ids,
        used_oldals,
        posts_per_correction=min(4, posts_per_date),
    )
    sample_rows.sort(key=lambda row: row["ts"])
    logger.info(
        "fetched %d pages; sampled %d posts (%d gap corrections)",
        sampler.pages_fetched,
        len(sample_rows),
        corrections,
    )
    return sample_rows, log_rows


def write_sample(
    sample_rows: Sequence[dict[str, Any]],
    log_rows: Sequence[dict[str, Any]],
    *,
    out_dir: Path = EVAL_DIR,
) -> tuple[Path, Path]:
    """Write ``sample.csv`` and ``sampling_log.csv`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample.csv"
    log_path = out_dir / "sampling_log.csv"

    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_FIELDS))
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(
                {
                    "hsz_id": row["hsz_id"],
                    "post_id": row["post_id"],
                    "ts": row["ts"].isoformat(),
                    "text": row["text"],
                }
            )

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_FIELDS))
        writer.writeheader()
        writer.writerows(log_rows)

    return sample_path, log_path


# --- CLI -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the T3.1 eval sample from the forum."
    )
    parser.add_argument("--n-dates", type=int, default=DEFAULT_N_DATES)
    parser.add_argument("--posts-per-date", type=int, default=DEFAULT_POSTS_PER_DATE)
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    rows, log_rows = build_sample(
        n_dates=args.n_dates, posts_per_date=args.posts_per_date
    )
    sample_path, log_path = write_sample(rows, log_rows, out_dir=args.out_dir)
    print(f"wrote {len(rows)} posts -> {sample_path}")
    print(f"wrote {len(log_rows)} dates -> {log_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

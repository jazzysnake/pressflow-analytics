"""T1.3 — pagination discovery and target-date binary search.

The forum addresses pages by ``?oldal=N`` where ``oldal=1`` is the **oldest**
page and the newest page is the highest ``oldal`` (served at the bare thread
URL and also reachable via ``?oldal=<max>``, which redirects there).  A page has
~20 posts ordered newest→oldest, so its *first* parsed post is its newest.

Everything here is a pure function or takes an injected ``page_newest_ts``
callable, so the search logic is unit-testable with mocked pages and no network
(see ``tests/test_forum_pagination.py``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

from pressflow import schema as s
from pressflow.forum.timestamps import resolve_timestamp

#: ``oldal=12345`` links anywhere in the page (pagination widgets).
_OLDAL_RE = re.compile(r"oldal=(\d+)")

#: Type of the injected "newest timestamp of page N" probe.
NewestTsFn = Callable[[int], datetime]

__all__ = [
    "extract_oldal_values",
    "discover_max_oldal",
    "newest_timestamp",
    "find_start_oldal",
    "resolve_start_oldal",
]


def extract_oldal_values(html: str) -> list[int]:
    """Return every ``oldal=N`` value found in the HTML, in document order."""
    return [int(value) for value in _OLDAL_RE.findall(html)]


def discover_max_oldal(newest_html: str) -> int:
    """Discover the highest ``oldal`` from the **newest** page markup.

    On the newest page the active page is not a link, so the largest linked
    ``oldal`` is the second-newest page: ``max_oldal = max(links) + 1``.
    """
    values = extract_oldal_values(newest_html)
    if not values:
        raise ValueError("no 'oldal=' links found; not a forum pagination page")
    return max(values) + 1


def newest_timestamp(records: list[dict], now: datetime) -> datetime | None:
    """Resolve the newest post timestamp on a page.

    ``records`` are in document order (newest→oldest); the first one whose
    timestamp resolves wins.  Returns ``None`` if the page has no resolvable
    timestamp.
    """
    for record in records:
        resolved = resolve_timestamp(record.get(s.TS), now)
        if resolved is not None:
            return resolved
    return None


def find_start_oldal(
    page_newest_ts: NewestTsFn,
    target: datetime,
    max_oldal: int,
) -> int:
    """Smallest ``oldal`` whose newest post is ``>= target``.

    Pages get newer as ``oldal`` grows, so the predicate is monotonic.  If even
    the oldest page already reaches ``target``, ``1`` is returned (T1.3 DoD).
    If ``target`` is newer than the newest page, ``max_oldal`` is returned.
    """
    if max_oldal < 1:
        raise ValueError("max_oldal must be >= 1")
    if page_newest_ts(1) >= target:
        return 1
    if page_newest_ts(max_oldal) < target:
        return max_oldal

    low, high = 1, max_oldal
    while low < high:
        mid = (low + high) // 2
        if page_newest_ts(mid) >= target:
            high = mid
        else:
            low = mid + 1
    return low


def resolve_start_oldal(
    fetch_newest_ts: NewestTsFn,
    target: datetime,
    *,
    newest_html: str | None = None,
    max_oldal: int | None = None,
) -> int:
    """Binary-search the start page for ``target``.

    ``newest_html`` (base thread page) lets this discover ``max_oldal`` itself;
    otherwise pass an explicit ``max_oldal``.  ``fetch_newest_ts`` must resolve
    the newest post timestamp of a given ``oldal`` (network in production).
    """
    if max_oldal is None:
        if newest_html is None:
            raise ValueError("provide either newest_html or max_oldal")
        max_oldal = discover_max_oldal(newest_html)
    return find_start_oldal(fetch_newest_ts, target, max_oldal)

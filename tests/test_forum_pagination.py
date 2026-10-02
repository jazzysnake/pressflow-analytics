"""T1.3 — pagination discovery & date binary-search tests (mocked, no network)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from pressflow.forum.pagination import (
    discover_max_oldal,
    extract_oldal_values,
    find_start_oldal,
    newest_timestamp,
    resolve_start_oldal,
)

BUD = ZoneInfo("Europe/Budapest")

#: Verified live checkpoints (T1.3): ``(oldal, newest post timestamp)``.
#: ``50666`` is the exact 10-year start for target ``2016-10-01``:
#: newest(50666) = 2016-10-03 >= target, newest(50665) = 2016-09-30 < target.
LIVE_CHECKPOINTS: list[tuple[int, datetime]] = [
    (30000, datetime(2010, 11, 25, 10, 17, tzinfo=BUD)),
    (50665, datetime(2016, 9, 30, tzinfo=BUD)),
    (50666, datetime(2016, 10, 3, tzinfo=BUD)),
    (66773, datetime(2026, 8, 18, tzinfo=BUD)),
]
TEN_YEAR_TARGET = datetime(2016, 10, 1, tzinfo=BUD)
TEN_YEAR_START_OLDAL = 50666


def _linear_probe(points: list[tuple[int, datetime]]):
    """Piecewise-linear ``oldal -> newest ts`` probe through live checkpoints."""
    xs = [point[0] for point in points]
    stamps = [point[1] for point in points]

    def probe(oldal: int) -> datetime:
        if oldal <= xs[0]:
            i = 0
        elif oldal >= xs[-1]:
            i = len(xs) - 2
        else:
            i = next(j for j in range(len(xs) - 1) if xs[j] <= oldal <= xs[j + 1])
        x0, x1 = xs[i], xs[i + 1]
        t0, t1 = stamps[i], stamps[i + 1]
        frac = (oldal - x0) / (x1 - x0)
        return t0 + timedelta(seconds=frac * (t1 - t0).total_seconds())

    return probe


def _pagination_html(max_linked: int) -> str:
    """Newest-page pagination: links show oldal = max-1 .. 1 (no self link)."""
    links = "".join(
        f'<a href="https://forum.portfolio.hu/topics/x/5224?oldal={n}">{n}</a>'
        for n in range(max_linked, 0, -1)
    )
    return f"<html><body><ul class='pagination'>{links}</ul></body></html>"


def test_extract_oldal_values_ignores_other_numbers():
    html = '<a href="?oldal=5">5</a><a href="?limit=40">40</a><a href="?oldal=3">3</a>'
    assert extract_oldal_values(html) == [5, 3]


def test_discover_max_oldal_from_newest_page():
    # links show 67047 (page 2) .. 1, so max_oldal == 67048.
    html = _pagination_html(67047)
    assert discover_max_oldal(html) == 67048


def test_discover_max_oldal_no_links_raises():
    with pytest.raises(ValueError):
        discover_max_oldal("<html>no pagination here</html>")


def test_newest_timestamp_uses_first_resolvable_record():
    records = [
        {"ts": "tegnap, 09:00"},
        {"ts": "ma, 08:00"},
    ]
    now = datetime(2026, 10, 2, 12, 0, tzinfo=BUD)
    assert newest_timestamp(records, now) == datetime(2026, 10, 1, 9, 0, tzinfo=BUD)


def test_newest_timestamp_skips_unresolvable():
    records = [{"ts": "garbage"}, {"ts": "2010. 11. 25. 10:17"}]
    now = datetime(2026, 10, 2, tzinfo=BUD)
    assert newest_timestamp(records, now) == datetime(2010, 11, 25, 10, 17, tzinfo=BUD)


def test_find_start_oldal_matches_synthetic_index():
    """oldal o has newest ts = 2010-01-01 + o days (monotonic)."""
    base = datetime(2010, 1, 1, tzinfo=BUD)

    def probe(oldal: int) -> datetime:
        return base + timedelta(days=oldal)

    target = base + timedelta(days=500)
    assert find_start_oldal(probe, target, 1000) == 500
    # target sits between page 500's fresh edge and page 499.
    assert probe(499) < target <= probe(500)


def test_target_older_than_oldest_page_returns_one():
    base = datetime(2010, 1, 1, tzinfo=BUD)

    def probe(oldal: int) -> datetime:
        return base + timedelta(days=oldal)

    assert find_start_oldal(probe, datetime(2000, 1, 1, tzinfo=BUD), 1000) == 1


def test_target_newer_than_newest_page_returns_max():
    base = datetime(2010, 1, 1, tzinfo=BUD)

    def probe(oldal: int) -> datetime:
        return base + timedelta(days=oldal)

    assert find_start_oldal(probe, datetime(2030, 1, 1, tzinfo=BUD), 1000) == 1000


def test_binary_search_is_logarithmic():
    """At most ~log2(max) probes for a large synthetic index."""
    base = datetime(2000, 1, 1, tzinfo=BUD)
    calls: list[int] = []

    def probe(oldal: int) -> datetime:
        calls.append(oldal)
        return base + timedelta(days=oldal)

    find_start_oldal(probe, base + timedelta(days=100_000), 1_000_000)
    assert len(calls) <= 2 * 20 + 3  # log2(1e6) ≈ 20


def test_resolve_start_oldal_with_newest_html():
    base = datetime(2010, 1, 1, tzinfo=BUD)

    def probe(oldal: int) -> datetime:
        return base + timedelta(days=oldal)

    html = _pagination_html(999)  # -> max_oldal == 1000
    assert (
        resolve_start_oldal(probe, base + timedelta(days=250), newest_html=html) == 250
    )


def test_live_checkpoints_yield_exact_ten_year_start():
    """The verified 10-year start for 2016-10-01 is exactly ``oldal=50666``.

    The probe interpolates the recorded live checkpoints; the boundary is
    pinned by the two verified pages 50665 (2016-09-30) and 50666 (2016-10-03).
    """
    probe = _linear_probe(LIVE_CHECKPOINTS)
    assert probe(50665) < TEN_YEAR_TARGET <= probe(50666)
    assert find_start_oldal(probe, TEN_YEAR_TARGET, 67048) == TEN_YEAR_START_OLDAL

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

#: Documented real checkpoints (ADR-0004 / T1.3), used by the sanity test.
CHECKPOINT_OLD = (30000, datetime(2010, 11, 25, tzinfo=BUD))
CHECKPOINT_NEW = (66773, datetime(2026, 8, 17, tzinfo=BUD))


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


def test_realistic_checkpoints_yield_plausible_start():
    """2026-10-01 minus 10 years should land near oldal ~43,000.

    The probe is a piecewise-linear model calibrated to the two documented
    checkpoints; the assertion is a plausibility band, not an exact value
    (exact value comes from a real crawl sanity check).
    """
    (o1, d1), (o2, d2) = CHECKPOINT_OLD, CHECKPOINT_NEW
    pages_per_second = (o2 - o1) / (d2 - d1).total_seconds()

    def probe(oldal: int) -> datetime:
        return d2 + timedelta(seconds=(oldal - o2) / pages_per_second)

    target = datetime(2016, 10, 1, tzinfo=BUD)
    start = find_start_oldal(probe, target, 67048)
    assert 40_000 <= start <= 47_000

"""T1.2 — raw timestamp resolution tests."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from pressflow.forum.timestamps import (
    BUDAPEST,
    parse_http_date,
    resolve_http_date,
    resolve_timestamp,
)

# Fri, 02 Oct 2026 18:24:11 GMT == 20:24:11 in Budapest (CEST, UTC+2).
CTX = parse_http_date("Fri, 02 Oct 2026 18:24:11 GMT")


def test_parse_http_date_is_aware():
    parsed = parse_http_date("Fri, 02 Oct 2026 18:24:11 GMT")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.astimezone(timezone.utc).hour == 18


def test_parse_http_date_invalid_returns_none():
    assert parse_http_date(None) is None
    assert parse_http_date("") is None
    assert parse_http_date("not a date") is None


def test_today_relative():
    result = resolve_timestamp("ma, 20:08", CTX)
    assert result == datetime(2026, 10, 2, 20, 8, tzinfo=BUDAPEST)
    assert result.tzinfo == ZoneInfo("Europe/Budapest")


def test_yesterday_relative():
    result = resolve_timestamp("tegnap, 14:30", CTX)
    assert result == datetime(2026, 10, 1, 14, 30, tzinfo=BUDAPEST)


def test_absolute():
    result = resolve_timestamp("2010. 11. 25. 10:17", CTX)
    assert result == datetime(2010, 11, 25, 10, 17, tzinfo=BUDAPEST)
    assert str(result.tzinfo) == "Europe/Budapest"


def test_absolute_tolerates_irregular_spacing():
    assert resolve_timestamp("2010.11.25. 10:17", CTX) == datetime(
        2010, 11, 25, 10, 17, tzinfo=BUDAPEST
    )
    assert resolve_timestamp("2010. 11. 25.10:17", CTX) == datetime(
        2010, 11, 25, 10, 17, tzinfo=BUDAPEST
    )


def test_midnight_boundary_uses_budapest_date():
    """A page fetched 00:10 BUD (22:10 UTC) must call that day ``ma``."""
    header = "Fri, 02 Oct 2026 22:10:00 GMT"  # 00:10 on 2026-10-03 in Budapest
    now = parse_http_date(header)
    assert now.astimezone(BUDAPEST).date().isoformat() == "2026-10-03"

    assert resolve_timestamp("ma, 00:05", now) == datetime(
        2026, 10, 3, 0, 5, tzinfo=BUDAPEST
    )
    assert resolve_timestamp("tegnap, 23:55", now) == datetime(
        2026, 10, 2, 23, 55, tzinfo=BUDAPEST
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hétfő, 18:24", datetime(2026, 9, 28, 18, 24, tzinfo=BUDAPEST)),
        ("kedd, 09:00", datetime(2026, 9, 29, 9, 0, tzinfo=BUDAPEST)),
        ("szerda, 09:00", datetime(2026, 9, 30, 9, 0, tzinfo=BUDAPEST)),
        ("csütörtök, 09:00", datetime(2026, 10, 1, 9, 0, tzinfo=BUDAPEST)),
        ("péntek, 09:00", datetime(2026, 10, 2, 9, 0, tzinfo=BUDAPEST)),
        ("szombat, 09:00", datetime(2026, 9, 26, 9, 0, tzinfo=BUDAPEST)),
        ("vasárnap, 09:00", datetime(2026, 9, 27, 9, 0, tzinfo=BUDAPEST)),
    ],
)
def test_weekday_relative(raw, expected):
    """The site uses weekday names for the current week (not in original ticket)."""
    assert resolve_timestamp(raw, CTX) == expected


def test_weekday_boundary_is_not_in_the_future():
    # CTX is a Friday; the most recent Friday is today, not next week.
    result = resolve_timestamp("péntek, 23:59", CTX)
    assert result.date() == CTX.astimezone(BUDAPEST).date()


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "garbage", "ma", "ma, 25:00", "tegnap, 12:99", "2026. 13. 01. 10:00"],
)
def test_invalid_returns_none(raw):
    assert resolve_timestamp(raw, CTX) is None


def test_resolve_http_date_convenience():
    assert resolve_http_date("ma, 20:08", "Fri, 02 Oct 2026 18:24:11 GMT") == datetime(
        2026, 10, 2, 20, 8, tzinfo=BUDAPEST
    )
    assert resolve_http_date("ma, 20:08", None) is None


def test_output_is_tz_aware_budapest():
    result = resolve_timestamp("2010. 11. 25. 10:17", CTX)
    assert result.tzinfo is not None
    assert str(result.tzinfo) == "Europe/Budapest"

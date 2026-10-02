"""T1.2 — resolve raw forum timestamps to tz-aware Europe/Budapest datetimes.

The forum renders these timestamp forms:

- ``ma, 21:09``      — "today" at HH:MM
- ``tegnap, 14:30``  — "yesterday" at HH:MM
- ``hétfő, 18:24``   — the most recent named weekday at HH:MM (site shows a
  weekday name for posts within the current week; not in the original ticket
  but required to avoid dropping ~a week of posts)
- ``2010. 11. 25. 10:17`` — absolute

"Now" must be the **HTTP ``Date`` header of the request that fetched the page**,
never a process-global clock: pages fetched around midnight would otherwise
mislabel ``ma`` / ``tegnap`` / weekday names.  Use :func:`parse_http_date` to
turn the header into an aware datetime and pass it as ``now`` to
:func:`resolve_timestamp`.

Invalid or unparseable values return ``None`` (counted by T1.5) and never raise.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

#: Canonical timezone for every post timestamp (ADR-0015).
BUDAPEST = ZoneInfo("Europe/Budapest")

#: ``ma, HH:MM`` / ``tegnap, HH:MM``.
_RELATIVE_RE = re.compile(r"^(ma|tegnap)\s*,\s*(\d{1,2}):(\d{2})$", re.IGNORECASE)

#: Hungarian weekday names -> ``datetime.weekday()`` (Monday == 0).
HUNGARIAN_WEEKDAYS: dict[str, int] = {
    "hétfő": 0,
    "kedd": 1,
    "szerda": 2,
    "csütörtök": 3,
    "péntek": 4,
    "szombat": 5,
    "vasárnap": 6,
}

#: ``hétfő, HH:MM`` etc. (weekday form used for the current week's posts).
_WEEKDAY_RE = re.compile(
    r"^(hétfő|kedd|szerda|csütörtök|péntek|szombat|vasárnap)\s*,\s*(\d{1,2}):(\d{2})$",
    re.IGNORECASE,
)

#: ``YYYY. MM. DD. HH:MM`` (spacing after the dots is optional).
_ABSOLUTE_RE = re.compile(
    r"^(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{1,2}):(\d{2})$"
)

__all__ = [
    "BUDAPEST",
    "HUNGARIAN_WEEKDAYS",
    "parse_http_date",
    "resolve_timestamp",
    "resolve_http_date",
]


def parse_http_date(value: str | None) -> datetime | None:
    """Parse an HTTP ``Date`` header (RFC 1123) into an aware datetime.

    Returns ``None`` for missing/unparseable values.  Naive results (rare) are
    assumed to be UTC.
    """
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def resolve_timestamp(raw: str | None, now: datetime) -> datetime | None:
    """Resolve one raw timestamp string to a tz-aware Budapest datetime.

    ``now`` is the fetch time (e.g. from :func:`parse_http_date`); it may be in
    any timezone and is converted to Europe/Budapest before extracting the date
    for ``ma`` / ``tegnap``.  Returns ``None`` for unknown/invalid formats.
    """
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None

    relative = _RELATIVE_RE.match(value)
    if relative is not None:
        label, hours, minutes = relative.groups()
        try:
            hour = int(hours)
            minute = int(minutes)
        except ValueError:  # pragma: no cover - regex guarantees digits
            return None
        if hour > 23 or minute > 59:
            return None
        local_now = now.astimezone(BUDAPEST)
        day = local_now.date()
        if label.lower() == "tegnap":
            day = day - timedelta(days=1)
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=BUDAPEST)

    weekday = _WEEKDAY_RE.match(value)
    if weekday is not None:
        label, hours, minutes = weekday.groups()
        hour = int(hours)
        minute = int(minutes)
        if hour > 23 or minute > 59:
            return None
        local_now = now.astimezone(BUDAPEST)
        target = HUNGARIAN_WEEKDAYS[label.lower()]
        # Most recent occurrence of that weekday on or before "now".
        days_back = (local_now.weekday() - target) % 7
        day = local_now.date() - timedelta(days=days_back)
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=BUDAPEST)

    absolute = _ABSOLUTE_RE.match(value)
    if absolute is not None:
        year, month, day, hour, minute = (int(g) for g in absolute.groups())
        try:
            return datetime(year, month, day, hour, minute, tzinfo=BUDAPEST)
        except ValueError:
            return None

    return None


def resolve_http_date(raw: str | None, date_header: str | None) -> datetime | None:
    """Convenience: resolve ``raw`` using an HTTP ``Date`` header string."""
    now = parse_http_date(date_header)
    if now is None:
        return None
    return resolve_timestamp(raw, now)

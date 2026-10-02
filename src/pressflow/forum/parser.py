"""T1.1 — single-page parser for the Portfolio.hu OTP thread.

The forum is server-rendered HTML (``lang="hu"``); one ``?oldal=N`` response
holds ~20 posts (plus a pinned "topic opener" comment which has no timestamp and
is skipped here).  The exact CSS markers are specified in ADR-0004 and the
table below is the contract this module implements.

========================================  ==========================================
Field                                      Marker
========================================  ==========================================
author                                     ``a.username`` text (deleted accounts show
                                           a ``<span>Törölt felhasználó</span>``
                                           fallback instead)
raw timestamp                              ``span.date`` ``title`` attribute
display post id                            ``.id`` div text ``#1340744`` (strip ``#``)
hsz_id                                     vote span ``data-hsz_id`` attribute
user_id                                    vote span ``data-user_id`` attribute
reply-to display id                        "Előzmény" ``a`` anchor text ``#1340742``
reply-to hsz_id                            "Előzmény" ``a`` href ``/prev/5224/<id>``
upvotes / downvotes                        ``span[data-vote="1"/"-1"] span``
text                                       ``div.text > div`` inner text
========================================  ==========================================

Records are returned as plain dicts whose keys match ``schema.POSTS_COLUMNS``
except that ``ts`` is still the raw timestamp *string* — it is resolved to a
tz-aware datetime later (T1.2 / T1.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup, Tag

from pressflow import schema as s

#: ``/prev/<thread>/<hsz_id>`` -> hsz_id (reply-to natural key).
_REPLY_HSZ_RE = re.compile(r"/prev/\d+/(\d+)")

#: ``#1340744`` / ``#18`` -> integer display id.
_HASH_INT_RE = re.compile(r"#?\s*(\d+)")

#: Paired non-post nodes stripped before parsing (T1.7).  Their content is
#: CDATA-like (JS/CSS), so a non-greedy regex is safe and much faster than
#: letting lxml build a tree for ~13 MB of markup that holds no post data.
_PAIRED_NON_CONTENT_RE = re.compile(
    r"<(script|style|noscript)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

#: Void/self-contained non-post nodes (external resources).
_LINK_RE = re.compile(r"<link\b[^>]*>", re.IGNORECASE)

#: HTML comments (never hold post data).
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

#: Inter-tag whitespace (newline-anchored).  Real pages are ~99% indentation
#: whitespace; removing whitespace-only text nodes that contain a newline keeps
#: every rendered field identical (single-space text nodes are left untouched)
#: while shrinking a 13.6 MB page to ~100 KB before parsing (T1.7).
_INTER_TAG_WS_RE = re.compile(r">\s*\n\s*<")

__all__ = [
    "PageParseStats",
    "parse_page",
    "parse_page_detailed",
    "strip_non_content",
    "is_root_post",
    "POST_RECORD_KEYS",
]


#: Ordered keys of a parsed (pre-resolution) post record.
POST_RECORD_KEYS: tuple[str, ...] = s.POSTS_COLUMNS


def _int_from_text(value: str | None) -> int | None:
    """Parse ``#1340744`` (or ``#18``) into an int, else ``None``."""
    if not value:
        return None
    match = _HASH_INT_RE.search(value)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None


def _author(comment: Tag) -> str | None:
    """Return the username, or the site's fallback for deleted accounts.

    Active accounts render ``a.username``; deleted accounts render a plain
    ``<span>Törölt felhasználó</span>`` where the username anchor would be (the
    real markup never uses an ``<i>``).  The ``author`` column is non-nullable
    in ``POSTS_SCHEMA``, so the displayed fallback is used rather than ``None``.
    """
    anchor = comment.select_one("a.username")
    if anchor is not None:
        text = anchor.get_text(strip=True)
        if text:
            return text
    # ``.date`` is the timestamp span; every other span in ``.commentdata`` is
    # the deleted-account label.
    fallback = comment.select_one(".commentdata span:not(.date)")
    if fallback is not None:
        text = fallback.get_text(strip=True)
        if text:
            return text
    return None


def _raw_timestamp(comment: Tag) -> str | None:
    """Raw timestamp from ``span.date`` (``title`` attr preferred)."""
    span = comment.select_one("span.date")
    if span is None:
        return None
    title = span.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    text = span.get_text(strip=True)
    return text or None


def _vote_count(comment: Tag, value: str) -> int:
    """Vote count for ``data-vote`` == ``value`` (``"1"`` / ``"-1"``)."""
    span = comment.select_one(f'span[data-vote="{value}"] span')
    if span is None:
        return 0
    text = span.get_text(strip=True)
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0


def _text(comment: Tag) -> str | None:
    """Post body text, preserving ``<br>`` line breaks."""
    div = comment.select_one("div.text > div")
    if div is None:
        return None
    for br in div.find_all("br"):
        br.replace_with("\n")
    text = div.get_text()
    return text.strip() or None


def _parse_comment(comment: Tag, page: int) -> dict[str, Any] | None:
    """Parse one ``div.comment``; return ``None`` for the pinned topic opener."""
    raw_ts = _raw_timestamp(comment)
    if raw_ts is None:
        # The topic opener (and any non-post widget) has no timestamp.
        return None

    vote = comment.select_one("span.vote[data-hsz_id]")
    if vote is None:
        return None
    try:
        hsz_id = int(vote["data-hsz_id"])
    except (KeyError, TypeError, ValueError):
        return None

    user_id: int | None = None
    if vote.has_attr("data-user_id"):
        try:
            user_id = int(vote["data-user_id"])
        except (TypeError, ValueError):
            user_id = None

    reply_anchor = comment.select_one("div.prevcomment a")
    reply_to_post_id: int | None = None
    reply_to_hsz_id: int | None = None
    if reply_anchor is not None:
        reply_to_post_id = _int_from_text(reply_anchor.get_text(strip=True))
        href = reply_anchor.get("href")
        if isinstance(href, str):
            match = _REPLY_HSZ_RE.search(href)
            if match is not None:
                reply_to_hsz_id = int(match.group(1))

    id_el = comment.select_one(".id")
    return {
        s.POST_ID: _int_from_text(id_el.get_text() if id_el is not None else None),
        s.HSZ_ID: hsz_id,
        s.AUTHOR: _author(comment),
        s.USER_ID: user_id,
        s.TS: raw_ts,
        s.REPLY_TO_POST_ID: reply_to_post_id,
        s.REPLY_TO_HSZ_ID: reply_to_hsz_id,
        s.UPVOTES: _vote_count(comment, "1"),
        s.DOWNVOTES: _vote_count(comment, "-1"),
        s.TEXT: _text(comment),
        s.PAGE: page,
    }


def strip_non_content(html: str) -> str:
    """Remove non-post markup before BeautifulSoup/lxml parses the page (T1.7).

    Strips paired ``<script>`` / ``<style>`` / ``<noscript>`` blocks plus void
    ``<link>`` tags and HTML comments, then collapses inter-tag whitespace.
    All post fields live in ``div.comment`` nodes, so extraction is unchanged
    while parse time drops from ~1.4 s to <0.3 s on a real 13.6 MB page.
    """
    html = _PAIRED_NON_CONTENT_RE.sub("", html)
    html = _LINK_RE.sub("", html)
    html = _HTML_COMMENT_RE.sub("", html)
    html = _INTER_TAG_WS_RE.sub("><", html)
    return html


@dataclass
class PageParseStats:
    """Pre-write skip counters for one parsed page (T1.6 issue 5)."""

    comments_seen: int = 0
    skipped_no_date: int = 0
    skipped_no_key: int = 0


def parse_page_detailed(
    html: str, page: int
) -> tuple[list[dict[str, Any]], PageParseStats]:
    """Like :func:`parse_page` but also reports skipped-comment counts.

    Comments without a timestamp (the pinned topic opener/widgets) and comments
    without a vote span / ``hsz_id`` natural key are dropped; the counters let
    T1.5 surface how many records never reach the Parquet write.
    """
    soup = BeautifulSoup(strip_non_content(html), "lxml")
    stats = PageParseStats()
    records: list[dict[str, Any]] = []
    for comment in soup.select("div.comment"):
        stats.comments_seen += 1
        if _raw_timestamp(comment) is None:
            stats.skipped_no_date += 1
            continue
        record = _parse_comment(comment, page)
        if record is None or record[s.HSZ_ID] is None:
            stats.skipped_no_key += 1
            continue
        records.append(record)
    return records, stats


def parse_page(html: str, page: int) -> list[dict[str, Any]]:
    """Parse a single ``?oldal=`` response into per-post records.

    Posts are returned in document order (newest → oldest on the live site).
    The pinned topic opener and any comment without a natural key are skipped.
    """
    return parse_page_detailed(html, page)[0]


def is_root_post(record: dict[str, Any]) -> bool:
    """``True`` when the post is not a reply (no "Előzmény" link)."""
    return record.get(s.REPLY_TO_HSZ_ID) is None

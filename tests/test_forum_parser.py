"""T1.1 — single-page parser tests (fixture + synthetic edge cases)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pressflow import schema as s
from pressflow.forum.parser import is_root_post, parse_page

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "forum_page.html"
FIXTURE_PAGE = 67047  # oldal the fixture was fetched from


@pytest.fixture(scope="module")
def fixture_records() -> list[dict]:
    return parse_page(FIXTURE.read_text(encoding="utf-8"), FIXTURE_PAGE)


def test_fixture_returns_about_a_page_of_records(fixture_records):
    # ADR-0004: ~20-24 posts per page (the pinned topic opener is skipped).
    assert 20 <= len(fixture_records) <= 24


def test_fixture_records_have_canonical_keys(fixture_records):
    for record in fixture_records:
        assert set(record) == set(s.POSTS_COLUMNS)


def test_fixture_required_fields_present(fixture_records):
    """Only reply-to fields may be null; everything else is present."""
    for record in fixture_records:
        for column in s.POSTS_COLUMNS:
            if column in (s.REPLY_TO_POST_ID, s.REPLY_TO_HSZ_ID):
                continue
            assert record[column] is not None, column


def test_fixture_types_and_page(fixture_records):
    for record in fixture_records:
        assert isinstance(record[s.POST_ID], int)
        assert isinstance(record[s.HSZ_ID], int)
        assert isinstance(record[s.UPVOTES], int)
        assert isinstance(record[s.DOWNVOTES], int)
        assert isinstance(record[s.USER_ID], int)
        assert isinstance(record[s.AUTHOR], str)
        assert isinstance(record[s.TEXT], str)
        assert record[s.PAGE] == FIXTURE_PAGE


def test_fixture_text_non_empty_rate(fixture_records):
    non_empty = sum(1 for r in fixture_records if r[s.TEXT])
    assert non_empty / len(fixture_records) >= 0.95


def test_fixture_has_replies_and_votes(fixture_records):
    assert any(not is_root_post(r) for r in fixture_records)
    assert any(r[s.UPVOTES] > 0 or r[s.DOWNVOTES] > 0 for r in fixture_records)


def test_fixture_timestamps_are_raw_strings(fixture_records):
    for record in fixture_records:
        assert isinstance(record[s.TS], str)
        assert record[s.TS]


def test_reply_hsz_and_display_id_align(fixture_records):
    replies = [r for r in fixture_records if not is_root_post(r)]
    assert replies
    for record in replies:
        assert isinstance(record[s.REPLY_TO_HSZ_ID], int)
        assert isinstance(record[s.REPLY_TO_POST_ID], int)


def test_deleted_user_fallback_used():
    """Deleted accounts have no ``a.username``; the displayed label is used."""
    html = """
    <div class="comment">
      <div class="commentdata"><i>Törölt felhasználó</i>
        <span class="date" title="2006. 07. 24. 15:14">2006. 07. 24. 15:14</span>
      </div>
      <div class="prevcomment"></div>
      <div class="id">#20</div>
      <div class="text"><div>szia<br />világ</div></div>
      <span class="vote" data-hsz_id="205384" data-user_id="23795" data-vote="1"><span>3</span></span>
      <span class="vote" data-hsz_id="205384" data-user_id="23795" data-vote="-1"><span>1</span></span>
    </div>
    """
    records = parse_page(html, 1)
    assert len(records) == 1
    assert records[0][s.AUTHOR] == "Törölt felhasználó"
    # <br> becomes a line break
    assert records[0][s.TEXT] == "szia\nvilág"


def test_topic_opener_without_date_is_skipped():
    html = """
    <div class="comment mt-0">
      <div id="firstcomment">akik otp részvénnyel kereskednek</div>
    </div>
    """
    assert parse_page(html, 1) == []


def test_missing_votes_default_to_zero():
    html = """
    <div class="comment">
      <div class="commentdata"><a class="username">bob</a>
        <span class="date" title="2010. 11. 25. 10:17">2010. 11. 25. 10:17</span>
      </div>
      <div class="id">#5</div>
      <div class="text"><div>hello</div></div>
      <span class="vote" data-hsz_id="42" data-user_id="7" data-vote="1"><span></span></span>
    </div>
    """
    records = parse_page(html, 9)
    assert len(records) == 1
    assert records[0][s.UPVOTES] == 0
    assert records[0][s.DOWNVOTES] == 0
    assert records[0][s.REPLY_TO_POST_ID] is None
    assert records[0][s.REPLY_TO_HSZ_ID] is None
    assert records[0][s.PAGE] == 9

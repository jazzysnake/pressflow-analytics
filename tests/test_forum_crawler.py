"""T1.4 — resumable crawler tests (fake client, no network)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from pressflow import schema as s
from pressflow.forum.crawler import CrawlerPaths, crawl
from pressflow.forum.http import FetchError, PageResponse
from pressflow.forum.timestamps import BUDAPEST, parse_http_date

SERVER_DATE = "Fri, 02 Oct 2026 18:24:11 GMT"
MAX_OLDAL = 5
POSTS_PER_PAGE = 4


def _make_page_html(hszs: list[int]) -> str:
    comments = []
    for hsz in hszs:
        comments.append(
            f"""
            <div class="comment">
              <div class="commentdata"><a class="username">user{hsz}</a>
                <span class="date" title="2026. 08. 17. 10:00">2026. 08. 17. 10:00</span>
              </div>
              <div class="prevcomment"></div>
              <div class="id">#{hsz}</div>
              <div class="text"><div>post body {hsz}</div></div>
              <span class="vote" data-hsz_id="{hsz}" data-user_id="{hsz % 1000}"
                    data-vote="1"><span>1</span></span>
              <span class="vote" data-hsz_id="{hsz}" data-user_id="{hsz % 1000}"
                    data-vote="-1"><span>0</span></span>
            </div>
            """
        )
    return "<html><body>" + "".join(comments) + "</body></html>"


def _pagination(max_linked: int) -> str:
    return "".join(
        f'<a href="https://forum.portfolio.hu/topics/x/5224?oldal={n}">{n}</a>'
        for n in range(max_linked, 0, -1)
    )


class FakeClient:
    """Serves preset pages; ``None`` addresses the newest page (base)."""

    def __init__(self, pages: dict[int | None, str]) -> None:
        self.pages = pages
        self.calls: list[int | None] = []
        self.server_date = parse_http_date(SERVER_DATE)

    def fetch(self, oldal: int | None) -> PageResponse:
        self.calls.append(oldal)
        if oldal not in self.pages:
            raise FetchError(f"no page for oldal={oldal}")
        return PageResponse(
            oldal=oldal,
            url=f"https://forum.portfolio.hu/topics/x/5224?oldal={oldal}",
            status=200,
            html=self.pages[oldal],
            server_date=self.server_date,
        )


def _hszs(oldal: int) -> list[int]:
    return [oldal * 100 + i for i in range(POSTS_PER_PAGE)]


def _build_pages(overlap: bool = False) -> dict[int | None, str]:
    pages: dict[int | None, str] = {}
    for oldal in range(1, MAX_OLDAL + 1):
        hs = _hszs(oldal)
        body = _make_page_html(hs)
        if oldal == MAX_OLDAL:
            # newest page includes pagination links to 4..1
            body = body + _pagination(MAX_OLDAL - 1)
            pages[None] = body
        pages[oldal] = body
    if overlap:
        # page 4 repeats the last hsz of the newest page
        shared = _hszs(MAX_OLDAL)[-1]
        pages[4] = _make_page_html(_hszs(4)[:-1] + [shared])
    return pages


def _paths(tmp_path: Path) -> CrawlerPaths:
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    return CrawlerPaths(
        raw_dir=raw,
        processed_dir=processed,
        checkpoint=raw / "crawl_checkpoint.json",
        parts_dir=raw / "parts",
        posts=processed / "posts.parquet",
    )


def test_crawl_writes_schema_valid_parquet(tmp_path):
    client = FakeClient(_build_pages())
    paths = _paths(tmp_path)
    stats = crawl(
        client=client, paths=paths, fresh=True, max_pages=100, batch_pages=2
    )

    assert paths.posts.is_file()
    table = pq.read_table(paths.posts)
    assert table.schema.equals(s.POSTS_SCHEMA)
    assert table.num_rows == MAX_OLDAL * POSTS_PER_PAGE
    assert stats.posts_written == MAX_OLDAL * POSTS_PER_PAGE
    assert stats.duplicates == 0
    # base + pages 4..1 = 5 logical pages, each fetched once.
    assert client.calls == [None, 4, 3, 2, 1]
    assert len(client.calls) == len(set(client.calls))


def test_crawl_dedupes_on_hsz_id(tmp_path):
    client = FakeClient(_build_pages(overlap=True))
    paths = _paths(tmp_path)
    stats = crawl(client=client, paths=paths, fresh=True, max_pages=100, batch_pages=2)

    assert stats.duplicates == 1
    table = pq.read_table(paths.posts)
    hszs = table.column(s.HSZ_ID).to_pylist()
    assert len(hszs) == len(set(hszs))
    assert table.num_rows == MAX_OLDAL * POSTS_PER_PAGE - 1


def test_crawl_resumes_without_gaps_or_duplicates(tmp_path):
    paths = _paths(tmp_path)

    # First run: newest page + oldal 4 only, then "interrupt".
    client1 = FakeClient(_build_pages())
    stats1 = crawl(client=client1, paths=paths, fresh=True, max_pages=2, batch_pages=50)
    assert client1.calls == [None, 4]
    checkpoint = paths.checkpoint.read_text(encoding="utf-8")
    assert '"next_oldal": 3' in checkpoint

    # Second run resumes from the checkpoint.
    client2 = FakeClient(_build_pages())
    stats2 = crawl(client=client2, paths=paths, max_pages=100, batch_pages=50)
    assert client2.calls == [3, 2, 1]

    # No page fetched more than once across the whole run.
    all_calls = client1.calls + client2.calls
    assert len(all_calls) == len(set(all_calls))

    table = pq.read_table(paths.posts)
    assert table.num_rows == MAX_OLDAL * POSTS_PER_PAGE
    hszs = table.column(s.HSZ_ID).to_pylist()
    assert len(hszs) == len(set(hszs))
    assert stats1.posts_written + stats2.posts_written == MAX_OLDAL * POSTS_PER_PAGE


def test_crawl_skips_permanently_failing_page(tmp_path):
    pages = _build_pages()
    del pages[3]
    # Remove page 3 so the fake client raises FetchError.
    client = FakeClient(pages)
    paths = _paths(tmp_path)
    stats = crawl(client=client, paths=paths, fresh=True, max_pages=100)

    assert stats.pages_skipped == 1
    assert stats.failures and stats.failures[0]["oldal"] == 3
    assert stats.pages_fetched == 4  # base, 4, 2, 1
    assert paths.posts.is_file()


def _make_page_html_dated(hszs: list[int], title: str) -> str:
    comments = []
    for hsz in hszs:
        comments.append(
            f'''
            <div class="comment">
              <div class="commentdata"><a class="username">user{hsz}</a>
                <span class="date" title="{title}">{title}</span>
              </div>
              <div class="prevcomment"></div>
              <div class="id">#{hsz}</div>
              <div class="text"><div>post body {hsz}</div></div>
              <span class="vote" data-hsz_id="{hsz}" data-user_id="{hsz % 1000}"
                    data-vote="1"><span>1</span></span>
              <span class="vote" data-hsz_id="{hsz}" data-user_id="{hsz % 1000}"
                    data-vote="-1"><span>0</span></span>
            </div>
            '''
        )
    return "<html><body>" + "".join(comments) + "</body></html>"


def _build_dated_pages() -> dict[int | None, str]:
    """Page ``oldal`` has newest timestamp ``2026-10-01 + oldal`` days."""
    base = datetime(2026, 10, 1, tzinfo=BUDAPEST)
    pages: dict[int | None, str] = {}
    for oldal in range(1, MAX_OLDAL + 1):
        title = (base + timedelta(days=oldal)).strftime("%Y. %m. %d. 10:00")
        body = _make_page_html_dated(_hszs(oldal), title)
        if oldal == MAX_OLDAL:
            body = body + _pagination(MAX_OLDAL - 1)
            pages[None] = body
        pages[oldal] = body
    return pages


def test_target_date_reuses_probe_pages(tmp_path):
    """T1.6 issue 2: binary-search probes are not re-fetched by the crawl."""
    client = FakeClient(_build_dated_pages())
    paths = _paths(tmp_path)
    # Page 3's newest post -> start_oldal == 3.
    target = datetime(2026, 10, 4, tzinfo=BUDAPEST)
    stats = crawl(
        client=client,
        paths=paths,
        fresh=True,
        target_date=target,
        max_pages=100,
        batch_pages=50,
    )

    assert len(client.calls) == len(set(client.calls))
    assert client.calls.count(3) == 1  # probed once, reused by the crawl loop
    assert stats.pages_reused >= 1
    table = pq.read_table(paths.posts)
    assert table.num_rows == 3 * POSTS_PER_PAGE  # pages max, max-1, start


def test_fresh_clears_stale_parts(tmp_path):
    """T1.6 issue 4: ``--fresh`` removes stale part files before writing."""
    paths = _paths(tmp_path)
    paths.parts_dir.mkdir(parents=True, exist_ok=True)
    stale = paths.parts_dir / "part-09999.parquet"
    stale.write_bytes(b"stale")

    client = FakeClient(_build_pages())
    crawl(client=client, paths=paths, fresh=True, max_pages=1, batch_pages=1)
    assert not stale.exists()


def test_crawl_counts_skipped_comments(tmp_path):
    """T1.6 issue 5: parser skip counters are accumulated for validation."""
    base = _make_page_html(_hszs(2))
    base += '''
      <div class="comment">
        <div class="commentdata"><a class="username">ghost</a>
          <span class="date" title="2026. 08. 17. 10:00">d</span>
        </div>
        <div class="id">#99999</div>
        <div class="text"><div>no vote span</div></div>
      </div>
      <div class="comment mt-0"><div id="firstcomment">opener</div></div>
    ''' + _pagination(1)
    pages: dict[int | None, str] = {None: base, 1: _make_page_html(_hszs(1))}

    stats = crawl(
        client=FakeClient(pages), paths=_paths(tmp_path), fresh=True, batch_pages=10
    )

    assert stats.comments_seen == 10  # 6 on the base page, 4 on page 1
    assert stats.comments_skipped_no_date == 1  # topic opener
    assert stats.comments_skipped_no_key == 1  # missing vote span
    assert stats.posts_seen == 8


def test_compact_dedupes_parts(tmp_path):
    from pressflow.forum.crawler import _write_part, compact_parts

    parts = tmp_path / "parts"
    record = {
        s.POST_ID: 1,
        s.HSZ_ID: 999,
        s.AUTHOR: "a",
        s.USER_ID: 1,
        s.TS: parse_http_date(SERVER_DATE),
        s.REPLY_TO_POST_ID: None,
        s.REPLY_TO_HSZ_ID: None,
        s.UPVOTES: 0,
        s.DOWNVOTES: 0,
        s.TEXT: "x",
        s.PAGE: 1,
    }
    _write_part(parts, 1, [record])
    _write_part(parts, 2, [dict(record, text="y")])
    rows = compact_parts(parts, tmp_path / "posts.parquet")
    assert rows == 1
    table = pq.read_table(tmp_path / "posts.parquet")
    assert table.schema.equals(s.POSTS_SCHEMA)
    assert table.column(s.TEXT).to_pylist() == ["y"]  # keep last

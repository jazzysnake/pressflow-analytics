"""T1.4 — polite HTTP client tests (fake session, injected sleeper)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pressflow.forum.http import FetchError, ForumClient


class FakeResponse:
    def __init__(self, status_code=200, text="<html></html>", headers=None, url="http://x"):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.url = url


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[str] = []

    def get(self, url, timeout=None, allow_redirects=True):
        self.calls.append(url)
        result = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


def _cfg(delay=0.35, retries=3):
    return SimpleNamespace(
        forum=SimpleNamespace(base_url="https://forum.example/thread"),
        scraping=SimpleNamespace(delay_sec=delay, retries=retries),
    )


def _client(session, cfg=None, sleeps=None):
    recorded = sleeps if sleeps is not None else []
    return ForumClient(
        cfg or _cfg(), session=session, sleeper=recorded.append
    ), recorded


def test_fetch_success_parses_date_header():
    session = FakeSession(
        [FakeResponse(200, "<html>ok</html>", {"Date": "Fri, 02 Oct 2026 18:24:11 GMT"})]
    )
    client, _ = _client(session)
    response = client.fetch(42)

    assert response.status == 200
    assert response.html == "<html>ok</html>"
    assert response.server_date is not None
    assert session.calls == ["https://forum.example/thread?oldal=42"]


def test_fetch_retries_then_succeeds_with_backoff():
    session = FakeSession(
        [FakeResponse(500), FakeResponse(503), FakeResponse(200, "done")]
    )
    client, sleeps = _client(session, _cfg(retries=3))
    response = client.fetch(1)

    assert response.html == "done"
    assert len(session.calls) == 3
    # first request: no pre-sleep; then 2 backoffs (1s, 2s) before retries.
    assert sleeps == [1.0, 2.0]


def test_fetch_exhausts_retries_and_raises():
    session = FakeSession([FakeResponse(500)])
    client, _ = _client(session, _cfg(retries=2))
    with pytest.raises(FetchError):
        client.fetch(7)
    assert len(session.calls) == 3  # initial + 2 retries


def test_fetch_retries_on_request_exception():
    import requests

    session = FakeSession(
        [requests.ConnectionError("boom"), FakeResponse(200, "ok")]
    )
    client, _ = _client(session, _cfg(retries=3))
    assert client.fetch(1).html == "ok"
    assert len(session.calls) == 2


def test_politeness_delay_applied_between_requests():
    session = FakeSession([FakeResponse(200), FakeResponse(200)])
    client, sleeps = _client(session, _cfg(delay=0.4, retries=0))
    client.fetch(None)
    client.fetch(None)

    assert sleeps == [0.4]
    assert session.calls == [
        "https://forum.example/thread",
        "https://forum.example/thread",
    ]


def test_default_retries_from_config():
    session = FakeSession([FakeResponse(500)])
    client, _ = _client(session, _cfg(retries=3))
    with pytest.raises(FetchError):
        client.fetch(1)
    assert len(session.calls) == 4

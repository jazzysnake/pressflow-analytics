"""Shared polite HTTP client for the forum crawler (T1.4, ADR-0004/0015).

Wraps a ``requests.Session`` with the project User-Agent / Accept-Language, a
request timeout, and retry-with-exponential-backoff.  The politeness delay
(``scraping.delay_sec``, ≥ 0.35 s) is applied before every request after the
first.  The HTTP ``Date`` header is surfaced so relative timestamps can be
resolved against the fetch time (T1.2).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import requests

from pressflow.config import Config
from pressflow.forum.timestamps import parse_http_date

#: Request timeout in seconds (ADR-0004 does not pin one; 30 s is polite).
DEFAULT_TIMEOUT_SEC = 30.0

#: Base backoff between retries; doubles each attempt.
BACKOFF_BASE_SEC = 1.0

#: Identifies the project to the forum (ADR-0004: set UA + Accept-Language).
USER_AGENT = (
    "pressflow-analytics/0.1 (research; polite scraper; "
    "contact: https://github.com/pressflow-analytics)"
)

#: Hungarian-first content negotiation.
ACCEPT_LANGUAGE = "hu-HU,hu;q=0.9,en;q=0.8"

__all__ = [
    "DEFAULT_TIMEOUT_SEC",
    "BACKOFF_BASE_SEC",
    "USER_AGENT",
    "ACCEPT_LANGUAGE",
    "FetchError",
    "PageResponse",
    "build_session",
    "ForumClient",
]


class FetchError(RuntimeError):
    """Raised when a page cannot be fetched after all retries."""


@dataclass
class PageResponse:
    """A successfully fetched forum page."""

    oldal: int | None
    url: str
    status: int
    html: str
    server_date: datetime | None


def build_session() -> requests.Session:
    """Create a ``requests.Session`` with the project's polite headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": ACCEPT_LANGUAGE,
        }
    )
    return session


class ForumClient:
    """Polite, retrying forum fetcher.

    Parameters
    ----------
    cfg:
        Validated config; supplies ``forum.base_url``,
        ``scraping.delay_sec`` and ``scraping.retries``.
    session:
        Optional pre-built session (tests inject a fake).
    timeout:
        Per-request timeout in seconds.
    sleeper:
        Injectable sleep function (tests pass a no-op recorder).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = cfg.forum.base_url
        self.delay_sec = cfg.scraping.delay_sec
        self.retries = cfg.scraping.retries
        self.timeout = timeout
        self.session = session if session is not None else build_session()
        self._sleeper = sleeper
        self._first_request = True

    def url_for(self, oldal: int | None) -> str:
        """Build the thread URL for ``oldal`` (``None`` = newest page)."""
        if oldal is None:
            return self.base_url
        return f"{self.base_url}?oldal={oldal}"

    def _sleep_before(self) -> None:
        if not self._first_request:
            self._sleeper(self.delay_sec)

    def fetch(self, oldal: int | None) -> PageResponse:
        """Fetch one page, retrying with exponential backoff.

        Raises :class:`FetchError` after ``scraping.retries`` failed attempts.
        """
        url = self.url_for(oldal)
        self._sleep_before()
        self._first_request = False
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    url, timeout=self.timeout, allow_redirects=True
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    return PageResponse(
                        oldal=oldal,
                        url=str(getattr(response, "url", url)),
                        status=response.status_code,
                        html=response.text,
                        server_date=parse_http_date(response.headers.get("Date")),
                    )
                last_error = FetchError(f"HTTP {response.status_code} for {url}")

            if attempt < self.retries:
                self._sleeper(BACKOFF_BASE_SEC * (2**attempt))

        raise FetchError(f"failed to fetch {url}: {last_error}")

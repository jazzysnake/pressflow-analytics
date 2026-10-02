"""Forum scraper for the Portfolio.hu OTP thread (Epic 01).

Modules
-------
parser
    T1.1 — parse one saved HTML page into per-post records.
timestamps
    T1.2 — resolve the forum's three raw timestamp forms to Budapest time.
pagination
    T1.3 — discover the newest page and binary-search a target date.
http
    Shared polite HTTP client (retries + backoff) used by the crawler.
crawler
    T1.4 — resumable newest→oldest crawler writing Parquet.
validation
    T1.5 — scrape validation report over the posts Parquet.
"""

from __future__ import annotations

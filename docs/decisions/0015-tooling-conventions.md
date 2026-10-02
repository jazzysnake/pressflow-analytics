# 0015 — Tooling & conventions

- **Status:** Accepted
- **Date:** 2026-10-01

## Decision

- **uv** for all dependencies; **Python 3.12** (`>=3.12,<3.13`).
- **Parquet** for all data artifacts; `data/` gitignored.
- Package `src/pressflow/`; notebooks for exploration; `tests/` with pytest.
- Config-driven (single `config/default.yaml`, validated — no hardcoded params).
- Timezone: **Europe/Budapest** for all post timestamps and trading days.
- Scraping politeness: ~0.35s delay, retry+backoff, checkpoint/resume.
- Schemas defined once (`docs/schema.md` + code constants).

## Consequences

- Juniors can implement tickets independently; consistent data contracts
  between stages.

## References

- Tickets T0.1–T0.3

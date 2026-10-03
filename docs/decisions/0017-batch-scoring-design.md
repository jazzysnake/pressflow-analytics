# 0017 — Batch scoring design (Jev API)

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

Scoring ~277k posts through a paid, rate-limited, network-dependent API
(ADR-0016). Naive serial requests would take days; a crash mid-run would
re-spend paid tokens; a transient network blip must not kill the run.

## Decision

- **Concurrency**: ThreadPoolExecutor, 32 workers (configurable), ~0.2s
  per-request delay; newest→oldest order.
- **Resumability**: checkpoint sidecar of completed `hsz_id`s (JSON) plus
  read-back of the scores Parquet; flush every 500 completions and on
  shutdown; resume skips already-scored rows.
- **Failure tolerance**: per-post retry (exponential backoff, 3 attempts) on
  network errors / 429 / 5xx; a post that still fails goes to a
  `failed_hsz_ids` side file and the run continues; the run aborts only after
  `max_consecutive_failures = 10` failures in a row (any success resets the
  counter).
- **Cost safety**: accumulate `usage` token counts, estimate cost at
  $0.042/1M input (output assumed equal), log throughput/ETA, hard-stop at a
  $9 cap against a $10 balance.

## Consequences

- Full run ~2–3h at concurrency 32; ~$4–6 cost; resumable across disconnects;
  no silent overspend.
- `failed_hsz_ids` may leave a small tail to retry separately.

## References

- ADR-0016, T3.4, T0.3 (scores schema)

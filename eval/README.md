# `eval/` — sentiment eval set (T3.1)

Tracked, hand-checkable artifacts for selecting an off-the-shelf Hungarian
sentiment model (ADR-0001). No scraped data lives here — only small, reviewed
files.

## Files

| File | Purpose |
|---|---|
| `sample.csv` | The fixed, reproducible sample to be labeled by both models. Columns: `hsz_id, post_id, ts, text`. ~200 posts spread over the 10-year window. |
| `sampling_log.csv` | How each page was chosen. Columns: `target_date, oldal, n_posts` — one row per target date (plus a few gap-fill rows, see below). |
| `labels_modelA.csv` | Labels from **Model A** (implementer). Columns: `hsz_id, label, sarcasm, panic, note`. `label` ∈ {`bull`, `bear`, `neutral`}; `sarcasm`/`panic` are `true`/`false`. |
| `labels_modelB.csv` | Labels from **Model B** (independent, different model; must not read Model A's output). Same schema. *(not yet present)* |
| `eval_set.csv` | Consensus subset — rows where Models A and B agree on `label` (to be produced after Model B runs). |

## Sample construction

`src/pressflow/eval_sampling.py` builds `sample.csv`:

1. `sampling_start = 2016-10-01`, `sampling_end = 2026-10-01`; 20 evenly-spaced
   target dates (~every 6 months), 10 posts each.
2. The newest page is fetched once to discover `max_oldal`; the first target
   date is located with the existing pagination binary search
   (`pressflow.forum.pagination.find_start_oldal`, T1.3). Its ~16 probe pages
   are cached and reused as calibration anchors.
3. Each remaining target picks its page by piecewise-linear interpolation in the
   `date → oldal` plane over the accumulated `(newest_ts, oldal)` anchors.
4. Because page density is highly non-uniform around 2020–2022
   (~1.0k–5.0k pages/year), any date gap wider than 8 months is split by
   fetching the page-bracket midpoint (monotone in time), adding a few posts.
   This guarantees the no->9-month-gap requirement.
5. Posts are written sorted by `ts`; politeness uses the shared
   `ForumClient` delay (0.35 s). A full run fetches ~40 pages.

Re-run with:

```bash
uv run python -m pressflow.eval_sampling
```

## Result of the committed run

- 220 posts spanning `2016-09-30` → `2026-10-01`; largest gap **224 days**
  (well under 9 months).
- Label distribution (Model A): **bull 32 · bear 35 · neutral 153**;
  sarcasm 12, panic 1.

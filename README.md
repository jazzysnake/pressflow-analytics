# pressflow-analytics

> **Can retail stock-forum sentiment predict stock returns?** We scraped 10 years
> of Hungarian stock-forum posts, scored them for sentiment, aggregated the crowd
> into a daily signal, and backtested a long-only strategy against the S&P 500 —
> **with a pre-registered kill decision.** The honest answer: the signal is real
> but too weak to trade.

**Verdict: 🔴 KILL** — a statistically significant but economically worthless
edge. Full reasoning in [`docs/reports/go-nogo.md`](docs/reports/go-nogo.md).

---

## TL;DR

| Stage | Result |
|---|---|
| Data | **277,217 posts** scraped from the Portfolio.hu OTP thread (2016–2026) |
| Sentiment | Off-the-shelf LLM (Jev), **$5.33** to score the whole corpus |
| Signal quality | rank-IC **+0.06** vs next-day return — real, but weak |
| Calibration | Brier **0.2497** ≈ coin-flip climatology **0.2493** |
| Backtest | Strategy **−0.01%** vs **SPY +363%** / **OTP +731%** over 10 years |

The crowd signal cleared the *statistical* gate (it genuinely correlates with
next-day direction) and died at the *economics*: it never got confident enough
to justify taking a position.

## The question

Portfolio.hu hosts one of the most active Hungarian retail-investor forums — the
OTP thread alone has ~1.3M posts. The hypothesis: **crowd sentiment carries
tradeable information about next-day price direction.** We tested it with an
exploration-first discipline: kill criteria agreed *before* looking
([ADR-0014](docs/decisions/0014-kill-criteria.md)), a hard gate before any
backtest, and an explicit go/no-go memo at the end.

## The pipeline

```
posts ──► scores ──┐
                   ├──► signals ──► calibration ──► backtest ──► kill decision
         prices ───┘
```

1. **Scrape** — 16,382 forum pages → 277k posts, polite + resumable crawler
   ([Epic 01](docs/tickets/epic-01-forum-scraper/)).
2. **Score** — per-post bull/bear/neutral probabilities. Off-the-shelf models
   were near-chance on Hungarian (HF `hts5` dir-F1 0.23); an LLM API (Jev) hit
   dir-F1 **0.57** on a two-model-consensus eval set
   ([`model-selection.md`](docs/reports/model-selection.md)).
3. **Aggregate** — daily engagement-weighted sentiment, disagreement, rolling
   z-score ([ADR-0003](docs/decisions/0003-statistical-method.md)).
4. **Gate** — Spearman rank-IC vs next-day return, both momentum/contrarian
   directions ([`rank-ic.md`](docs/reports/rank-ic.md)).
5. **Calibrate** — walk-forward logistic → P(next-day up)
   ([`calibration.md`](docs/reports/calibration.md)).
6. **Backtest** — long-only fractional Kelly, costs on turnover
   ([`backtest.md`](docs/reports/backtest.md)).
7. **Decide** — the four pre-registered kill criteria
   ([`go-nogo.md`](docs/reports/go-nogo.md)).

## The findings

### The signal is real, but tiny

The best daily signal — engagement-weighted mean sentiment — had a next-day
rank-IC of **+0.060** (momentum direction), significant but with a bootstrap
confidence interval of **[+0.022, +0.096]** — a marginal pass of the first
gate.

### It does *not* lead big moves

A lead-lag analysis ([`lead-lag-analysis.md`](docs/reports/lead-lag-analysis.md))
showed sentiment mostly **co-moves** with price (same-day corr +0.34) and
**reacts** to it (lag corr +0.15), with only a weak 1-day lead (+0.054). The
apparent multi-day "lead" was a 30-day moving-average artifact
(autocorrelation 0.989), not prediction.

### It can't be traded

The calibrated probability never left **[0.48, 0.58]** against a 53% base rate,
so Kelly sizing never invested more than **2%** of capital. The strategy sat
essentially in cash for ten years:

| | Total return | Sharpe |
|---|---|---|
| **Sentiment strategy** | **−0.01%** | **−0.005** |
| SPY buy-and-hold | +363% | +0.88 |
| OTP buy-and-hold | +731% | +0.75 |

### The verdict

| Kill criterion (ADR-0014) | Result |
|---|---|
| \|rank-IC\| < 0.05 | ✗ didn't fire (0.060) |
| **After-cost Sharpe ≤ SPY Sharpe** | ✅ **FIRES** (−0.005 vs +0.88) |
| Block-bootstrap p > 0.10 | moot (strategy ≈ 0%) |
| Drawdown/turnover untradeable | moot (no meaningful position) |

**KILL.** Retail forum sentiment, aggregated this way, is not tradeable alpha.

## Why this is still a useful result

Negative results are cheap to ignore and expensive to skip. For **$5.33** of
inference and a few hours of compute, we answered a question that could
otherwise justify building (and losing money on) a real system. The repo is a
worked example of:

- pre-registered kill criteria *before* any backtest;
- a first hard gate (rank-IC) that stops work before wasted effort;
- honest calibration (the model admitted it was coin-flip);
- lead-lag analysis that caught a visual illusion (smoothing ≠ prediction);
- every ticket implemented on a branch, independently verified, then merged.

## Repository layout

- [`docs/decisions/`](docs/decisions/) — 18 architecture decision records
- [`docs/tickets/`](docs/tickets/) — implementation tickets, one file per task
- [`docs/reports/`](docs/reports/) — the actual results (eval, QA, rank-IC,
  calibration, backtest, kill memo, lead-lag)
- [`docs/schema.md`](docs/schema.md) — canonical Parquet contracts
- [`src/pressflow/`](src/pressflow/) — the pipeline code
- [`tests/`](tests/) — 348 tests

## Setup & reproduce

Python 3.12 via [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # core dependencies
uv sync --extra ml      # optional ML stack (sklearn for calibration)
uv run pytest           # 348 tests
```

The data are gitignored (`data/`), so reproduction starts with the crawler:

```bash
uv run python -m pressflow.forum.crawler --pages N        # scrape posts
uv run python -m pressflow.batch_score                     # sentiment-score (needs TYPESAFE_API_KEY)
uv run python -m pressflow.aggregate                       # daily signals
uv run python -m pressflow.join_prices                     # signal→return panel
uv run python -m pressflow.calibrate                       # walk-forward P(up)
uv run python -m pressflow.backtest_run                    # backtest + metrics report
```

## Disclaimer

This is a research/educational project, not financial advice. The honest
conclusion of the analysis is that this signal is **not** tradeable.

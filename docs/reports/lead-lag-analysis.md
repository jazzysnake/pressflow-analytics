# Lead–lag analysis: does forum sentiment lead price?

**Date:** 2026-10-03
**Scope:** exploratory timing check on the scored OTP-thread sentiment
(`signals.parquet`, `panel.parquet`, `prices.parquet`), run after the T4.3
rank-IC gate.

## Question

Eyeballing the z-scored overlay, sentiment *looked* like it led large price
moves. Is that real, and over what horizon?

## Data & plots

- `docs/reports/sentiment_vs_price.png` — 3-panel: price / `score_wmean` /
  `z_score` (30-day rolling overlays).
- `docs/reports/sentiment_vs_price_normalized.png` — min-max [0,1] overlay.
- `docs/reports/sentiment_vs_price_zscore.png` — z-scored overlay (the better
  visual; min-max flattens sentiment because its range is set by outlier
  single-post days).
- `docs/reports/sentiment_lead_lag.png` — cross-correlogram, raw daily
  sentiment vs daily return.

## Findings

### 1. Sentiment co-moves and reacts — it does not lead

Cross-correlogram `corr(sentiment[t], return[t+lag])` on **raw daily**
`score_wmean`:

| lag | meaning | corr |
|---|---|---|
| 0 | same-day co-movement | **+0.339** |
| −1 | sentiment lags (reacts to yesterday) | +0.150 |
| −2 … −4 | decaying reaction | +0.06 → +0.05 |
| +1 | sentiment leads | **+0.054** |
| +2 … +15 | lead side | ≈ noise (≤ 0.04, sign-flipping) |

The correlation is strongly asymmetric toward the **lagging/contemporaneous**
side. Posts mostly happen *during* the move (+0.34) and echo it the next day
(+0.15). The only genuine lead is the weak next-day effect.

### 2. The tradeable lead is exactly one day

Forward rank-IC of the clean tradeable signal (`score_wmean` from day **t−1**
→ return over the next k days):

| horizon | rank-IC | p |
|---|---|---|
| **1 day** | **+0.053** | 0.008 |
| 2–21 days | max \|IC\| = 0.023, none significant | all > 0.24 |

Consistent with the T4.3 gate (`score_wmean` IC +0.060 on `count ≥ 10`). The
predictive power is confined to the **next trading day** and vanishes to noise
immediately after.

### 3. The 30-day moving average manufactured the "long lead"

The smoothed "mood" series showed forward rank-IC *growing* with horizon (0.026
@ 1d → **0.072 @ 10d**). That growth is an artifact, not a slow lead:

| series | lag-1 autocorrelation |
|---|---|
| raw `score_wmean` | 0.145 |
| 30-day MA | **0.989** |

A rolling mean is a backward filter (it cannot lead) and at 0.989 autocorrelation
it is near a random walk — it simply carries its own past forward. Its apparent
10-day "predictive power" is the same 1-day effect smeared across time by the
filter's persistence.

### 4. Visual-inspection caveats (why it "looked like a lead")

- **Smoothing asymmetry**: sentiment was smoothed (30-day MA) while price was
  plotted raw; a lag filter looks "earlier" next to a noisy series.
- **Same-day co-movement**: the +0.34 lag-0 bar means any big price move has a
  sentiment dip sitting on top of it, which reads as anticipation.
- **Normalization**: min-max scaling flattens sentiment (its range is set by a
  few ±1 single-post weekend days), making price dominate visually. z-scoring is
  the correct overlay.

## Conclusion

Sentiment is a **weak 1-day momentum signal** (rank-IC ≈ 0.05–0.06) and nothing
more. The impression of a multi-day lead was three stacked illusions: same-day
co-movement, the MA's visual lag, and the MA's near-unit-root persistence. There
is no multi-day lead to trade.

## Implications

1. Build on the **1-day signal only** — proceed with T4.4 calibration
   (`score_wmean`/`z_score` → P(up next day) → `p_up`), backtest, then T5.4
   block-bootstrap significance.
2. Do **not** build a multi-day "mood" strategy from the rolling mean — it will
   not survive the bootstrap.
3. If a cleaner *leading* signal is ever wanted (posts made before the open,
   rather than during the move), that is **T6.2** (overnight/pre-open buckets),
   Phase 2 — gated on the 1-day signal showing an edge first.
4. Long-horizon tests inflate naive p-values via overlapping returns; always
   confirm with the T5.4 block bootstrap before believing any multi-day claim.

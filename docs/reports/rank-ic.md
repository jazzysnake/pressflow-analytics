# T4.3 — Rank-IC gate (first kill gate)

**Status:** Gate evaluated
**Date:** 2026-10-03
**Author:** pipeline agent (read-only analysis)
**Data:** `data/processed/panel.parquet` (2,775 trading-day rows), signals from
`data/processed/signals.parquet`.

## TL;DR

| Item | Value |
|---|---|
| Best signal | **`score_wmean`** (engagement-weighted sentiment) |
| Best direction | **Momentum** (positive rank-IC) |
| Best rank-IC (robust, `count ≥ 10`) | **+0.060** (p_naive = 0.0030) |
| Kill threshold (ADR-0014 #1) | \|rank-IC\| < 0.05 → STOP |
| **Verdict** | **GO — gate passes** (0.060 ≥ 0.05), but **marginal** |
| N_eff (best signal, ρ₁ = 0.312) | ≈ **1,272** (from N = 2,424) |

The crowd is a **weak momentum** indicator, not contrarian (ADR-0008: data
decides). The raw sentiment level (`score_wmean`) edges out the rolling
`z_score` and is the cleanest, most monotone signal. The relationship is real
but small: decile 1 → decile 10 mean next-day return spread is ≈ +0.3 pp/day.
It clears the 0.05 gate by a thin margin, so proceed to Epic 05 **with
caution** — the after-cost Sharpe and block-bootstrap gates (T5.4/T5.6) will
be decisive.

---

## 0. Data checks (leakage / coverage)

- `signal_date < date` for **all** rows (0 violations) — no lookahead.
- Panel `signal_date` is weekday-only (day-of-week 0–4); **0 weekend signal
  dates** are joined to returns. Weekend signal rows (e.g. the `count=1`,
  `z_score = −5.36` on 2018-03-10) live only in `signals.parquet`; they never
  enter the return panel. They *do* enter the trailing-60-day z-score rolling
  window, which is one reason the `count` cuts matter (see §3).
- 2,775 panel rows; **2,480** have non-null signals (295 pre-signal warm-up
  trading days, 2015-10 → 2016-11, are dropped from every IC below).
- Signal autocorrelation (lag-1, full panel): `score_wmean` +0.290,
  `z_score` +0.296, `count` +0.796. Return autocorrelation (lag-1):
  `ret_log` +0.105. Hence naive Spearman p-values are optimistic (ADR-0003).

---

## 1. Rank-IC table (all signals × both directions)

Spearman rank correlation of each signal against next-day `ret_log`
(**momentum** direction) and against negated `ret_log` (**contrarian**
direction; equals −ρ, same p). Naive p-values shown — see §5 for the
autocorrelation-robust check.

### Full panel (N = 2,480)

| signal | rank-IC vs ret (momentum) | p (naive) | rank-IC vs −ret (contrarian) |
|---|---:|---:|---:|
| `z_score` | **+0.051** | 0.011 | −0.051 |
| `score_wmean` | **+0.053** | 0.008 | −0.053 |
| `disagreement_std` | +0.001 | 0.946 | −0.001 |
| `entropy` | +0.034 | 0.087 | −0.034 |
| `count` | +0.010 | 0.633 | −0.010 |

### Robustness cut (a) — `count >= 5` (N = 2,457)

| signal | rank-IC vs ret (momentum) | p (naive) | rank-IC vs −ret (contrarian) |
|---|---:|---:|---:|
| `z_score` | **+0.055** | 0.007 | −0.055 |
| `score_wmean` | **+0.057** | 0.005 | −0.057 |
| `disagreement_std` | +0.001 | 0.964 | −0.001 |
| `entropy` | +0.033 | 0.102 | −0.033 |
| `count` | +0.007 | 0.737 | −0.007 |

### Robustness cut (b) — `count >= 10` (N = 2,424)

| signal | rank-IC vs ret (momentum) | p (naive) | rank-IC vs −ret (contrarian) |
|---|---:|---:|---:|
| `z_score` | **+0.058** | 0.004 | −0.058 |
| `score_wmean` | **+0.060** | 0.003 | −0.060 |
| `disagreement_std` | +0.004 | 0.839 | −0.004 |
| `entropy` | +0.035 | 0.082 | −0.035 |
| `count` | +0.007 | 0.744 | −0.007 |

### Robustness cut (c) — excluding weekend signal dates

**No-op.** The panel already contains zero weekend signal dates (verified in
§0), so this cut is identical to the full-panel table above. The weekend
single-post extremes affect only the z-score *normalization window*; the
`count` cuts are the effective guard against that pollution.

**Takeaway:** the signal is *stronger* under the robust cuts than in the full
panel (0.060 vs 0.053 for `score_wmean`), confirming the low-count days are
diluting noise, not providing signal. `disagreement_std` and `count` carry
essentially zero rank-IC in every cut; `entropy` is weak and loses the 5%
bar under cuts.

---

## 2. Deciles (mean next-day `ret_simple` per decile)

Plot: [`rank_ic_deciles.png`](rank_ic_deciles.png) (full panel, 5 signals)
and [`rank_ic_deciles_robust.png`](rank_ic_deciles_robust.png) (count ≥ 10,
best two signals).

| signal | D1 mean ret | D10 mean ret | D1→D10 spread | ρ(decile, mean ret) |
|---|---:|---:|---:|---:|
| `z_score` (full) | −0.24% | +0.24% | **+0.48 pp** | +0.87 |
| `score_wmean` (full) | −0.10% | +0.19% | **+0.29 pp** | +0.87 |
| `disagreement_std` | +0.18% | +0.07% | −0.11 pp | −0.18 (flat) |
| `entropy` | +0.09% | +0.16% | +0.06 pp | +0.60 (weak) |
| `count` | −0.08% | +0.11% | +0.18 pp | −0.19 (flat) |
| `score_wmean` (count ≥ 10) | −0.11% | +0.22% | **+0.33 pp** | +0.87 |
| `z_score` (count ≥ 10) | −0.23% | +0.29% | **+0.52 pp** | +0.89 |

**Monotonicity:** `z_score` and `score_wmean` are monotonically increasing
from decile 1 to decile 10 (rank correlation of decile-vs-mean ≈ +0.87), with
most of the separation at the extremes (low sentiment → below-average next-day
return; high sentiment → above-average). `disagreement_std` and `count` show no
monotone pattern. `entropy` is weakly positive but non-monotone in the middle
deciles.

---

## 3. Robustness verdict (critical)

The user-flagged concern (weekend `count=1` days with extreme z-scores) is
confirmed as noise that **dilutes** the signal rather than faking it:

- Full panel `score_wmean` IC = +0.053 → `count ≥ 5` = +0.057 → `count ≥ 10`
  = **+0.060**. Same monotone improvement for `z_score`.
- Only 6 `count=1` rows exist in the panel; 23 rows have `count < 5`, 56 have
  `count < 10` — dropping them tightens the estimate without shrinking the
  effect.
- The weekend-exclusion cut is vacuous (panel already weekday-only), so the
  count cuts are the binding robustness filter.

**The gate is judged on the robust signal (`count ≥ 10`), not the noise-
dominated full panel.**

---

## 4. Best signal + direction (ADR-0008)

- **Best signal:** `score_wmean` (engagement-weighted mean sentiment), on the
  `count ≥ 10` cut. rank-IC = **+0.0602** (naive p = 0.0030).
  `z_score` is statistically indistinguishable at +0.0582 (naive p = 0.0042);
  `score_wmean` wins on magnitude, interpretability, and it is the signal the
  z-score is derived from.
- **Direction:** **Momentum**. The rank-IC is *positive* in every cut:
  higher (more bullish) crowd sentiment precedes higher next-day return. The
  contrarian direction has negative IC and is decisively rejected by the data
  (ADR-0008: no prior, data decides).

---

## 5. Kill criterion 1 (ADR-0014) + effective sample size

- **Criterion:** best |rank-IC| < 0.05 → STOP.
- **Actual:** |rank-IC| = **0.060** (0.0602) for `score_wmean`, `count ≥ 10`.
- **Verdict:** **0.060 ≥ 0.05 → gate PASSES (no kill).** This is a **narrow
  pass**, not a comfortable one.

**Effective sample size (ADR-0003).** For the best signal `score_wmean`
(`count ≥ 10`), lag-1 autocorrelation ρ = 0.312 and N = 2,424:

N_eff ≈ N · (1 − ρ) / (1 + ρ) = 2424 × 0.688 / 1.312 ≈ **1,272**

**Significance, autocorrelation-robust (not just naive p).**

| check | result |
|---|---|
| Naive Spearman p | 0.0030 |
| Block phase-randomization null p (returns block-shuffled, signal intact; L = 5/10/20, 5,000 reps) | 0.003–0.005 |
| Joint moving-block bootstrap 95% CI (L = 10, 10,000 reps) | **[+0.022, +0.096]** |

Two independent autocorrelation-aware checks agree with the naive p: the
relationship survives and is not an artifact of persistence. This is because
`ret_log` is only weakly autocorrelated (ρ = 0.105), so the effective-sample
penalty is dominated by the *signal's* persistence (captured by N_eff ≈ 1,272)
rather than by the returns. The bootstrap CI lower bound (+0.022) sits *below*
the 0.05 gate, which is the honest reason to call this pass "marginal".

---

## 6. Recommendation

**GO — pass the T4.3 kill gate, with caution.**

- The gate criterion (|rank-IC| ≥ 0.05) is met: **0.060 ≥ 0.05**.
- Direction locked as **momentum** on **`score_wmean`** (z_score as a close
  secondary); `disagreement_std` and `count` are dead and can be deprioritized
  as features; `entropy` is marginal and not worth carrying alone.
- The effect is **small** (decile spread ≈ 0.3–0.5 pp/day, IC ≈ 0.06) and the
  pass is **narrow** (CI lower bound +0.022 < 0.05). This is a "soft" edge at
  the exploration gate, exactly what ADR-0014's later gates are designed to
  catch: proceed to Epic 05, but treat the after-cost Sharpe (criterion 2) and
  the block-bootstrap p-value of the traded edge (criterion 3) as the real
  decision points.

## Artifacts

- `docs/reports/rank_ic_deciles.png` — full-panel decile means (5 signals).
- `docs/reports/rank_ic_deciles_robust.png` — count ≥ 10 deciles (score_wmean,
  z_score).

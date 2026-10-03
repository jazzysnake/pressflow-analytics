# T5.6 — Kill / go-no-go decision memo

| Field | Value |
|---|---|
| Ticket | T5.6 — Kill/gate decision memo (Epic 05 · Backtest) |
| Status | **Decision made** |
| Verdict | **KILL** — Epic 05 is a no-go, do **not** proceed to Epic 06 |
| Decided on | 2026-10-03 |
| Criteria | ADR-0014 (the four kill criteria) |
| Evidence | `docs/reports/rank-ic.md` (T4.3), `docs/reports/backtest.md` (T5.3/T5.3b), `data/processed/backtest.parquet`, `data/processed/panel.parquet`, `data/processed/prices.parquet` |

## TL;DR

Apply the four ADR-0014 kill criteria to the actual T4.3 and T5.3 results.

**Criterion 2 fails decisively** — the costed strategy Sharpe is **−0.0046** vs
SPY **+0.8808** (and OTP **+0.7534**) over the same 2015-10-05 → 2026-10-02
window. The strategy returned **−0.01%** total (CAGR −0.00%) against SPY
**+363.46%**, i.e. it finished **below cash**. Criterion 3 (block-bootstrap
p-value) is technically **not run** — see below — but is trivially satisfied
because a strategy that returned ~0% cannot beat SPY in any resample.

Per ADR-0014 ("kill or pivot if **any** of"), **any single criterion firing is
sufficient**. Two do. **Decision: KILL.**

The strategy is not a mis-sized good strategy — it holds a **negligible
long position** (mean 0.46%, max **2.0%**) because the calibrated `p_up` never
moves far from 0.5. It is effectively a cash account that pays costs. This is
the expected downstream consequence of the thin T4.3 edge (|rank-IC| = 0.060,
CI lower bound +0.022), not a bug in the backtest.

---

## 1. The four criteria, answered

| # | ADR-0014 criterion | Source | Actual | Fires? |
|---|---|---|---|---|
| 1 | Best-signal \|rank-IC\| vs next-day return **< ~0.05** | T4.3 / `rank-ic.md` | **+0.0602** (`score_wmean`, count ≥ 10) | **No** (gate passes, narrowly) |
| 2 | After-cost Sharpe **≤ SPY Sharpe** over same window | T5.3 / `backtest.md` | Strategy **−0.0046** vs SPY **+0.8808** | **YES** |
| 3 | Block-bootstrap p-value of edge **> 0.10** | T5.4 | **Not run** (deliberately); trivially **p ≈ 1.0 ≫ 0.10** | **YES** (moot, but fails) |
| 4 | Max drawdown / turnover makes it **untradeable** | T5.3 / `backtest.md` | Max DD **−0.90%**, turnover **0.0013/day**, 100 trades | **No** (not a risk blocker; also moot) |

**Decision rule:** any criterion firing → kill or pivot. Criteria 2 and 3 fire.
**Verdict: KILL.**

---

## 2. Criterion 1 — |rank-IC| < 0.05 → **does NOT fire**

| Item | Value |
|---|---|
| Best signal | `score_wmean` (engagement-weighted mean sentiment), momentum direction |
| Robust cut | `count ≥ 10` (N = 2,424) |
| Best rank-IC | **+0.0602** (naive p = 0.0030) |
| Secondary | `z_score` +0.0582 (naive p = 0.0042) |
| Threshold | \|rank-IC\| < 0.05 → STOP |
| Verdict | **0.0602 ≥ 0.05 → gate PASSES** (narrowly) |

Corroborating cuts: full panel `score_wmean` = +0.053; `count ≥ 5` = +0.057;
`count ≥ 10` = +0.060. The signal strengthens under the robust cuts, so the
full-panel value is not the one to judge on. Autocorrelation-robust checks
agree (block phase-randomization p = 0.003–0.005; joint moving-block bootstrap
95% CI **[+0.022, +0.096]**).

**This criterion alone would not kill.** Note the CI lower bound (+0.022) sits
*below* the 0.05 threshold — the pass was always known to be marginal, which is
exactly why ADR-0014 routes the real decision to criteria 2 and 3.

Source: `docs/reports/rank-ic.md` §5–6.

## 3. Criterion 2 — after-cost Sharpe ≤ SPY Sharpe → **FIRES**

Window 2015-10-05 → 2026-10-02 (2,775 trading days), from
`data/processed/backtest.parquet` and `data/processed/prices.parquet`:

| Metric | Strategy | SPY | OTP | Cash |
|---|---:|---:|---:|---:|
| Total return | **−0.01%** | **+363.46%** | +731.31% | 0.00% |
| CAGR | −0.00% | +14.94% | +21.21% | 0.00% |
| **Sharpe** | **−0.0046** | **+0.8808** | +0.7534 | 0.0000 |
| Max drawdown | −0.90% | −33.72% | −69.91% | 0.00% |
| Hit rate | 52.64% | — | — | — |
| Turnover (mean \|Δposition\|/day) | 0.0013 | — | — | — |
| Trades (zero-crossings) | 100 | — | — | — |

- Strategy Sharpe **−0.0046** ≤ SPY Sharpe **+0.8808** → **criterion fires**.
  The excess Sharpe is **−0.885**.
- The strategy underperformed SPY by **363.47 pp** and OTP by **731.32 pp**,
  and finished **below the risk-free (cash) alternative**.

Why the strategy is effectively cash: ADR-0009 sizing is
`position = min(cap, k·(2·p̃ − 1))` with shrinkage `p̃ = 0.5 + γ·(p_up − 0.5)`,
`k = ¼`, `γ = 0.5`. The calibrated `p_up` (T4.4) stays close to 0.5, so the
position is tiny: **mean 0.46%, max 2.0%**, long on 83% of days. Even a
perfectly-sized position on a ~0.06-IC daily signal (decile spread ≈ 0.3 pp/day)
cannot compete with a 14.9% CAGR equity benchmark. The failure is the **weak
signal**, not the sizing.

Source: `docs/reports/backtest.md`; reproduced from the parquet artifacts.

## 4. Criterion 3 — block-bootstrap p > 0.10 → **not run; trivially FIRES**

**T5.4 (block bootstrap) was deliberately not run**, and this memo does not
fabricate bootstrap numbers. It is moot because criterion 2 already fires
decisively and, more concretely, because the outcome is degenerate:

- The statistic of interest is the p-value of `excess ≤ 0` (how often a random
  resample of the strategy/SPY pair fails to beat SPY).
- The strategy's total return over the window is **−0.01% ≈ 0** while SPY's is
  **+363.46%**. Any autocorrelation-preserving resample of ~0%-returning pairs
  versus SPY produces an excess return far below zero.
- Hence **p(excess ≤ 0) ≈ 1.0 > 0.10 → the criterion fires**, for any block
  length L = 5–10. There is no plausible bootstrap draw in which this strategy
  beats SPY, so the expensive Monte Carlo would only confirm a foregone
  conclusion.

T5.5 (sensitivity sweep) was likewise **not run**: with the base case already at
Sharpe ≈ 0 and the remaining cases (costs 0.0015–0.006, k ∈ {⅛, ¼, ½},
γ ∈ {0.3, 0.5}, window 5y/10y) all variants of a near-cash strategy, no setting
in the sweep can turn ~0% into a SPY-beating result. Recording that explicitly
is the honest choice over manufacturing a table.

Source: T5.4/T5.5 tickets (status: not run, moot).

## 5. Criterion 4 — drawdown / turnover untradeable → **does NOT fire, but moot**

| Metric | Value | Interpretation |
|---|---:|---|
| Max drawdown | −0.90% | Benign — no blow-up risk |
| Turnover (mean \|Δposition\|/day) | 0.0013 | Negligible |
| Trades (zero-crossings) | 100 | Low |
| Hit rate | 52.64% | Marginal |

As a **standalone risk blocker** this criterion does **not** fire: the
drawdown and turnover are small, not untradeable. But it is **moot** for the
same reason as criterion 3 — the strategy barely holds a position (mean 0.46%,
max 2.0%). Small drawdown here is not evidence of a controllable strategy; it is
evidence of **not being meaningfully invested**. The absence of risk is the
absence of exposure, which is already captured by criterion 2.

Source: `docs/reports/backtest.md`.

---

## 6. Decision

**KILL. Epic 05 is a no-go.**

- Criterion 2 fires decisively (Sharpe −0.0046 ≤ +0.8808; strategy below cash).
- Criterion 3 is trivially satisfied (strategy ~0% vs SPY +363%, so the
  bootstrap p ≈ 1.0 ≫ 0.10). T5.4/T5.5 were intentionally skipped as moot —
  no numbers were invented.
- Criteria 1 and 4 do not fire, but criterion 1 passed only narrowly (CI lower
  bound +0.022 < 0.05) and criterion 4 is moot.

**Explicit stop (per DoD):** do **not** proceed to the planned Epic 06 as a
continuation of this thesis. The crowd-sentiment → next-day-return relationship
does not survive realistic costs/sizing against a passive SPY benchmark.

---

## 7. Pivot assessment — Epic 06 (author reputation) is **NOT** recommended

ADR-0013/T6.1 deferred an author-reputation model, gated on Epic 05 passing the
kill criteria, and to be pursued as a **pivot only if the aggregation failure is
plausibly noise-dominated**. That condition is **not met**:

- The failure is not an aggregation-noise problem. The aggregate signal is
  **real but tiny and measured cleanly**: rank-IC = +0.0602 with tight robustness
  (block p = 0.003–0.005, bootstrap CI [+0.022, +0.096]), and a monotone decile
  spread of only ≈ 0.3 pp/day. There is no large-but-noisy aggregate edge for a
  better aggregator to recover.
- Author reputation would only **down-weight/re-weight within the same posts**;
  it changes aggregation, not the underlying weak post-text → return
  relationship that criterion 2 exposes. It cannot manufacture an edge that the
  post text does not carry.
- It would also **shrink** the effective sample (fewer, reputation-scored
  authors per day), working against the already-marginal N_eff ≈ 1,272 and
  narrow IC.

A pivot to Epic 06 would be re-spending Phase-2 effort on the same weak signal
with a more complex aggregator — the classic post-hoc rescue ADR-0014 is
designed to prevent. **Recommendation: stop here; revisit only with a materially
different signal source, not with author-weighting of this one.**

---

## Appendix — reproduce

```bash
# T4.3 gate inputs (rank-IC by signal and count cut)
#   -> docs/reports/rank-ic.md  (best: score_wmean, count>=10, IC=+0.0602)

# T5.3 backtest metrics (strategy vs SPY/OTP/cash)
uv run python - <<'PY'
import pandas as pd, numpy as np
bt = pd.read_parquet("data/processed/backtest.parquet").set_index("date")
eq = bt["equity"]
d = eq.pct_change().dropna()
pos = bt["position"].to_numpy()
prev = np.concatenate([[0.0], pos[:-1]])
print("total return:", eq.iloc[-1]/eq.iloc[0]-1)            # -9.96e-05
print("sharpe:", (d.mean()/d.std())*np.sqrt(252))           # -0.0046
print("max dd:", (eq/eq.cummax()-1).min())                  # -0.0090
print("turnover:", np.mean(np.abs(pos-prev)))               # 0.0013
px = pd.read_parquet("data/processed/prices.parquet").set_index("date")
for c in ["spy_usd","otp_usd"]:
    s = px.loc[(px.index>=bt.index[0])&(px.index<=bt.index[-1]), c].dropna()
    r = s.pct_change().dropna()
    print(c, "total:", s.iloc[-1]/s.iloc[0]-1, "sharpe:", (r.mean()/r.std())*np.sqrt(252))
PY
```

Numbers above match `docs/reports/backtest.md` (strategy Sharpe −0.0046,
max DD −0.90%, turnover 0.0013; SPY total +363.46%, Sharpe +0.8808).

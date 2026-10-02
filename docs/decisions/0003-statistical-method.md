# 0003 — Statistical method

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Question: "Bayesian + Monte Carlo, or something else?" The honest answer is
that the heavy lifting is NLP + aggregation; Bayesian/Monte Carlo earn their
place in calibration and significance.

## Decision

- **Aggregation** (frequentist/simple): engagement-weighted mean sentiment,
  post count, disagreement (std + entropy), rolling z-score of the level.
- **Calibration** (Bayesian): logistic/isotonic → P(next-day up), giving a
  distribution over the probability so sizing is uncertainty-aware; shrink
  toward 0.5.
- **Significance** (Monte Carlo): block bootstrap + permutation tests, because
  returns are autocorrelated and the signal is persistent — naive t-stats lie.
- **Effective sample size** is respected: N_eff ≈ N·(1−ρ)/(1+ρ).

## Consequences

- Sentiment extraction stays simple and strong; no Bayesian complexity there.
- The decisive backtest statistic is the bootstrap p-value (T5.4).

## References

- ADR-0007 (frequency), ADR-0009 (sizing), ADR-0014 (kill criteria)

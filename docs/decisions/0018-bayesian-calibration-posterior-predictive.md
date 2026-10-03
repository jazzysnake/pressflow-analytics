# 0018 — Bayesian calibration & posterior-predictive sizing

- **Status:** Accepted (gated on T4.3)
- **Date:** 2026-10-03

## Context

ADR-0003 commits to "Bayesian calibration" and "uncertainty-aware sizing", but
the concrete ticket T4.4 implements a *point* logistic estimate with shrink
toward 0.5, not a full posterior. We want the posterior over P(up) and a
posterior-predictive distribution over next-day return, so we can (a) size by
conviction with honest uncertainty and (b) answer "P(return > x%)" instead of a
binary up/down.

## Decision

- Calibration becomes **Bayesian**: logistic regression (or linear regression on
  returns, whichever the data supports) with a weakly informative prior on the
  coefficients, fit **walk-forward** (expanding window), no lookahead
  (ADR-0010).
- Produce per day: a **posterior over P(up)** (mean, SD, credible interval) and
  a **posterior-predictive distribution of next-day return** → `P(return > x%)`
  for cost-relevant thresholds.
- Sizing consumes the posterior (T5.7): position scales with expected P(up) and
  **shrinks with posterior uncertainty**; long-only, in [0, 1] (ADR-0009).
- This **supersedes** the T4.4 point estimate as the production calibration;
  T4.4 remains as the quick baseline for A/B comparison.

## Gating

This ADR is **conditional**: it only matters if the crowd signal is not pure
noise. Implementation starts **only if T4.3 passes** (best-signal |rank-IC| ≳
0.05). If ADR-0014 kill criterion 1 fires, this ADR is moot.

## Consequences

- Honest uncertainty propagation; no false precision from a point p̂.
- Modest extra compute: MCMC/variational inference on ~2,500 daily observations
  fits in seconds.
- Risk of overfitting latent structure onto a weak signal — mitigated by the
  gate and by requiring Brier/log-loss no worse than the T4.4 baseline.

## References

- ADR-0003 (statistical method), ADR-0009 (sizing), ADR-0010 (no lookahead),
  ADR-0014 (kill criteria), Tickets T4.3, T4.4, T4.6, T5.7

# 0016 — Adopt Jev (TypeSafe API) as the sentiment model

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

ADR-0001 chose off-the-shelf HF sentiment classifiers and picked
`NYTK/sentiment-hts5-xlm-roberta-hungarian` in T3.3. On the 201-post eval set
those models are near-chance on the directional classes (hts5 bull row 9-9-9
= random; bull recall 33%, bear recall 36%). ADR-0001's premise — that an
off-the-shelf model is "mostly right" — was not met, which threatened the
project's kill criteria (ADR-0014). We tested Jev (TypeSafe's API-only
decision model) as an alternative (T3.6).

## Decision

Use **Jev** (`jev-latest`, TypeSafe API) as the sentiment model. Each post is
sent to `POST /v1/systemone` with a `choice` question over {bull, bear,
neutral}; Jev's returned probabilities map to `p_bull`/`p_bear`/`p_neutral`
with `net = p_bull − p_bear`. This supersedes the HF candidate list in
ADR-0001. It remains zero-shot — no fine-tuning.

## Evidence

- Jev directional macro-F1 **0.566** vs hts5 **0.230** (2.5×).
- Jev bull recall **74%** vs 33%; bear recall **84%** vs 36%.
- Jev macro-F1 **0.645** vs 0.354 (hts5), 0.418 (ohb3), 0.319 (cardiffnlp).
- The original T3.6 "accuracy > neutral baseline" gate was dropped post-hoc:
  on a 74%-neutral eval set it penalizes committing to a direction — the
  opposite of what the signal needs.

## Consequences

- New dependency: paid, network-available API. ~$0.042/1M input tokens →
  ~$4–6 for the full 277k-post corpus. Latency ~0.5s/call → needs concurrency
  (ADR-0017).
- Probabilities are claimed calibrated (decision-model training); ECE on our
  data still to be verified (T3.5).
- Jev has a bear-lean on neutral posts (29 bear vs 17 bull mislabels); watch
  for aggregate bias in T4.1.
- Still no training — ADR-0001's "no fine-tuning" holds.

## References

- ADR-0001 (superseded candidate list), ADR-0014, T3.3, T3.6

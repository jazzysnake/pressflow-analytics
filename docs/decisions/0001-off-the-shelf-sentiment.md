# 0001 — Use off-the-shelf sentiment, no training

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

~1.3–1.6M Hungarian forum posts, no labeled data, no wish to train a
classifier. The hypothesis is that alpha lives in the *aggregation* of noisy
opinions, not in perfect per-post classification.

## Decision

Use pretrained sentiment models as-is, no fine-tuning. Evaluate 2–3 candidates
on a ~200-post eval set labeled by two independent LLM models (keep only posts where both agree) and pick one. Use continuous
probabilities (P bull / P bear), not hard classes.

**Candidates:**
- `NYTK/sentiment-hts5-xlm-roberta-hungarian` (5-class, Twitter-trained)
- `NYTK/sentiment-ohb3-xlm-roberta-hungarian` (3-class, opinion-text-trained)
- `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual 3-class baseline)

## Consequences

- Cheap and fast; can be run overnight on CPU.
- Per-post noise is acceptable — aggregation averages it out.
- Risk: systematic bias (e.g., sarcasm misclassification) is *not* averaged
  away; mitigated by model selection (T3.3) and the disagreement signal.

## References

- Tickets: T3.1–T3.5

# pressflow-analytics

Sentiment analysis of Hungarian forum posts, used to backtest a long-only
trading signal.

## Setup

Uses [uv](https://docs.astral.sh/uv/) with Python 3.12.

```bash
uv sync                 # lightweight dependencies
uv sync --extra ml      # add the heavy ML stack (torch, transformers, ...)
uv run pytest           # run the test suite
```

Raw and processed data live under `data/` and are gitignored.

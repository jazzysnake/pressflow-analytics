"""Canonical Parquet schemas for the pressflow pipeline.

The contract is documented in ``docs/schema.md``; keep the two in sync
(ADR-0015). Every schema here is the single source of truth for column names,
types and nullability, so downstream stages never have to guess.

Pipeline (ADR-0002)::

    posts  ──►  scores ──┐
                         ├──►  signals  ──►  backtest
              prices ────┘
"""

from __future__ import annotations

import pyarrow as pa

#: Timezone for every post timestamp and trading day (ADR-0015).
BUDAPEST_TZ = "Europe/Budapest"

#: Precision (unit) for all temporal columns.
TS_UNIT = "us"

#: Shared tz-aware, microsecond timestamp type used by ``ts`` and ``date``.
DATETIME = pa.timestamp(TS_UNIT, tz=BUDAPEST_TZ)

# --- column-name constants -------------------------------------------------
# Shared columns
DATE = "date"
HSZ_ID = "hsz_id"

# posts
POST_ID = "post_id"
AUTHOR = "author"
USER_ID = "user_id"
TS = "ts"
REPLY_TO_POST_ID = "reply_to_post_id"
REPLY_TO_HSZ_ID = "reply_to_hsz_id"
UPVOTES = "upvotes"
DOWNVOTES = "downvotes"
TEXT = "text"
PAGE = "page"

# prices
OTP_CLOSE_HUF = "otp_close_huf"
OTP_ADJ_CLOSE_HUF = "otp_adj_close_huf"
USDHUF = "usdhuf"
SPY_ADJ_CLOSE_USD = "spy_adj_close_usd"
SPX_CLOSE = "spx_close"
OTP_USD = "otp_usd"
SPY_USD = "spy_usd"

# scores
MODEL = "model"
P_BULL = "p_bull"
P_BEAR = "p_bear"
P_NEUTRAL = "p_neutral"
NET = "net"

# signals
SCORE_WMEAN = "score_wmean"
BEAR_WMEAN = "bear_wmean"
COUNT = "count"
DISAGREEMENT_STD = "disagreement_std"
ENTROPY = "entropy"
Z_SCORE = "z_score"

# backtest
POSITION = "position"
RET = "ret"
COST = "cost"
EQUITY = "equity"

# --- ordered column-name tuples --------------------------------------------

POSTS_COLUMNS: tuple[str, ...] = (
    POST_ID,
    HSZ_ID,
    AUTHOR,
    USER_ID,
    TS,
    REPLY_TO_POST_ID,
    REPLY_TO_HSZ_ID,
    UPVOTES,
    DOWNVOTES,
    TEXT,
    PAGE,
)

PRICES_COLUMNS: tuple[str, ...] = (
    DATE,
    OTP_CLOSE_HUF,
    OTP_ADJ_CLOSE_HUF,
    USDHUF,
    SPY_ADJ_CLOSE_USD,
    SPX_CLOSE,
    OTP_USD,
    SPY_USD,
)

SCORES_COLUMNS: tuple[str, ...] = (
    HSZ_ID,
    MODEL,
    P_BULL,
    P_BEAR,
    P_NEUTRAL,
    NET,
)

SIGNALS_COLUMNS: tuple[str, ...] = (
    DATE,
    SCORE_WMEAN,
    BEAR_WMEAN,
    COUNT,
    DISAGREEMENT_STD,
    ENTROPY,
    Z_SCORE,
)

BACKTEST_COLUMNS: tuple[str, ...] = (
    DATE,
    POSITION,
    RET,
    COST,
    EQUITY,
)

# --- pyarrow schemas -------------------------------------------------------

POSTS_SCHEMA = pa.schema(
    [
        pa.field(POST_ID, pa.int64(), nullable=False),
        pa.field(HSZ_ID, pa.int64(), nullable=False),
        pa.field(AUTHOR, pa.string(), nullable=False),
        pa.field(USER_ID, pa.int64(), nullable=False),
        pa.field(TS, DATETIME, nullable=False),
        pa.field(REPLY_TO_POST_ID, pa.int64(), nullable=True),
        pa.field(REPLY_TO_HSZ_ID, pa.int64(), nullable=True),
        pa.field(UPVOTES, pa.int64(), nullable=False),
        pa.field(DOWNVOTES, pa.int64(), nullable=False),
        pa.field(TEXT, pa.string(), nullable=False),
        pa.field(PAGE, pa.int64(), nullable=False),
    ]
)

PRICES_SCHEMA = pa.schema(
    [
        pa.field(DATE, DATETIME, nullable=False),
        pa.field(OTP_CLOSE_HUF, pa.float64(), nullable=False),
        pa.field(OTP_ADJ_CLOSE_HUF, pa.float64(), nullable=False),
        pa.field(USDHUF, pa.float64(), nullable=False),
        pa.field(SPY_ADJ_CLOSE_USD, pa.float64(), nullable=False),
        pa.field(SPX_CLOSE, pa.float64(), nullable=False),
        pa.field(OTP_USD, pa.float64(), nullable=False),
        pa.field(SPY_USD, pa.float64(), nullable=False),
    ]
)

SCORES_SCHEMA = pa.schema(
    [
        pa.field(HSZ_ID, pa.int64(), nullable=False),
        pa.field(MODEL, pa.string(), nullable=False),
        pa.field(P_BULL, pa.float64(), nullable=False),
        pa.field(P_BEAR, pa.float64(), nullable=False),
        pa.field(P_NEUTRAL, pa.float64(), nullable=False),
        pa.field(NET, pa.float64(), nullable=False),
    ]
)

SIGNALS_SCHEMA = pa.schema(
    [
        pa.field(DATE, DATETIME, nullable=False),
        pa.field(SCORE_WMEAN, pa.float64(), nullable=False),
        pa.field(BEAR_WMEAN, pa.float64(), nullable=False),
        pa.field(COUNT, pa.int64(), nullable=False),
        pa.field(DISAGREEMENT_STD, pa.float64(), nullable=False),
        pa.field(ENTROPY, pa.float64(), nullable=False),
        pa.field(Z_SCORE, pa.float64(), nullable=False),
    ]
)

BACKTEST_SCHEMA = pa.schema(
    [
        pa.field(DATE, DATETIME, nullable=False),
        pa.field(POSITION, pa.float64(), nullable=False),
        pa.field(RET, pa.float64(), nullable=False),
        pa.field(COST, pa.float64(), nullable=False),
        pa.field(EQUITY, pa.float64(), nullable=False),
    ]
)

# --- registries ------------------------------------------------------------

#: Artifact name -> ordered column names.
COLUMN_NAMES: dict[str, tuple[str, ...]] = {
    "posts": POSTS_COLUMNS,
    "prices": PRICES_COLUMNS,
    "scores": SCORES_COLUMNS,
    "signals": SIGNALS_COLUMNS,
    "backtest": BACKTEST_COLUMNS,
}

#: Artifact name -> canonical pyarrow schema.
SCHEMAS: dict[str, pa.Schema] = {
    "posts": POSTS_SCHEMA,
    "prices": PRICES_SCHEMA,
    "scores": SCORES_SCHEMA,
    "signals": SIGNALS_SCHEMA,
    "backtest": BACKTEST_SCHEMA,
}

#: Artifact name -> natural/dedupe key columns.
NATURAL_KEYS: dict[str, tuple[str, ...]] = {
    "posts": (HSZ_ID,),
    "prices": (DATE,),
    "scores": (HSZ_ID,),
    "signals": (DATE,),
    "backtest": (DATE,),
}

__all__ = [
    "BUDAPEST_TZ",
    "TS_UNIT",
    "DATETIME",
    "POSTS_COLUMNS",
    "PRICES_COLUMNS",
    "SCORES_COLUMNS",
    "SIGNALS_COLUMNS",
    "BACKTEST_COLUMNS",
    "POSTS_SCHEMA",
    "PRICES_SCHEMA",
    "SCORES_SCHEMA",
    "SIGNALS_SCHEMA",
    "BACKTEST_SCHEMA",
    "COLUMN_NAMES",
    "SCHEMAS",
    "NATURAL_KEYS",
]

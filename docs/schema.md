# Data schemas & conventions

Canonical Parquet contracts for the four-stage pipeline (ADR-0002). Defined once
here and mirrored in code by `src/pressflow/schema.py` — **keep the two in sync**
(ADR-0015).

## Conventions

- **Format:** Parquet everywhere. `data/` is gitignored.
- **Timezone:** every timestamp and trading day is **Europe/Budapest**
  (ADR-0015). Temporal columns are tz-aware `timestamp` (microsecond precision).
- **Numeric types:** integers are `int64`, probabilities/returns are `float64`.
  Nullability is explicit — only columns marked *nullable* below may be null.
- **Natural keys:** each artifact has a dedupe key (section: *Natural key*).
- Paths are relative to the repository root.

---

### posts

`data/processed/posts.parquet` — scraped forum posts (Epic 01).

**Natural key:** `hsz_id`

| Column | Type | Unit / notes |
|---|---|---|
| `post_id` | int64 | Display post id from the `.id` div (e.g. `#1340744` → `1340744`). |
| `hsz_id` | int64 | Vote-span `data-hsz_id`; stable natural key. |
| `author` | string | Username from `a.username`. |
| `user_id` | int64 | `data-user_id` account id. |
| `ts` | timestamp[us, tz=Europe/Budapest] | Absolute post timestamp. |
| `reply_to_post_id` | int64, nullable | Display id of the quoted post; null for a root post. |
| `reply_to_hsz_id` | int64, nullable | `hsz_id` from the "Előzmény" link; null for a root post. |
| `upvotes` | int64 | Count of `+1` votes as scraped. |
| `downvotes` | int64 | Count of `-1` votes as scraped. |
| `text` | string | Post body plain text. |
| `page` | int64 | Forum page number (`?oldal=N`) the post was scraped from. |

---

### prices

`data/processed/prices.parquet` — daily USD-denominated price panel (Epic 02).

**Natural key:** `date` (one row per BÉT trading day)

| Column | Type | Unit / notes |
|---|---|---|
| `date` | timestamp[us, tz=Europe/Budapest] | BÉT trading day. |
| `otp_close_huf` | float64 | OTP.BD raw close, HUF per share. |
| `otp_adj_close_huf` | float64 | OTP.BD dividend-adjusted close, HUF per share (total-return basis). |
| `usdhuf` | float64 | USDHUF=X close, HUF per 1 USD. |
| `spy_adj_close_usd` | float64 | SPY adjusted close, USD per share (raw series). |
| `spx_close` | float64 | ^GSPC close, index points (reference only). |
| `otp_usd` | float64 | Derived: OTP adjusted close converted to USD via `usdhuf`, USD per share (see T2.2). |
| `spy_usd` | float64 | Derived: SPY total return in USD, USD per share (see T2.2). |

---

### scores

`data/processed/scores.parquet` — per-post sentiment scores (Epic 03).

**Natural key:** `hsz_id`

| Column | Type | Unit / notes |
|---|---|---|
| `hsz_id` | int64 | Foreign key to `posts.hsz_id`. |
| `model` | string | Model identifier (name + version). |
| `p_bull` | float64 | Probability the post is bullish, in [0, 1]. |
| `p_bear` | float64 | Probability the post is bearish, in [0, 1]. |
| `p_neutral` | float64 | Probability the post is neutral, in [0, 1]. |
| `net` | float64 | Signed sentiment `p_bull - p_bear`, in [-1, 1]. |

---

### signals

`data/processed/signals.parquet` — one aggregated row per Budapest trading day
(Epic 04).

**Natural key:** `date` (one row per Budapest trading day)

| Column | Type | Unit / notes |
|---|---|---|
| `date` | timestamp[us, tz=Europe/Budapest] | Budapest trading day. |
| `score_wmean` | float64 | Engagement-weighted mean of `net` (weight `w = 1 + upvotes`), in [-1, 1]. |
| `bear_wmean` | float64 | Engagement-weighted mean of `p_bear`, in [0, 1]. |
| `count` | int64 | Number of scored posts that day. |
| `disagreement_std` | float64 | Unweighted standard deviation of `net` within the day. |
| `entropy` | float64 | Mean Shannon entropy of (`p_bull`, `p_bear`, `p_neutral`) across posts. |
| `z_score` | float64 | Rolling z-score of `score_wmean` over the trailing 60-day window. |

---

### backtest

`data/processed/backtest.parquet` — daily long-only strategy curve (Epic 05).

**Natural key:** `date` (one row per trading day)

| Column | Type | Unit / notes |
|---|---|---|
| `date` | timestamp[us, tz=Europe/Budapest] | Trading day. |
| `position` | float64 | Fraction of capital held long, in [0, 1]; 0 = cash. |
| `ret` | float64 | Daily net strategy return = `position · ret_simple - cost`, as a fraction. |
| `cost` | float64 | Daily transaction cost from turnover, as a fraction of capital. |
| `equity` | float64 | Compounded equity curve, indexed to 1.0 at the start. |

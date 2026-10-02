"""T2.2 — currency & benchmark alignment (ADR-0005, ADR-0011).

Turns the raw multi-ticker download from T2.1 into the aligned, USD-denominated
``prices`` panel defined in :mod:`pressflow.schema` / ``docs/schema.md``:

* trading calendar = every day ``OTP.BD`` has a bar (BÉT);
* ``otp_usd = otp_adj_close_huf / usdhuf`` (USDHUF is HUF per 1 USD — division);
* ``spy_usd = SPY Adj Close`` (total return, USD);
* the SPY/^GSPC series are forward-filled onto the BÉT calendar (the US and
  Hungarian calendars do not coincide), as is USDHUF for rare missing quotes;
* the date index is normalised to tz-aware Europe/Budapest midnight.

Usage::

    uv run python -m pressflow.align_prices
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pressflow.config import REPO_ROOT, Config, get_config
from pressflow.fetch_prices import (
    RAW_FILENAME,
    RAW_OTP_ADJ_CLOSE,
    RAW_OTP_CLOSE,
    RAW_SPX_CLOSE,
    RAW_SPY_ADJ_CLOSE,
    RAW_USDHUF,
)
from pressflow.schema import (
    BUDAPEST_TZ,
    DATE,
    OTP_ADJ_CLOSE_HUF,
    OTP_CLOSE_HUF,
    OTP_USD,
    PRICES_COLUMNS,
    PRICES_SCHEMA,
    SPX_CLOSE,
    SPY_ADJ_CLOSE_USD,
    SPY_USD,
    USDHUF,
)

logger = logging.getLogger(__name__)

PROCESSED_FILENAME = "prices.parquet"

#: Raw columns the alignment depends on, in a stable order.
REQUIRED_RAW: tuple[str, ...] = (
    RAW_OTP_CLOSE,
    RAW_OTP_ADJ_CLOSE,
    RAW_USDHUF,
    RAW_SPY_ADJ_CLOSE,
    RAW_SPX_CLOSE,
)


def to_budapest_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalise any date index to tz-aware Europe/Budapest midnight."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize(BUDAPEST_TZ)
    else:
        idx = idx.tz_convert(BUDAPEST_TZ)
    return idx.normalize().as_unit("us")


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the 8-column ``prices`` panel from the flattened raw frame.

    Raises :class:`KeyError` if a required raw column is missing and
    :class:`ValueError` if any BÉT trading day still has a NaN after filling.
    """
    missing = [col for col in REQUIRED_RAW if col not in raw.columns]
    if missing:
        raise KeyError(f"raw prices missing columns: {missing}")

    frame = raw.copy()
    frame[DATE] = pd.to_datetime(frame[DATE])
    frame = frame.set_index(DATE).sort_index()
    frame.index = to_budapest_index(frame.index)

    work = frame[list(REQUIRED_RAW)]
    otp_calendar = work[RAW_OTP_CLOSE].notna()

    # Forward-fill each series across the union calendar, then back-fill only
    # the leading edge. The 1-year buffer means any back-fill sits before the
    # analysis window and cannot leak into signals.
    fx_missing_on_bet = work[RAW_USDHUF].isna() & otp_calendar
    spy_missing_on_bet = work[RAW_SPY_ADJ_CLOSE].isna() & otp_calendar
    spx_missing_on_bet = work[RAW_SPX_CLOSE].isna() & otp_calendar
    filled = work.ffill().bfill()

    panel = filled.loc[otp_calendar]
    if panel.empty:
        raise ValueError("no OTP.BD bars found in raw prices")

    out = pd.DataFrame(index=panel.index)
    out[OTP_CLOSE_HUF] = panel[RAW_OTP_CLOSE]
    out[OTP_ADJ_CLOSE_HUF] = panel[RAW_OTP_ADJ_CLOSE]
    out[USDHUF] = panel[RAW_USDHUF]
    out[SPY_ADJ_CLOSE_USD] = panel[RAW_SPY_ADJ_CLOSE]
    out[SPX_CLOSE] = panel[RAW_SPX_CLOSE]
    out[OTP_USD] = out[OTP_ADJ_CLOSE_HUF] / out[USDHUF]
    out[SPY_USD] = out[SPY_ADJ_CLOSE_USD]

    out.index.name = DATE
    out = out.reset_index()
    out = out[list(PRICES_COLUMNS)]

    n_fx = int(fx_missing_on_bet.sum())
    n_spy = int(spy_missing_on_bet.sum())
    n_spx = int(spx_missing_on_bet.sum())
    if n_fx:
        logger.warning(
            "forward-filled USDHUF on %d BÉT trading day(s): %s",
            n_fx,
            _dates(fx_missing_on_bet),
        )
    if n_spy:
        logger.info("forward-filled SPY Adj Close on %d BÉT day(s)", n_spy)
    if n_spx:
        logger.info("forward-filled ^GSPC Close on %d BÉT day(s)", n_spx)

    if out[list(PRICES_COLUMNS)].isna().any().any():
        bad = out[out[list(PRICES_COLUMNS)].isna().any(axis=1)][DATE].tolist()
        raise ValueError(f"panel still has NaN on BÉT days: {bad[:5]}")
    return out


def _dates(mask: pd.Series) -> list[str]:
    return [str(d.date()) for d in mask.index[mask][:5]]


def processed_path(config: Config | None = None) -> Path:
    """Absolute path of the processed prices artifact."""
    config = config or get_config()
    return REPO_ROOT / config.paths.processed / PROCESSED_FILENAME


def save_panel(panel: pd.DataFrame, path: Path) -> None:
    """Write the panel with the canonical :data:`pressflow.schema.PRICES_SCHEMA`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, schema=PRICES_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


def build_and_save(config: Config | None = None) -> pd.DataFrame:
    """Read raw prices, build the panel and write ``data/processed/prices.parquet``."""
    config = config or get_config()
    raw_file = REPO_ROOT / config.paths.raw / RAW_FILENAME
    if not raw_file.is_file():
        raise FileNotFoundError(f"raw prices not found: {raw_file} (run T2.1 first)")
    raw = pd.read_parquet(raw_file)
    panel = build_panel(raw)
    out_file = processed_path(config)
    save_panel(panel, out_file)
    print(f"wrote {out_file} ({len(panel)} rows, {len(panel.columns)} columns)")
    return panel


def main() -> None:  # pragma: no cover - thin CLI wrapper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_and_save()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "PROCESSED_FILENAME",
    "REQUIRED_RAW",
    "to_budapest_index",
    "build_panel",
    "save_panel",
    "processed_path",
    "build_and_save",
]

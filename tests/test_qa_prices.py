"""Tests for T2.3 — total-return QA report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressflow.align_prices import save_panel
from pressflow.fetch_prices import normalize_dates
from pressflow.qa_prices import (
    build_report,
    calendar_gaps,
    dividend_events,
    invalid_price_rows,
    run_qa,
    usdhuf_anomalies,
)
from pressflow.schema import (
    DATE,
    OTP_ADJ_CLOSE_HUF,
    OTP_CLOSE_HUF,
    OTP_USD,
    SPX_CLOSE,
    SPY_ADJ_CLOSE_USD,
    SPY_USD,
    USDHUF,
)


def _panel(dates, otp_close, otp_adj, usdhuf, spy, spx) -> pd.DataFrame:
    otp_close = np.asarray(otp_close, dtype="float64")
    otp_adj = np.asarray(otp_adj, dtype="float64")
    usdhuf = np.asarray(usdhuf, dtype="float64")
    spy = np.asarray(spy, dtype="float64")
    spx = np.asarray(spx, dtype="float64")
    return pd.DataFrame(
        {
            DATE: normalize_dates(pd.to_datetime(dates)),
            OTP_CLOSE_HUF: otp_close,
            OTP_ADJ_CLOSE_HUF: otp_adj,
            USDHUF: usdhuf,
            SPY_ADJ_CLOSE_USD: spy,
            SPX_CLOSE: spx,
            OTP_USD: otp_adj / usdhuf,
            SPY_USD: spy,
        }
    )


def _dividend_panel() -> pd.DataFrame:
    return _panel(
        dates=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        otp_close=[100.0, 100.0, 100.0, 100.0],
        otp_adj=[95.0, 95.0, 100.0, 100.0],
        usdhuf=[300.0, 300.0, 300.0, 300.0],
        spy=[400.0, 401.0, 402.0, 403.0],
        spx=[5000.0, 5010.0, 5020.0, 5030.0],
    )


def test_dividend_events_detects_step_and_yield():
    events = dividend_events(_dividend_panel())
    assert len(events) == 1
    assert events.loc[0, DATE] == pd.Timestamp("2024-01-04", tz="Europe/Budapest")
    assert events.loc[0, "implied_yield"] == pytest.approx(0.05)
    assert events.loc[0, "approx_dividend_huf"] == pytest.approx(5.0)


def test_dividend_events_empty_when_no_adjustment():
    panel = _panel(
        dates=["2024-01-02", "2024-01-03"],
        otp_close=[100.0, 101.0],
        otp_adj=[100.0, 101.0],
        usdhuf=[300.0, 300.0],
        spy=[400.0, 401.0],
        spx=[5000.0, 5010.0],
    )
    assert dividend_events(panel).empty


def test_usdhuf_anomalies_flags_out_of_range_and_non_positive():
    with np.errstate(divide="ignore"):
        panel = _panel(
            dates=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            otp_close=[100.0, 100.0, 100.0, 100.0],
            otp_adj=[100.0, 100.0, 100.0, 100.0],
            usdhuf=[300.0, 0.0, 100.0, 700.0],
            spy=[400.0, 401.0, 402.0, 403.0],
            spx=[5000.0, 5010.0, 5020.0, 5030.0],
        )
    anomalies = usdhuf_anomalies(panel)
    assert len(anomalies) == 3
    assert list(pd.to_datetime(anomalies[DATE]).dt.strftime("%Y-%m-%d")) == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]


def test_calendar_gaps_flags_long_holiday_gap():
    panel = _panel(
        dates=["2024-01-02", "2024-01-03", "2024-01-15"],
        otp_close=[100.0, 101.0, 102.0],
        otp_adj=[100.0, 101.0, 102.0],
        usdhuf=[300.0, 300.0, 300.0],
        spy=[400.0, 401.0, 402.0],
        spx=[5000.0, 5010.0, 5020.0],
    )
    gaps = calendar_gaps(panel, max_days=4)
    assert len(gaps) == 1
    assert gaps.loc[0, DATE] == pd.Timestamp("2024-01-15", tz="Europe/Budapest")
    assert gaps.loc[0, "gap_days"] == 12


def test_invalid_price_rows_flags_zero_and_negative():
    panel = _panel(
        dates=["2024-01-02", "2024-01-03", "2024-01-04"],
        otp_close=[100.0, 0.0, 102.0],
        otp_adj=[100.0, 99.0, -1.0],
        usdhuf=[300.0, 300.0, 300.0],
        spy=[400.0, 401.0, 402.0],
        spx=[5000.0, 5010.0, 5020.0],
    )
    invalid = invalid_price_rows(panel)
    assert len(invalid) == 2


def test_build_report_contains_all_sections_and_findings():
    text = build_report(_dividend_panel())
    assert "# Price data QA" in text
    assert "## OTP dividend adjustments" in text
    assert "## USDHUF sanity" in text
    assert "## Calendar gaps" in text
    assert "## Zero / negative quotes" in text
    assert "## Caveats" in text
    assert "**1**" in text  # one dividend event
    assert "2024-01-04" in text


def test_run_qa_writes_markdown_report(tmp_path, monkeypatch):
    import pressflow.qa_prices as q

    panel = _dividend_panel()
    panel_file = tmp_path / "prices.parquet"
    save_panel(panel, panel_file)
    report_file = tmp_path / "prices-qa.md"

    monkeypatch.setattr(q, "processed_path", lambda config=None: panel_file)
    monkeypatch.setattr(q, "report_path", lambda: report_file)

    q.run_qa()
    assert report_file.is_file()
    assert "# Price data QA" in report_file.read_text(encoding="utf-8")

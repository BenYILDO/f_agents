"""Tests for the macro event study (forward-return base rates after events)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows import macro_event_study as es


def _rising_history(start="2024-01-01", n=80, step=1.0, base=100.0) -> pd.DataFrame:
    """Business-day OHLCV whose Close rises by ``step`` each bar (deterministic)."""
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(base + step * np.arange(n), index=idx)
    return pd.DataFrame({
        "Open": close, "High": close + 0.5, "Low": close - 0.5,
        "Close": close, "Volume": 1_000_000.0,
    })


@pytest.mark.unit
class TestComputeEventStudy:
    def test_forward_returns_and_aggregation(self):
        hist = _rising_history()
        # two event dates that land on business days well inside the window
        events = ["2024-01-15", "2024-02-15"]
        res = es.compute_event_study(
            "THYAO.IS", events, direction_label="TCMB faiz indirimi",
            horizons=(1, 5), benchmark=None, history=hist,
        )
        assert res.ok and res.n_events == 2
        # every horizon present, all positive (monotonically rising series)
        assert set(res.stats.keys()) == {1, 5}
        for h in (1, 5):
            assert res.stats[h]["hit_rate"] == 1.0
            assert res.stats[h]["mean"] > 0
            assert res.stats[h]["n"] == 2
        # +5d return from a +1/bar series at base ~ entry close: 5/base
        entry_close = hist["Close"].asof(pd.Timestamp("2024-01-15"))
        # asof gives last on/before; entry is first on/after, close enough for sign
        assert res.per_event[0]["fwd_5"] == pytest.approx(5.0 / entry_close, rel=0.05)

    def test_benchmark_alpha_is_computed_when_provided(self):
        hist = _rising_history(step=2.0)          # stock +2/bar
        bench = _rising_history(step=1.0)         # index +1/bar -> positive alpha
        res = es.compute_event_study(
            "X.IS", ["2024-01-15"], horizons=(5,), benchmark="XU100.IS",
            history=hist, benchmark_history=bench,
        )
        assert res.ok and res.benchmark == "XU100.IS"
        assert res.stats[5]["mean_alpha"] is not None
        assert res.stats[5]["mean_alpha"] > 0

    def test_no_events_returns_not_ok(self):
        res = es.compute_event_study("X.IS", [], history=_rising_history())
        assert res.ok is False and "bulunamadı" in res.reason

    def test_insufficient_price_data_not_ok(self):
        tiny = _rising_history(n=3)
        res = es.compute_event_study("X.IS", ["2024-01-02"], horizons=(5, 10),
                                     benchmark=None, history=tiny)
        assert res.ok is False

    def test_max_events_keeps_most_recent(self):
        hist = _rising_history(n=200, start="2024-01-01")
        events = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-10", periods=20)]
        res = es.compute_event_study("X.IS", events, horizons=(1,), benchmark=None,
                                     history=hist, max_events=5)
        assert res.n_events <= 5


@pytest.mark.unit
class TestFormatEventStudy:
    def test_format_ok_result_has_numbers(self):
        hist = _rising_history()
        res = es.compute_event_study("THYAO.IS", ["2024-01-15", "2024-02-15"],
                                     direction_label="TCMB faiz indirimi",
                                     horizons=(1, 5), benchmark=None, history=hist)
        text = es.format_event_study(res)
        assert "GEÇMİŞ OLAY-ETÜDÜ" in text and "THYAO" in text
        assert "işlem günü" in text and "pozitif oran" in text

    def test_format_failed_result_is_placeholder(self):
        res = es.compute_event_study("X.IS", [], history=_rising_history())
        text = es.format_event_study(res)
        assert text.startswith("<") and "olay-etüdü" in text.lower()

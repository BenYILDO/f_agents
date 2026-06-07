"""Tests for the dip-buy / top-sell technical strategy (video.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.strategy import dip_signal


def _make_ohlcv(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic but realistic OHLCV with a dip-then-rally so signals can fire."""
    rng = np.random.default_rng(seed)
    # V-shape close path: fall then recover, plus noise
    t = np.linspace(0, 1, n)
    base = 100 - 30 * np.sin(np.pi * t) + rng.normal(0, 1.2, n).cumsum() * 0.1
    close = pd.Series(base, index=pd.date_range("2025-01-01", periods=n, freq="D"))
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    openp = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(rng.uniform(1e6, 5e6, n), index=close.index)
    return pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close, "Volume": vol})


@pytest.mark.unit
class TestComputeSmi:
    def test_smi_in_bounds_and_signal_present(self):
        df = _make_ohlcv()
        smi, signal = dip_signal.compute_smi(df)
        valid = smi.dropna()
        assert len(valid) > 50
        # SMI is bounded roughly within [-100, 100]
        assert valid.between(-120, 120).all()
        assert len(signal.dropna()) > 50

    def test_custom_params_change_output(self):
        df = _make_ohlcv()
        a, _ = dip_signal.compute_smi(df, k=10, d=3, ema_len=3)
        b, _ = dip_signal.compute_smi(df, k=5, d=2, ema_len=2)
        assert not a.dropna().equals(b.dropna())


@pytest.mark.unit
class TestComputeSignals:
    def test_adds_expected_columns(self):
        out = dip_signal.compute_signals(_make_ohlcv())
        for col in ("smi", "smi_signal", "smi_vwma", "bb_mid", "bb_upper",
                    "cross_up_below0", "buy", "sell"):
            assert col in out.columns
        # buy/sell are clean booleans (no NaN leaking through)
        assert out["buy"].dtype == bool
        assert out["sell"].dtype == bool

    def test_buy_requires_close_above_bb_mid(self):
        out = dip_signal.compute_signals(_make_ohlcv())
        buys = out[out["buy"]]
        # every fired buy must satisfy the Bollinger-mid confirmation
        assert (buys["Close"] > buys["bb_mid"]).all()

    def test_vwma_weights_by_volume(self):
        # constant series -> VWMA equals the constant regardless of volume
        s = pd.Series([5.0] * 20)
        v = pd.Series(np.arange(1, 21, dtype=float))
        out = dip_signal._vwma(s, v, 7).dropna()
        assert np.allclose(out.values, 5.0)


@pytest.mark.unit
class TestAnalyze:
    def test_handles_no_data_gracefully(self, monkeypatch):
        class _Empty:
            def history(self, **kw):
                return pd.DataFrame()
        monkeypatch.setattr(dip_signal.yf, "Ticker", lambda *_a, **_k: _Empty())
        res = dip_signal.analyze("ZZZZ.IS")
        assert res.ok is False
        assert "yok" in res.error.lower() or "alınamadı" in res.error.lower()

    def test_returns_status_and_signals(self, monkeypatch):
        df = _make_ohlcv(150)

        class _Stub:
            def history(self, **kw):
                return df
        monkeypatch.setattr(dip_signal.yf, "Ticker", lambda *_a, **_k: _Stub())
        res = dip_signal.analyze("THYAO.IS", "Günlük (1g)")
        assert res.ok is True
        assert res.status in ("AL BÖLGESİ", "SAT UYARISI", "NÖTR")
        assert isinstance(res.signals, list)
        assert set(res.conditions.keys())  # non-empty condition map


@pytest.mark.unit
class TestScan:
    def _fake_result(self, ticker, status, met):
        # minimal StrategyResult: df with one row + conditions matching `met`
        df = pd.DataFrame({"Close": [100.0], "smi": [42.0]})
        conds = {f"k{i}": (i < met) for i in range(3)}
        return dip_signal.StrategyResult(
            ok=True, ticker=ticker, interval_label="Günlük (1g)",
            df=df, status=status, conditions=conds, signals=[])

    def test_sorts_al_first_then_by_met(self, monkeypatch):
        canned = {
            "A.IS": self._fake_result("A.IS", "NÖTR", 2),
            "B.IS": self._fake_result("B.IS", "AL BÖLGESİ", 3),
            "C.IS": self._fake_result("C.IS", "SAT UYARISI", 0),
            "D.IS": self._fake_result("D.IS", "NÖTR", 3),
        }
        monkeypatch.setattr(dip_signal, "analyze", lambda tk, lbl=None: canned[tk])
        out = dip_signal.scan(list(canned.keys()), max_workers=2)
        assert [r["ticker"] for r in out] == ["B.IS", "D.IS", "A.IS", "C.IS"]
        assert out[0]["status"] == "AL BÖLGESİ" and out[0]["met"] == 3

    def test_failed_ticker_goes_last_and_no_raise(self, monkeypatch):
        def fake(tk, lbl=None):
            if tk == "BAD.IS":
                return dip_signal.StrategyResult(False, tk, "Günlük (1g)", error="yok")
            return self._fake_result(tk, "AL BÖLGESİ", 3)
        monkeypatch.setattr(dip_signal, "analyze", fake)
        out = dip_signal.scan(["BAD.IS", "GOOD.IS"], max_workers=2)
        assert out[0]["ticker"] == "GOOD.IS"
        assert out[-1]["ticker"] == "BAD.IS" and out[-1]["ok"] is False

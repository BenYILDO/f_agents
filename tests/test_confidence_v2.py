"""Teknik güven katmanı v2 — risk, teyit, MTF, backtest (saf/ağsız testler)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.analytics import backtest, confirmation, multiframe, risk


def _uptrend_df(n: int, start: float = 100.0, step: float = 1.0, vol: float = 1e6):
    close = np.arange(n) * step + start
    return pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": np.full(n, vol),
    })


@pytest.mark.unit
class TestRisk:
    def test_stop_target_rr(self):
        r = risk.compute_risk(_uptrend_df(80), atr_mult=2.0, target_rr=2.0)
        assert r.ok
        assert r.stop < r.close < r.target
        assert r.rr == pytest.approx(2.0, abs=0.01)   # hedef = close + 2*risk
        assert r.liquidity == "yüksek"                # 100×1e6 TL

    def test_target_price_overrides_rr(self):
        df = _uptrend_df(80)
        close = float(df["Close"].iloc[-1])
        r = risk.compute_risk(df, atr_mult=2.0, target_price=close + 100)
        assert r.target == pytest.approx(close + 100)
        assert r.rr > 2.0

    def test_low_liquidity_flagged(self):
        r = risk.compute_risk(_uptrend_df(80, vol=500), atr_mult=2.0)
        assert r.liquidity == "düşük"
        assert any("likidite" in n.lower() for n in r.notes)


@pytest.mark.unit
class TestConfirmation:
    def test_bullish_divergence_detected(self):
        # Fiyat: iki swing dip; ikincisi daha düşük (6 < 7)
        price = pd.Series([10, 9, 8, 7, 8, 9, 10, 9, 8, 6, 7, 8, 9, 10, 11])
        # Osilatör: ikinci dipte daha yüksek (30 > 20) → boğa uyumsuzluğu
        osc = pd.Series([50] * 15)
        osc.iloc[3], osc.iloc[9] = 20, 30
        assert confirmation._divergence(price, osc, k=3) == "pozitif (boğa)"

    def test_no_divergence(self):
        price = pd.Series([10, 9, 8, 7, 8, 9, 10, 9, 8, 6, 7, 8, 9, 10, 11])
        # Osilatör daha düşük dipte daha da düşük (fiyatla uyumlu) → uyumsuzluk yok
        osc = pd.Series([50] * 15, dtype=float)
        osc.iloc[3], osc.iloc[9] = 20, 15
        assert confirmation._divergence(price, osc, k=3) == "yok"

    def test_relative_strength_and_volume(self):
        stock = _uptrend_df(70, step=1.0)          # güçlü yükseliş
        bench = _uptrend_df(70, step=0.2)          # zayıf yükseliş
        stock.loc[stock.index[-1], "Volume"] = 5e6  # son bar yüksek hacim
        res = confirmation.compute_confirmation(stock, benchmark_df=bench, rs_window=60)
        assert res.ok
        assert res.rel_strength is not None and res.rel_strength > 0
        assert res.volume_confirms is True
        assert res.score > 0


@pytest.mark.unit
class TestMultiframe:
    def test_frame_direction_up_down(self):
        # Gerçekçi (gürültülü) trend — saf monotonik seride RSI tanımsız olur
        i = np.arange(220)
        up_c = 100 + i + np.sin(i) * 5      # gerçek iniş günleri olsun (RSI tanımlı)
        down_c = 320 - i + np.sin(i) * 5
        up_df = pd.DataFrame({"Open": up_c, "High": up_c + 1, "Low": up_c - 1,
                              "Close": up_c, "Volume": 1e6})
        down_df = pd.DataFrame({"Open": down_c, "High": down_c + 1, "Low": down_c - 1,
                                "Close": down_c, "Volume": 1e6})
        assert multiframe._frame_direction(up_df)[0] == 1
        assert multiframe._frame_direction(down_df)[0] == -1

    def test_aggregate_levels(self):
        all_up = {l: {"dir": 1, "detail": ""} for l in ("Haftalık", "Günlük", "4 Saatlik")}
        assert multiframe.aggregate(all_up).confluence == "güçlü AL"
        assert multiframe.aggregate(all_up).score == pytest.approx(1.0, abs=0.01)

        mixed = {"Haftalık": {"dir": -1, "detail": ""}, "Günlük": {"dir": 1, "detail": ""}}
        assert "karışık" in multiframe.aggregate(mixed).confluence


@pytest.mark.unit
class TestBacktest:
    def test_signal_edge_uptrend(self):
        close = pd.Series(np.arange(100, dtype=float) + 100)  # düz yükseliş
        entries = pd.Series(False, index=close.index)
        exits = pd.Series(False, index=close.index)
        entries.iloc[10] = True
        entries.iloc[50] = True
        exits.iloc[20] = True
        exits.iloc[60] = True
        res = backtest.signal_edge(close, entries, exits, horizons=(5, 10))
        assert res.ok
        assert res.n_signals == 2
        assert res.fwd["+5g"]["hit_rate"] == 100.0     # yükselişte hep pozitif
        assert res.equity["trades"] == 2
        assert res.equity["win_rate"] == 100.0
        assert res.equity["max_drawdown"] <= 0.0

    def test_no_signals(self):
        close = pd.Series(np.arange(100, dtype=float) + 100)
        empty = pd.Series(False, index=close.index)
        res = backtest.signal_edge(close, empty, empty)
        assert res.ok and res.n_signals == 0

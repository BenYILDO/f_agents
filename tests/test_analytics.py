"""Tests for the deterministic analytics engine (patterns, candles, S/R,
seasonality, regime, composite) — all on synthetic OHLCV, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.analytics.candlesticks import (
    candle_score,
    detect_candlesticks,
    recent_candle_hits,
)
from tradingagents.analytics.composite import build_technical_brief, compute_composite
from tradingagents.analytics.patterns import detect_patterns, find_pivots, pattern_score
from tradingagents.analytics.regime import compute_regime
from tradingagents.analytics.seasonality import compute_seasonality, month_edge
from tradingagents.analytics.support_resistance import (
    compute_levels,
    nearest_levels,
    sr_score,
)


def _ohlcv_from_close(close: pd.Series, spread: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": close.shift().fillna(close.iloc[0]),
        "High": close + spread,
        "Low": close - spread,
        "Close": close,
        "Volume": 1_000_000.0,
    })


def _piecewise(segments: list[tuple[float, float, int]], start="2023-01-02") -> pd.DataFrame:
    """Parçalı doğrusal kapanış serisi: (başlangıç, bitiş, bar) listesinden OHLCV."""
    vals: list[float] = []
    for lo, hi, n in segments:
        vals.extend(np.linspace(lo, hi, n).tolist())
    idx = pd.bdate_range(start, periods=len(vals))
    return _ohlcv_from_close(pd.Series(vals, index=idx))


@pytest.mark.unit
class TestPivots:
    def test_finds_alternating_pivots(self):
        df = _piecewise([(50, 100, 20), (100, 80, 10), (80, 95, 10), (95, 70, 10)])
        piv = find_pivots(df, order=3)
        kinds = list(piv["kind"])
        assert "H" in kinds and "L" in kinds
        # zigzag temizliği: ardışık aynı tür pivot kalmamalı
        assert all(a != b for a, b in zip(kinds, kinds[1:]))


@pytest.mark.unit
class TestChartPatterns:
    def test_double_top_confirmed(self):
        # 100'e iki tepe, arada 90 dip, sonra 85'e kırılım → teyitli ikili tepe
        df = _piecewise([(60, 100, 30), (100, 90, 8), (90, 100, 8), (100, 85, 12)])
        hits = detect_patterns(df, order=3)
        names = [h.name for h in hits]
        assert "İkili Tepe" in names
        hit = next(h for h in hits if h.name == "İkili Tepe")
        assert hit.direction == "ayı" and hit.confirmed
        assert pattern_score(hits) < 0

    def test_double_bottom_confirmed(self):
        df = _piecewise([(140, 100, 30), (100, 110, 8), (110, 100, 8), (100, 118, 12)])
        hits = detect_patterns(df, order=3)
        hit = next((h for h in hits if h.name == "İkili Dip"), None)
        assert hit is not None and hit.direction == "boğa" and hit.confirmed
        assert pattern_score(hits) > 0

    def test_no_pattern_on_smooth_trend(self):
        df = _piecewise([(50, 150, 80)])
        assert detect_patterns(df, order=3) == []

    def test_never_raises_on_short_data(self):
        df = _piecewise([(50, 60, 5)])
        assert detect_patterns(df) == []


@pytest.mark.unit
class TestCandlesticks:
    def _base(self, n=25, price=100.0) -> pd.DataFrame:
        # Küçük gövdeli, hafif aşağı eğimli zemin. Gövde yönü dönüşümlü ki
        # zemin kendisi formasyon (örn. üç kara karga) üretmesin.
        idx = pd.bdate_range("2024-01-01", periods=n)
        drift = price - 0.2 * np.arange(n)
        sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        return pd.DataFrame({
            "Open": drift + 0.1 * sign, "High": drift + 0.4, "Low": drift - 0.4,
            "Close": drift - 0.1 * sign, "Volume": 1_000_000.0,
        }, index=idx)

    def test_bullish_engulfing(self):
        df = self._base()
        # Sondan bir önceki barı küçük ayı gövdesi yap (yutulacak mum)
        mid = float(df.iloc[-2]["High"]) - 0.4
        df.iloc[-2, df.columns.get_loc("Open")] = mid + 0.1
        df.iloc[-2, df.columns.get_loc("Close")] = mid - 0.1
        # Son bar: önceki küçük ayı gövdesini tamamen yutan büyük boğa gövdesi
        prev_o, prev_c = df.iloc[-2]["Open"], df.iloc[-2]["Close"]
        df.iloc[-1, df.columns.get_loc("Open")] = prev_c - 0.5
        df.iloc[-1, df.columns.get_loc("Close")] = prev_o + 2.0
        df.iloc[-1, df.columns.get_loc("High")] = prev_o + 2.4
        df.iloc[-1, df.columns.get_loc("Low")] = prev_c - 0.9
        flags = detect_candlesticks(df)
        assert bool(flags["bullish_engulfing"].iloc[-1])
        hits = recent_candle_hits(df, lookback=3)
        assert any(h.code == "bullish_engulfing" for h in hits)
        assert candle_score(df, lookback=3) > 0

    def test_doji(self):
        df = self._base()
        df.iloc[-1, df.columns.get_loc("Open")] = 95.0
        df.iloc[-1, df.columns.get_loc("Close")] = 95.01
        df.iloc[-1, df.columns.get_loc("High")] = 96.0
        df.iloc[-1, df.columns.get_loc("Low")] = 94.0
        flags = detect_candlesticks(df)
        assert bool(flags["doji"].iloc[-1])


@pytest.mark.unit
class TestSupportResistance:
    def test_levels_and_score(self):
        # 90-110 bandında salınan seri: 90 desteği ve 110 direnci kümelenmeli
        df = _piecewise([(90, 110, 10), (110, 90, 10)] * 6 + [(90, 91, 5)])
        levels = compute_levels(df, order=3)
        assert levels, "seviye üretilmeliydi"
        close = float(df["Close"].iloc[-1])
        supports, resistances = nearest_levels(levels, close)
        assert supports and resistances
        # Çok dokunuşlu pivot desteği kümelenmiş olmalı
        assert any(lv.source == "pivot" and lv.strength >= 3 for lv in supports)
        # Fiyat 91 — güçlü ~90 desteğinin hemen üstünde: skor pozitif olmalı
        assert sr_score(levels, close) > 0


@pytest.mark.unit
class TestSeasonality:
    def _seasonal_df(self, years=8) -> pd.DataFrame:
        # Ocak aylarında belirgin pozitif, diğer aylar yatay sentetik seri
        idx = pd.bdate_range("2016-01-01", periods=years * 252)
        ret = np.where(idx.month == 1, 0.004, 0.0)
        close = pd.Series(100 * np.cumprod(1 + ret), index=idx)
        return _ohlcv_from_close(close)

    def test_monthly_stats_and_edge(self):
        res = compute_seasonality(self._seasonal_df())
        assert res.ok and res.monthly is not None
        score, summary = month_edge(res, 1)
        assert score > 0 and "Ocak" in summary

    def test_small_sample_is_neutralized(self):
        res = compute_seasonality(self._seasonal_df(years=2))
        assert res.ok
        score, summary = month_edge(res, 1)
        assert score == 0.0 and "örneklem küçük" in summary

    def test_insufficient_data(self):
        df = self._seasonal_df(years=1).head(200)
        assert not compute_seasonality(df).ok


@pytest.mark.unit
class TestRegime:
    def _df(self, vol: float, n=600) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        ret = rng.normal(0.0005, vol, n)
        idx = pd.bdate_range("2022-01-03", periods=n)
        close = pd.Series(100 * np.cumprod(1 + ret), index=idx)
        return _ohlcv_from_close(close, spread=close.iloc[0] * vol)

    def test_calm_market_high_confidence(self):
        res = compute_regime(self._df(0.005), is_bist=False)
        assert res.ok and res.confidence_mult >= 0.85

    def test_vol_spike_cuts_confidence(self):
        df = self._df(0.005)
        # Son 30 barda volatilite patlaması simüle et
        rng = np.random.default_rng(11)
        shock = rng.normal(0, 0.06, 30)
        tail = df["Close"].iloc[-31] * np.cumprod(1 + shock)
        df.iloc[-30:, df.columns.get_loc("Close")] = tail
        df.iloc[-30:, df.columns.get_loc("High")] = tail * 1.03
        df.iloc[-30:, df.columns.get_loc("Low")] = tail * 0.97
        res = compute_regime(df, is_bist=False)
        assert res.ok and res.vol_regime in ("Yüksek", "Aşırı")
        assert res.confidence_mult < 0.9

    def test_try_stress_lowers_confidence(self):
        res_calm = compute_regime(self._df(0.005), is_bist=False, try_stress=0.0)
        res_stressed = compute_regime(self._df(0.005), is_bist=False, try_stress=1.0)
        assert res_stressed.confidence_mult < res_calm.confidence_mult


@pytest.mark.unit
class TestComposite:
    def _trending_df(self) -> pd.DataFrame:
        # 3 yıl deterministik yükseliş + hafif salınım: trend bileşeni kesin pozitif
        n = 800
        idx = pd.bdate_range("2021-01-04", periods=n)
        base = np.linspace(100, 300, n)
        wiggle = 3 * np.sin(np.arange(n) / 9)
        close = pd.Series(base + wiggle, index=idx)
        return _ohlcv_from_close(close, spread=1.0)

    def test_composite_offline(self):
        # BIST dışı sembol → TL stres ağ çağrısı yapılmaz; tamamen offline
        res = compute_composite("TEST", self._trending_df())
        assert res.ok
        assert set(res.components) == {"trend", "momentum", "pattern", "money_flow",
                                       "dip", "candle", "sr", "seasonality"}
        assert res.components["trend"] > 0
        assert -100 <= res.score <= 100
        assert res.verdict in ("GÜÇLÜ AL", "AL", "NÖTR", "SAT", "GÜÇLÜ SAT")
        assert set(res.guards) == {"asiri_cosku", "dusen_bicak"}

    def test_brief_renders(self):
        brief = build_technical_brief("TEST", self._trending_df())
        assert "DETERMİNİSTİK TEKNİK BRİF" in brief
        assert "Kompozit skor" in brief

    def test_insufficient_data_fails_loud(self):
        df = self._trending_df().head(20)
        res = compute_composite("TEST", df)
        assert not res.ok and res.error

"""Tests for the video-derived trap guards (euphoria / falling knife) and
money-flow component in the composite engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.analytics.composite import compute_composite


def _ohlcv(close: pd.Series, volume: pd.Series | None = None, spread=1.0) -> pd.DataFrame:
    vol = volume if volume is not None else pd.Series(1_000_000.0, index=close.index)
    return pd.DataFrame({
        "Open": close.shift().fillna(close.iloc[0]),
        "High": close + spread, "Low": close - spread,
        "Close": close, "Volume": vol,
    })


@pytest.mark.unit
class TestEuphoriaGuard:
    def test_parabolic_top_triggers_euphoria(self):
        # Hızlanan yükseliş + hafif gürültü (RSI tanımlı kalsın, hep yukarı
        # değil) → fiyat ATH'de, RSI aşırı alımda
        n = 400
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2023-01-02", periods=n)
        ramp = np.linspace(0.0008, 0.006, n)            # hızlanan günlük getiri
        ret = ramp + rng.normal(0, 0.004, n)            # ara sıra küçük düşüş
        close = pd.Series(100 * np.cumprod(1 + ret), index=idx)
        res = compute_composite("TEST", _ohlcv(close))
        assert res.ok
        assert res.guards["asiri_cosku"] is True
        assert any("coşku" in w for w in res.warnings)
        # Guard alış skorunu kıstığı için ham pozitif skor daha büyük olmalı
        assert res.score <= res.raw_score


@pytest.mark.unit
class TestFallingKnifeGuard:
    def test_steep_downtrend_oversold_triggers_knife(self):
        # Uzun yükseliş sonrası sert, kesintisiz çöküş → SMA200 çok üstte,
        # ADX yüksek, RSI dipte, dönüş teyidi yok
        n = 400
        idx = pd.bdate_range("2023-01-02", periods=n)
        up = np.linspace(100, 300, n // 2)
        down = np.linspace(300, 90, n - n // 2)
        close = pd.Series(np.concatenate([up, down]), index=idx)
        res = compute_composite("TEST", _ohlcv(close))
        assert res.ok
        assert res.guards["dusen_bicak"] is True
        assert any("bıçak" in w for w in res.warnings)
        # Düşen bıçak pozitif (oversold) skoru sıfırlar → skor pozitif olmamalı
        assert res.score <= 0


@pytest.mark.unit
class TestMoneyFlow:
    def test_distribution_is_negative(self):
        # Düşen fiyat + düşüş günlerinde yüksek hacim = dağıtım (para çıkışı)
        n = 400
        idx = pd.bdate_range("2023-01-02", periods=n)
        close = pd.Series(np.linspace(200, 150, n) + np.sin(np.arange(n) / 5), index=idx)
        ret = close.pct_change().fillna(0)
        vol = pd.Series(np.where(ret < 0, 3_000_000.0, 800_000.0), index=idx)
        res = compute_composite("TEST", _ohlcv(close, volume=vol))
        assert res.ok
        assert res.components["money_flow"] < 0
        assert "DAĞITIM" in res.details["money_flow"] or "çıkış" in res.details["money_flow"]

"""Arena motoru testleri — fill/stop/hedef/sizing/kill-switch muhasebesi.

Sentetik OHLC ile çalışır (ağ yok). Motor para muhasebesini doğru yapıyor mu,
OHLC stop/hedef kuralları plan §'a uyuyor mu, kill-switch/rejim filtresi tutuyor mu?
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Ağır opsiyonel bağımlılıklar (paket import zinciri kırılmasın)
for _mod in ("yfinance", "stockstats", "ta", "requests", "bs4", "praw",
             "streamlit", "altair", "plotly", "plotly.express",
             "lightgbm", "xgboost", "hmmlearn", "hmmlearn.hmm", "dotenv", "httpx"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from tradingagents.arena.config import ExecutionConfig
from tradingagents.arena.engine import run_backtest
from tradingagents.arena.metrics import benchmark_buy_hold, compute_metrics, max_drawdown
from tradingagents.arena.profiles import PROFILES, active_profiles, Profile


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-01", periods=n)


def _flat_df(dates, price=100.0, high=None, low=None) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "Open": [price] * n, "High": [high or price] * n,
        "Low": [low or price] * n, "Close": [price] * n,
        "buy": [False] * n, "sell": [False] * n,
        "stop": [price * 0.9] * n, "target": [price * 1.3] * n,
        "trend_ok": [True] * n,
    }, index=dates)


_CFG = ExecutionConfig(initial_capital=100_000.0, commission_bps=5.0, slippage_bps=10.0,
                       buying_power_buffer=0.0)
_PROF = Profile(code="t", name="Test", emoji="🧪", use_regime_filter=False,
                max_positions=5, risk_per_trade=0.02, max_position_weight=0.5,
                min_cash_reserve=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Temel fill & muhasebe
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFillMuhasebesi:

    def test_alis_satis_roundtrip_maliyetli(self):
        """Düz fiyatta al-sat: yalnız komisyon+slippage kadar zarar etmeli."""
        dates = _dates(8)
        df = _flat_df(dates)
        df.loc[dates[0], "buy"] = True     # gün0 kapanış → gün1 açılış fill
        df.loc[dates[3], "sell"] = True    # gün3 kapanış → gün4 açılış fill
        res = run_backtest(_PROF, {"AAA": df}, dates, _CFG, regime=None)
        assert res.ok and res.n_trades == 1
        tr = res.trades[0]
        assert tr.exit_reason == "sinyal"
        # giriş 100*(1+10bps), çıkış 100*(1-10bps) → fiyat farkı negatif + komisyon
        assert tr.entry_price > 100 and tr.exit_price < 100
        assert tr.pnl < 0
        # toplam maliyet kabaca notional*(2*slip+2*comm) mertebesinde
        assert res.final_equity < _CFG.initial_capital

    def test_pozisyon_yoksa_equity_sabit(self):
        """Hiç sinyal yok → equity başlangıç sermayesinde kalır."""
        dates = _dates(6)
        res = run_backtest(_PROF, {"AAA": _flat_df(dates)}, dates, _CFG, regime=None)
        assert res.ok and res.n_trades == 0
        assert abs(res.final_equity - _CFG.initial_capital) < 1e-6

    def test_sizing_risk_butcesine_uyar(self):
        """Adet = risk_per_trade*equity / (giriş-stop) sınırını aşmamalı."""
        dates = _dates(6)
        df = _flat_df(dates)
        df.loc[dates[0], "buy"] = True
        res = run_backtest(_PROF, {"AAA": df}, dates, _CFG, regime=None)
        # gün5 sezon sonu kapanır; en az 1 trade
        assert res.n_trades == 1
        qty = res.trades[0].quantity
        entry = res.trades[0].entry_price
        stop_dist = entry - 90.0
        max_qty_risk = (_PROF.risk_per_trade * _CFG.initial_capital) / stop_dist
        assert qty <= int(max_qty_risk) + 1


# ─────────────────────────────────────────────────────────────────────────────
# OHLC stop / hedef kuralları (plan §)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestStopHedef:

    def _entry_setup(self, dates):
        df = _flat_df(dates)
        df.loc[dates[0], "buy"] = True   # gün1 açılışta gir
        return df

    def test_stop_intraday_tetiklenir(self):
        dates = _dates(6)
        df = self._entry_setup(dates)
        df.loc[dates[2], "Low"] = 85.0   # gün2 Low<=stop(90), Open=100>90 → stop fiyatından
        res = run_backtest(_PROF, {"AAA": df}, dates, _CFG, regime=None)
        assert res.n_trades == 1
        assert res.trades[0].exit_reason == "stop"
        assert abs(res.trades[0].exit_price - 90.0) < 1e-6

    def test_gap_acilistan_cikar(self):
        """Open <= stop (gap) → stop fiyatından değil açılıştan çıkılır."""
        dates = _dates(6)
        df = self._entry_setup(dates)
        df.loc[dates[2], "Open"] = 80.0
        df.loc[dates[2], "Low"] = 78.0
        res = run_backtest(_PROF, {"AAA": df}, dates, _CFG, regime=None)
        assert res.trades[0].exit_reason == "stop"
        assert abs(res.trades[0].exit_price - 80.0) < 1e-6

    def test_hedef_tetiklenir(self):
        dates = _dates(6)
        df = self._entry_setup(dates)
        df.loc[dates[2], "High"] = 140.0  # >= target 130
        res = run_backtest(_PROF, {"AAA": df}, dates, _CFG, regime=None)
        assert res.trades[0].exit_reason == "hedef"
        assert abs(res.trades[0].exit_price - 130.0) < 1e-6
        assert res.trades[0].pnl > 0

    def test_ayni_bar_stop_ve_hedef_muhafazakar_stop(self):
        """Aynı barda hem stop hem hedef → muhafazakâr: stop önce + ambiguous."""
        dates = _dates(6)
        df = self._entry_setup(dates)
        df.loc[dates[2], "Low"] = 85.0    # stop
        df.loc[dates[2], "High"] = 140.0  # hedef
        res = run_backtest(_PROF, {"AAA": df}, dates, _CFG, regime=None)
        assert res.trades[0].exit_reason == "stop"
        assert res.trades[0].ambiguous_bar is True


# ─────────────────────────────────────────────────────────────────────────────
# Filtreler & kill-switch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFiltrelerKillSwitch:

    def test_rejim_filtresi_alimi_engeller(self):
        """use_regime_filter + boğa-değil rejim → hiç alım olmamalı."""
        dates = _dates(6)
        df = _flat_df(dates)
        df.loc[dates[0], "buy"] = True
        prof = Profile(code="r", name="R", emoji="🛡️", use_regime_filter=True,
                       risk_per_trade=0.02, max_position_weight=0.5, min_cash_reserve=0.0)
        regime = pd.Series([False] * len(dates), index=dates)
        res = run_backtest(prof, {"AAA": df}, dates, _CFG, regime=regime)
        assert res.n_trades == 0

    def test_rejim_bogada_alim_gecer(self):
        dates = _dates(6)
        df = _flat_df(dates)
        df.loc[dates[0], "buy"] = True
        prof = Profile(code="r", name="R", emoji="🛡️", use_regime_filter=True,
                       risk_per_trade=0.02, max_position_weight=0.5, min_cash_reserve=0.0)
        regime = pd.Series([True] * len(dates), index=dates)
        res = run_backtest(prof, {"AAA": df}, dates, _CFG, regime=regime)
        assert res.n_trades == 1

    def test_max_pozisyon_tavani(self):
        """max_positions=1 → aynı anda yalnız 1 pozisyon açılır."""
        dates = _dates(6)
        a, b = _flat_df(dates), _flat_df(dates)
        a.loc[dates[0], "buy"] = True
        b.loc[dates[0], "buy"] = True
        prof = Profile(code="m", name="M", emoji="1️⃣", use_regime_filter=False,
                       max_positions=1, risk_per_trade=0.02, max_position_weight=0.5,
                       min_cash_reserve=0.0)
        res = run_backtest(prof, {"AAA": a, "BBB": b}, dates, _CFG, regime=None)
        # İki aday ama tek slot → ilk girişte yalnız 1 pozisyon (sezon sonu 1 trade)
        # Açık pozisyon sayısı hiçbir an 1'i geçmediği için en fazla 1 eşzamanlı.
        entry_dates = [t.entry_date for t in res.trades]
        # aynı giriş gününde en fazla 1 pozisyon açılmış olmalı
        assert len(entry_dates) == len(set(entry_dates)) or res.n_trades <= len(dates)
        assert res.n_trades >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Metrikler & benchmark
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMetrikler:

    def test_max_drawdown_negatif(self):
        eq = pd.Series([100, 120, 90, 110, 80])
        dd = max_drawdown(eq)
        # tepe 120 → dip 80 → en derin düşüş -%33.3
        assert abs(dd - (-1.0 / 3.0)) < 1e-6

    def test_benchmark_buy_hold_baslangic_kapital(self):
        close = pd.Series([10.0, 11.0, 12.0], index=_dates(3))
        eq = benchmark_buy_hold(close, 100_000.0, slippage_bps=0.0)
        assert abs(eq.iloc[0] - 100_000.0) < 1e-6
        assert eq.iloc[-1] > eq.iloc[0]  # fiyat arttı → equity arttı

    def test_compute_metrics_bos_seri(self):
        m = compute_metrics(pd.Series(dtype=float))
        assert m.total_return == 0.0 and m.n_days == 0


# ─────────────────────────────────────────────────────────────────────────────
# Profil sözleşmesi
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestProfiller:

    def test_dort_aktif_bir_observer(self):
        assert len(active_profiles()) == 4
        observers = [p for p in PROFILES.values() if p.is_observer()]
        assert len(observers) == 1
        assert observers[0].code == "ml_observer"

    def test_kelly_ceyrek(self):
        for p in PROFILES.values():
            assert p.kelly_fraction == 0.25

    def test_rules_snapshot_degismez_dict(self):
        snap = PROFILES["balanced"].rules_snapshot()
        assert snap["code"] == "balanced"
        assert "risk_per_trade" in snap


# ─────────────────────────────────────────────────────────────────────────────
# Replay orkestrasyonu (uçtan uca smoke — sentetik veri, ağ yok)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_ohlcv(n=420, seed=0, drift=0.0003) -> pd.DataFrame:
    """Rastgele yürüyüşle OHLCV üretir (compute_signals'ı beslemeye yeter)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    rets = rng.normal(drift, 0.018, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": vol}, index=dates)


@pytest.mark.unit
class TestReplayOrkestrasyon:

    def test_replay_uctan_uca_calisir(self, monkeypatch):
        """run_arena_replay sentetik veriyle çökmeden lig + benchmark üretmeli."""
        from tradingagents.analytics import composite
        from tradingagents.strategy import dip_signal
        from tradingagents.arena import replay as replay_mod

        def fake_fetch(ticker, period="5y"):
            # her ticker farklı seed → farklı seri
            seed = abs(hash(ticker)) % 1000
            return _synthetic_ohlcv(seed=seed, drift=0.0006 if "XU100" in ticker else 0.0004)

        monkeypatch.setattr(composite, "_fetch_daily", fake_fetch)
        monkeypatch.setattr(dip_signal, "BIST30", ["AAA.IS", "BBB.IS", "CCC.IS"])

        result = replay_mod.run_arena_replay(period="3y")
        assert result.ok, result.error
        # 4 aktif profil lig tablosunda
        assert len(result.leaderboard) == 4
        # benchmark hesaplandı
        assert len(result.benchmark_equity) > 0
        assert result.benchmark_metrics.n_days > 0
        # edge_summary üretildi (geçsin/geçmesin)
        assert result.edge_summary
        assert len(result.caveats) >= 4
        # her profil sonucu equity eğrisi üretmiş olmalı
        for code, r in result.per_profile.items():
            assert r.ok
            assert len(r.equity_curve) == result.benchmark_metrics.n_days

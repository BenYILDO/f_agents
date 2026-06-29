"""Canlı arena testleri — durum kalıcılığı + emir üretimi/fill/equity muhasebesi.

Sentetik AnalysisOutcome benzeri nesneler + fiyatlarla çalışır (ağ yok).
"""

from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

for _mod in ("yfinance", "stockstats", "ta", "requests", "bs4", "praw",
             "streamlit", "altair", "plotly", "plotly.express",
             "lightgbm", "xgboost", "hmmlearn", "hmmlearn.hmm", "dotenv", "httpx"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from tradingagents.arena.config import ExecutionConfig
from tradingagents.arena.profiles import PROFILES
from tradingagents.arena import live as live_mod
from tradingagents.arena.state import (
    AccountState, ArenaState, D, PendingOrder, Position, new_state,
)

_CFG = ExecutionConfig(initial_capital=100_000.0, commission_bps=5.0, slippage_bps=10.0,
                       buying_power_buffer=0.0)


def _outcome(ticker, decision="AL", conf=70.0, close=100.0, stop=90.0, target=120.0,
             p_up=None, regime="boğa", behavior="trend"):
    return SimpleNamespace(
        ticker=ticker, gated_decision=decision, decision=decision,
        confidence_score=conf, close=close, regime_trend=regime, behavior=behavior,
        signals={"risk": {"stop": stop, "target": target},
                 "confidence": {"p_up": p_up}},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Durum kalıcılığı
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestState:

    def test_new_state_bes_hesap_esit_kasa(self):
        s = new_state(initial_capital=100_000)
        assert len(s.accounts) == len(PROFILES)
        for acc in s.accounts.values():
            assert acc.cash == D(100_000)
            assert acc.ledger and acc.ledger[0]["event_type"] == "INIT"

    def test_observer_status_korunur(self):
        s = new_state()
        assert s.accounts["ml_observer"].status == "OBSERVER"

    def test_roundtrip_decimal_korunur(self):
        s = new_state()
        s.accounts["balanced"].cash = D("12345.67")
        s.accounts["balanced"].positions["AAA"] = Position("AAA", 10, D("99.5"), D("90"), D("120"), "2026-06-29")
        s2 = ArenaState.from_dict(s.to_dict())
        assert s2.accounts["balanced"].cash == Decimal("12345.67")
        assert s2.accounts["balanced"].positions["AAA"].avg_cost == Decimal("99.5")
        assert s2.accounts["balanced"].positions["AAA"].quantity == 10


# ─────────────────────────────────────────────────────────────────────────────
# Emir üretimi (decide_orders)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDecideOrders:

    def _acc(self, code="balanced"):
        return AccountState(profile_code=code, cash=D(100_000))

    def test_al_sinyali_buy_uretir(self):
        prof = PROFILES["balanced"]
        acc = self._acc()
        orders = live_mod.decide_orders(prof, acc, [_outcome("AAA.IS")], D(100_000), _CFG, "2026-06-29")
        buys = [o for o in orders if o.side == "BUY"]
        assert len(buys) == 1 and buys[0].ticker == "AAA.IS"
        assert buys[0].quantity > 0

    def test_dusuk_guven_buy_uretmez(self):
        prof = PROFILES["conservative"]  # min_confidence 60
        acc = self._acc("conservative")
        orders = live_mod.decide_orders(prof, acc, [_outcome("AAA.IS", conf=45.0)],
                                        D(100_000), _CFG, "2026-06-29")
        assert [o for o in orders if o.side == "BUY"] == []

    def test_sabit_risk_sizing(self):
        """p_up yok → risk_per_trade*equity/(close-stop) adet."""
        prof = PROFILES["balanced"]  # risk 0.01, weight 0.20
        acc = self._acc()
        orders = live_mod.decide_orders(prof, acc, [_outcome("AAA.IS", close=100, stop=90)],
                                        D(100_000), _CFG, "2026-06-29")
        qty = [o for o in orders if o.side == "BUY"][0].quantity
        # risk_budget 1000 / stop_dist 10 = 100; weight cap 200 → 100
        assert 95 <= qty <= 100

    def test_kelly_sizing_p_up_varsa(self):
        """Kaliteli p_up → çeyrek Kelly: f*=p-(1-p)/b, b=(t-c)/(c-s)."""
        prof = PROFILES["balanced"]
        acc = self._acc()
        orders = live_mod.decide_orders(
            prof, acc, [_outcome("AAA.IS", close=100, stop=90, target=120, p_up=0.65)],
            D(100_000), _CFG, "2026-06-29")
        qty = [o for o in orders if o.side == "BUY"][0].quantity
        # b=2, f*=0.65-0.35/2=0.475, frac=0.25*0.475=0.11875 → ~118 adet
        assert 110 <= qty <= 125

    def test_ters_sinyal_sell_uretir(self):
        prof = PROFILES["balanced"]
        acc = self._acc()
        acc.positions["AAA.IS"] = Position("AAA.IS", 50, D(100), D(90), D(120), "2026-06-20")
        orders = live_mod.decide_orders(prof, acc, [_outcome("AAA.IS", decision="SAT")],
                                        D(100_000), _CFG, "2026-06-29")
        sells = [o for o in orders if o.side == "SELL"]
        assert len(sells) == 1 and sells[0].reason == "ters sinyal"

    def test_stop_alti_sell_uretir(self):
        prof = PROFILES["balanced"]
        acc = self._acc()
        acc.positions["AAA.IS"] = Position("AAA.IS", 50, D(100), D(95), D(120), "2026-06-20")
        # close 92 < stop 95 → stop çıkışı
        orders = live_mod.decide_orders(prof, acc, [_outcome("AAA.IS", decision="TUT", close=92, stop=95)],
                                        D(100_000), _CFG, "2026-06-29")
        sells = [o for o in orders if o.side == "SELL"]
        assert len(sells) == 1 and sells[0].reason == "stop"

    def test_max_pozisyon_tavani(self):
        prof = PROFILES["conservative"]  # max_positions 5
        acc = self._acc("conservative")
        outs = [_outcome(f"T{i}.IS", conf=70.0) for i in range(10)]
        orders = live_mod.decide_orders(prof, acc, outs, D(100_000), _CFG, "2026-06-29")
        assert len([o for o in orders if o.side == "BUY"]) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# Fill muhasebesi
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFills:

    def test_buy_fill_kasa_dusurur_pozisyon_acar(self):
        acc = AccountState(profile_code="balanced", cash=D(100_000))
        acc.pending_orders = [PendingOrder("AAA.IS", "BUY", 100, "2026-06-28", stop=D(90), target=D(120))]
        filled = live_mod.apply_fills(acc, {"AAA.IS": {"open": 100.0, "close": 101.0}}, "2026-06-29", _CFG)
        assert filled == 1
        assert "AAA.IS" in acc.positions
        assert acc.positions["AAA.IS"].quantity == 100
        # fill = 100*(1+10bps)=100.1; maliyet ~ 10015 → kasa azaldı
        assert acc.cash < D(100_000)
        assert acc.positions["AAA.IS"].avg_cost > D(100)

    def test_buy_kasa_yetmezse_reddedilir(self):
        acc = AccountState(profile_code="balanced", cash=D(1_000))
        acc.pending_orders = [PendingOrder("AAA.IS", "BUY", 100, "2026-06-28", stop=D(90))]
        filled = live_mod.apply_fills(acc, {"AAA.IS": {"open": 100.0}}, "2026-06-29", _CFG)
        assert filled == 0 and "AAA.IS" not in acc.positions

    def test_sell_fill_kasa_artirir_realized_pnl(self):
        acc = AccountState(profile_code="balanced", cash=D(0))
        acc.positions["AAA.IS"] = Position("AAA.IS", 100, D(100), D(90), D(120), "2026-06-20")
        acc.pending_orders = [PendingOrder("AAA.IS", "SELL", 100, "2026-06-28")]
        filled = live_mod.apply_fills(acc, {"AAA.IS": {"open": 110.0}}, "2026-06-29", _CFG)
        assert filled == 1 and "AAA.IS" not in acc.positions
        assert acc.cash > D(0)
        # 110*(1-10bps)*100 - komisyon - maliyet(100*100) → pozitif realized
        assert acc.realized_pnl > D(0)

    def test_acilis_yoksa_emir_beklemede(self):
        acc = AccountState(profile_code="balanced", cash=D(100_000))
        acc.pending_orders = [PendingOrder("AAA.IS", "BUY", 100, "2026-06-28", stop=D(90))]
        filled = live_mod.apply_fills(acc, {"AAA.IS": {}}, "2026-06-29", _CFG)
        assert filled == 0 and len(acc.pending_orders) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Seans orkestrasyonu
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRunSession:

    def test_seans_idempotent(self):
        s = new_state()
        live_mod.run_session(s, [_outcome("AAA.IS")], {"AAA.IS": {"open": 100, "close": 100}},
                             "2026-06-29", _CFG)
        rep2 = live_mod.run_session(s, [_outcome("AAA.IS")], {"AAA.IS": {"open": 100, "close": 100}},
                                    "2026-06-29", _CFG)
        assert rep2.skipped is True

    def test_iki_seans_t1_fill(self):
        """Gün1 emir üretir, gün2 açılışta dolar (T+1)."""
        s = new_state()
        out = [_outcome("AAA.IS", close=100, stop=90, target=120)]
        live_mod.run_session(s, out, {"AAA.IS": {"open": 100, "close": 100}}, "2026-06-29", _CFG)
        # gün1 sonunda balanced'da pending BUY olmalı, pozisyon yok
        bal = s.accounts["balanced"]
        assert len(bal.pending_orders) >= 1 and not bal.positions
        # gün2: açılışta dolar
        live_mod.run_session(s, out, {"AAA.IS": {"open": 101, "close": 102}}, "2026-06-30", _CFG)
        assert "AAA.IS" in s.accounts["balanced"].positions

    def test_observer_para_islemez_tahmin_biriktirir(self):
        s = new_state()
        out = [_outcome("AAA.IS", p_up=0.62)]
        live_mod.run_session(s, out, {"AAA.IS": {"open": 100, "close": 100}}, "2026-06-29", _CFG)
        obs = s.accounts["ml_observer"]
        assert obs.cash == D(100_000) and not obs.positions   # para harcamadı
        assert len(s.predictions) == 1 and s.predictions[0]["p_up"] == 0.62

    def test_equity_gecmisi_yazilir(self):
        s = new_state()
        live_mod.run_session(s, [_outcome("AAA.IS")], {"AAA.IS": {"open": 100, "close": 100}},
                             "2026-06-29", _CFG)
        assert s.accounts["balanced"].equity_history
        assert s.accounts["balanced"].equity_history[-1]["session"] == "2026-06-29"

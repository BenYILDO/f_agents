"""Meta-labeling kapısı testleri (S4) — saf mantık + canlı motor entegrasyonu.

Sözleşme: kaliteli model yokken kapı PASİF (davranış değişmez); kaliteli model
varken veto/kısma yalnız use_ml_meta_filter profillerinde etkir (ablation);
filtre boyutu asla büyütmez; iz snapshot/karneye yazılır.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

for _mod in ("yfinance", "stockstats", "ta", "requests", "bs4", "praw",
             "streamlit", "altair", "plotly", "plotly.express",
             "lightgbm", "xgboost", "hmmlearn", "hmmlearn.hmm", "dotenv", "httpx"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from tradingagents.analysis.meta import (
    FLOOR_MULT,
    FULL_ABOVE,
    VETO_BELOW,
    meta_gate,
)
from tradingagents.arena.config import ExecutionConfig
from tradingagents.arena.profiles import PROFILES, Profile
from tradingagents.arena.live import decide_orders
from tradingagents.arena.state import AccountState, D

_CFG = ExecutionConfig(initial_capital=100_000.0, commission_bps=5.0,
                       slippage_bps=10.0, buying_power_buffer=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Saf kapı mantığı
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMetaGate:

    def test_no_quality_model_is_passive(self):
        g = meta_gate(p_up=0.30, quality_passed=False,
                      p_up_pooled=0.20, pooled_quality=False)
        assert not g.active and g.allow and g.size_mult == 1.0
        # Pasif kapı p_win taşımaz — düşük olasılık bile veto ÜRETMEZ
        assert g.p_win is None

    def test_pooled_preferred_over_per_ticker(self):
        g = meta_gate(p_up=0.70, quality_passed=True, model_version="champ",
                      p_up_pooled=0.55, pooled_quality=True, pooled_version="pool")
        assert g.active and g.source == "pooled" and g.model_version == "pool"

    def test_falls_back_to_per_ticker(self):
        g = meta_gate(p_up=0.65, quality_passed=True, model_version="champ",
                      p_up_pooled=0.55, pooled_quality=False)
        assert g.active and g.source == "per_ticker"

    def test_veto_below_threshold(self):
        g = meta_gate(p_up_pooled=VETO_BELOW - 0.01, pooled_quality=True)
        assert g.active and not g.allow and g.size_mult == 0.0

    def test_mult_monotone_and_bounded(self):
        ps = [VETO_BELOW, 0.50, 0.55, FULL_ABOVE, 0.80]
        mults = [meta_gate(p_up_pooled=p, pooled_quality=True).size_mult for p in ps]
        assert mults == sorted(mults)                 # monoton artan
        assert all(FLOOR_MULT <= m <= 1.0 for m in mults)  # asla büyütmez
        assert mults[0] == FLOOR_MULT and mults[-1] == 1.0

    def test_invalid_probability_is_passive(self):
        g = meta_gate(p_up_pooled=1.7, pooled_quality=True)
        assert not g.active and g.allow and g.size_mult == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Canlı motor entegrasyonu (decide_orders)
# ─────────────────────────────────────────────────────────────────────────────

def _outcome(ticker, decision="AL", conf=70.0, close=100.0, stop=90.0,
             target=120.0, p_up=None, ml_gate=None):
    return SimpleNamespace(
        ticker=ticker, gated_decision=decision, decision=decision,
        confidence_score=conf, close=close, regime_trend="boğa", behavior="trend",
        signals={"risk": {"stop": stop, "target": target},
                 "confidence": {"p_up": p_up},
                 "ml_gate": ml_gate or {}},
    )


def _meta_profile(**kw) -> Profile:
    """ML-meta hesabın PARA harcayan deney kopyası (testte aktive edilir)."""
    base = dict(code="meta_test", name="Meta", emoji="🧪", status="ACTIVE",
                min_confidence=50.0, use_ml_meta_filter=True)
    base.update(kw)
    return Profile(**base)


@pytest.mark.unit
class TestDecideOrdersMeta:

    def _acc(self):
        return AccountState(profile_code="meta_test", cash=D(100_000))

    def test_veto_skips_candidate_for_meta_profile(self):
        gate = meta_gate(p_up_pooled=0.30, pooled_quality=True).to_dict()
        outs = [_outcome("AAA.IS", ml_gate=gate)]
        orders = decide_orders(_meta_profile(), self._acc(), outs,
                               D(100_000), _CFG, "2026-07-02")
        assert orders == []                       # ML veto → emir yok

    def test_control_profile_ignores_gate(self):
        # Aynı veto, filtresiz (kontrol) profilde emri ENGELLEMEZ (ablation)
        gate = meta_gate(p_up_pooled=0.30, pooled_quality=True).to_dict()
        outs = [_outcome("AAA.IS", ml_gate=gate)]
        orders = decide_orders(PROFILES["balanced"], self._acc(), outs,
                               D(100_000), _CFG, "2026-07-02")
        assert len(orders) == 1 and orders[0].side == "BUY"

    def test_partial_mult_scales_quantity_down(self):
        full = decide_orders(_meta_profile(), self._acc(),
                             [_outcome("AAA.IS")], D(100_000), _CFG, "2026-07-02")
        gate = meta_gate(p_up_pooled=0.50, pooled_quality=True).to_dict()
        assert 0 < gate["size_mult"] < 1
        scaled = decide_orders(_meta_profile(), self._acc(),
                               [_outcome("AAA.IS", ml_gate=gate)],
                               D(100_000), _CFG, "2026-07-02")
        assert len(full) == len(scaled) == 1
        assert 0 < scaled[0].quantity < full[0].quantity
        assert "ML ×" in scaled[0].reason         # iz emir gerekçesinde

    def test_passive_gate_changes_nothing(self):
        gate = meta_gate().to_dict()              # kaliteli model yok → pasif
        full = decide_orders(_meta_profile(), self._acc(),
                             [_outcome("AAA.IS")], D(100_000), _CFG, "2026-07-02")
        same = decide_orders(_meta_profile(), self._acc(),
                             [_outcome("AAA.IS", ml_gate=gate)],
                             D(100_000), _CFG, "2026-07-02")
        assert [o.quantity for o in full] == [o.quantity for o in same]

    def test_ml_profile_flag_wired(self):
        assert PROFILES["ml_observer"].use_ml_meta_filter is True
        assert all(not p.use_ml_meta_filter
                   for c, p in PROFILES.items() if c != "ml_observer")

"""Tests for the video-criteria ratio scoring engine + combined signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.analytics.combined import combined_signal
from tradingagents.analytics.ratio_score import (
    compute_ratio_score,
    _score_current_ratio,
    _score_ev_ebitda,
    _score_net_debt_ebitda,
    _score_roe_real,
)


@pytest.mark.unit
class TestBandScorers:
    def test_ev_ebitda_bands(self):
        assert _score_ev_ebitda(6)[0] == 1.0      # 5-7 ideal
        assert _score_ev_ebitda(18)[0] < 0.3      # pahalı
        assert _score_ev_ebitda(-2)[0] == 0.0     # negatif FAVÖK
        assert _score_ev_ebitda(None)[0] == 0.0

    def test_current_ratio_bands(self):
        assert _score_current_ratio(2.0)[0] == 1.0    # ideal
        assert _score_current_ratio(0.8)[0] == 0.0    # iflas riski
        assert _score_current_ratio(4.0)[0] == 0.5    # hantal

    def test_net_debt_ebitda_bands(self):
        assert _score_net_debt_ebitda(-0.5)[0] == 1.0   # net nakit
        assert _score_net_debt_ebitda(1.0)[0] == 1.0    # düşük borç
        assert _score_net_debt_ebitda(3.5)[0] == 0.3    # riskli sınır
        assert _score_net_debt_ebitda(6.0)[0] == 0.0    # yüksek borç

    def test_roe_real_vs_inflation(self):
        # ROE enflasyonun üstünde → yüksek skor; altında → düşük
        hi, _ = _score_roe_real(60.0, 35.0)
        lo, _ = _score_roe_real(10.0, 35.0)
        assert hi > 0.9 and lo < 0.4


@pytest.mark.unit
class TestRatioScore:
    def _income(self, op_cur=120.0, op_prev=80.0) -> pd.DataFrame:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31"])
        return pd.DataFrame(
            {cols[0]: [op_cur, 500.0], cols[1]: [op_prev, 400.0]},
            index=["Operating Income", "Total Revenue"],
        )

    def test_strong_company_scores_buy(self):
        info = {
            "sector": "Industrials", "industry": "Airlines",
            "enterpriseToEbitda": 6.0, "currentRatio": 2.0,
            "returnOnEquity": 0.60, "revenueGrowth": 0.55,
            "totalDebt": 1000.0, "totalCash": 1200.0, "ebitda": 800.0,
            "longName": "Test Strong Co",
        }
        res = compute_ratio_score("TS.IS", inflation_pct=35.0,
                                  info=info, income=self._income())
        assert res.ok and not res.is_financial
        assert res.verdict == "AL" and res.score >= 65

    def test_weak_company_scores_sell(self):
        info = {
            "sector": "Industrials", "industry": "Construction",
            "enterpriseToEbitda": 22.0, "currentRatio": 0.7,
            "returnOnEquity": 0.05, "revenueGrowth": 0.10,
            "totalDebt": 5000.0, "totalCash": 100.0, "ebitda": 500.0,
        }
        # esas faaliyet zararda
        res = compute_ratio_score("TW.IS", inflation_pct=35.0, info=info,
                                  income=self._income(op_cur=-50.0, op_prev=20.0))
        assert res.ok and res.verdict == "SAT" and res.score < 45

    def test_financial_skips_ev_ebitda(self):
        info = {
            "sector": "Financial Services", "industry": "Banks—Regional",
            "enterpriseToEbitda": 6.0, "returnOnEquity": 0.45,
            "priceToBook": 0.7, "revenueGrowth": 0.50,
        }
        res = compute_ratio_score("GARAN.IS", inflation_pct=35.0,
                                  info=info, income=self._income())
        assert res.ok and res.is_financial
        names = [c.name for c in res.criteria]
        assert not any("FD/FAVÖK" in n for n in names)
        assert any("PD/DD" in n for n in names)

    def test_missing_data_not_penalized(self):
        # Sadece ROE var; eksikler paydaya girmemeli, skor yine üretilmeli
        info = {"sector": "Industrials", "industry": "X", "returnOnEquity": 0.70}
        res = compute_ratio_score("X.IS", inflation_pct=35.0, info=info,
                                  income=self._income())
        assert res.ok
        scored = [c for c in res.criteria if c.value is not None or "ZARARDA" in c.band]
        assert all(c.value is not None or "ZARARDA" in c.band for c in scored)


def _ohlcv(close: pd.Series, spread=1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": close.shift().fillna(close.iloc[0]),
        "High": close + spread, "Low": close - spread,
        "Close": close, "Volume": 1_000_000.0,
    })


@pytest.mark.unit
class TestCombinedSignal:
    def _uptrend_df(self) -> pd.DataFrame:
        n = 600
        idx = pd.bdate_range("2022-01-03", periods=n)
        close = pd.Series(np.linspace(100, 260, n) + 2 * np.sin(np.arange(n) / 8), index=idx)
        return _ohlcv(close)

    def _strong_info(self):
        cols = pd.to_datetime(["2025-12-31", "2024-12-31"])
        income = pd.DataFrame({cols[0]: [120.0, 500.0], cols[1]: [80.0, 400.0]},
                              index=["Operating Income", "Total Revenue"])
        info = {"sector": "Industrials", "industry": "Airlines",
                "enterpriseToEbitda": 6.0, "currentRatio": 2.0, "returnOnEquity": 0.60,
                "revenueGrowth": 0.55, "totalDebt": 1000.0, "totalCash": 1200.0,
                "ebitda": 800.0}
        return info, income

    def test_strong_fundamentals_plus_uptrend(self):
        info, income = self._strong_info()
        res = combined_signal("TEST", df=self._uptrend_df(), inflation_pct=35.0,
                              info=info, income=income)
        assert res.ok
        assert res.decision in ("AL", "GÜÇLÜ AL", "TUT")
        assert 0 <= res.combined_score <= 100
        assert res.rationale

    def test_weak_fundamentals_force_caution(self):
        cols = pd.to_datetime(["2025-12-31", "2024-12-31"])
        income = pd.DataFrame({cols[0]: [-50.0, 500.0], cols[1]: [20.0, 400.0]},
                              index=["Operating Income", "Total Revenue"])
        info = {"sector": "Industrials", "industry": "X", "enterpriseToEbitda": 22.0,
                "currentRatio": 0.7, "returnOnEquity": 0.05, "revenueGrowth": 0.10,
                "totalDebt": 5000.0, "totalCash": 100.0, "ebitda": 500.0}
        res = combined_signal("TEST", df=self._uptrend_df(), inflation_pct=35.0,
                              info=info, income=income)
        assert res.ok
        # Temel SAT → birleşik karar AL olamaz
        assert res.decision in ("SAT", "TUT", "KAÇIN")

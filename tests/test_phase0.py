"""Faz 0 — sinyal & veri sözleşmesi testleri.

F0.1  p_up yön düzeltmesi (confidence.py)
F0.2  Kalibrasyon kalitesi (probability.py) — quality_passed + BSS + brier ayrımı
F0.3  Makro şok entegrasyonu (run_hourly_analysis.py / scanner_page.py)
F0.4  Snapshot sürümleme (_meta alanı)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Ağır bağımlılıklar bu CI ortamında kurulu olmayabilir; modülleri sahte olarak ekle
# ki paket import zinciri kırılmasın (gerçek davranış test edilmiyor, sözleşme test ediliyor)
for _mod in ("yfinance", "stockstats", "ta", "requests", "bs4", "praw",
             "streamlit", "altair", "plotly", "plotly.express",
             "lightgbm", "xgboost", "hmmlearn", "hmmlearn.hmm",
             "dotenv", "httpx"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from tradingagents.analysis.confidence import unified_confidence


# ─────────────────────────────────────────────────────────────────────────────
# F0.1  p_up yön sözleşmesi
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPUpYon:
    """Model kararla uyumlu ise güven artar; karşı ise düşer; nötürde etki sıfırdır."""

    def _score(self, decision_raw: str, p_up: float) -> float:
        res = unified_confidence(
            agreement_level="nötr",
            decision_raw=decision_raw,
            p_up=p_up,
        )
        return res.score

    def test_al_yuksek_pup_güven_artirir(self):
        """AL kararı + yüksek p_up → referans (p_up=0.5) skorundan yüksek olmalı."""
        base = self._score("AL", 0.5)
        high = self._score("AL", 0.9)
        assert high > base, "AL + p_up=0.90 güveni artırmalı"

    def test_al_dusuk_pup_güven_azaltir(self):
        """AL kararı + düşük p_up (model aşağı diyor) → referanstan düşük olmalı."""
        base = self._score("AL", 0.5)
        low = self._score("AL", 0.1)
        assert low < base, "AL + p_up=0.10 güveni düşürmeli"

    def test_sat_dusuk_pup_güven_artirir(self):
        """SAT kararı + düşük p_up (model aşağı diyor, SAT ile uyumlu) → yüksek skor."""
        base = self._score("SAT", 0.5)
        high = self._score("SAT", 0.1)
        assert high > base, "SAT + p_up=0.10 güveni artırmalı"

    def test_sat_yuksek_pup_güven_azaltir(self):
        """SAT kararı + yüksek p_up (model yukarı diyor, SAT'a karşı) → düşük skor."""
        base = self._score("SAT", 0.5)
        low = self._score("SAT", 0.9)
        assert low < base, "SAT + p_up=0.90 güveni düşürmeli"

    def test_notr_karar_pup_etkisiz(self):
        """Nötr karar (İZLE/TUT) → p_up katkısı sıfır olmalı, skor değişmemeli."""
        score_05 = self._score("İZLE", 0.5)
        score_09 = self._score("İZLE", 0.9)
        score_01 = self._score("İZLE", 0.1)
        assert score_05 == score_09 == score_01, "Nötr kararla p_up skoru değiştirmemeli"

    def test_pup_yok_etki_yok(self):
        """p_up=None → skor, p_up dahil edilmeden hesaplanmalı."""
        no_pup = unified_confidence(agreement_level="güçlü", decision_raw="AL", p_up=None)
        with_05 = unified_confidence(agreement_level="güçlü", decision_raw="AL", p_up=0.5)
        # p_up=0.5 → model_edge=0 → katkı=0 → aynı skor
        assert abs(no_pup.score - with_05.score) < 0.01

    def test_al_pup_katkisi_asimetrik(self):
        """AL kararında yüksek p_up katkısı, düşük p_up cezasına eşit büyüklükte."""
        base = self._score("AL", 0.5)
        up_diff = self._score("AL", 0.9) - base
        down_diff = base - self._score("AL", 0.1)
        assert abs(up_diff - down_diff) < 0.01, "Katkı simetrik olmalı (|+12| == |-12|)"


# ─────────────────────────────────────────────────────────────────────────────
# F0.2  ProbabilityResult kalite sözleşmesi
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestProbabilityResultSozlesmesi:
    """ProbabilityResult'ın yeni alan sözleşmesini doğrular (hesaplama değil)."""

    def test_basarisiz_sonuc_quality_false(self):
        from tradingagents.analytics.probability import ProbabilityResult
        r = ProbabilityResult(ok=False, reason="test", rejection_reasons=["yetersiz_veri"])
        assert r.quality_passed is False
        assert len(r.rejection_reasons) > 0

    def test_basarili_sonuc_alanlari_mevcut(self):
        from tradingagents.analytics.probability import ProbabilityResult
        r = ProbabilityResult(
            ok=True, ticker="GARAN", p_up=0.62, p_up_raw=0.58,
            brier_raw=0.22, brier_calibrated=0.20, brier_skill_score=0.05,
            auc=0.55, n_samples=300, n_calibration=70, n_test=30,
            quality_passed=True,
        )
        assert r.p_up == 0.62
        assert r.brier_raw is not None
        assert r.brier_calibrated is not None
        assert r.brier_skill_score is not None
        assert r.n_calibration > 0 and r.n_test > 0

    def test_brier_property_geriye_uyumlu(self):
        """Eski .brier erişimi brier_raw'a yönlendirmeli."""
        from tradingagents.analytics.probability import ProbabilityResult
        r = ProbabilityResult(ok=True, brier_raw=0.21)
        assert r.brier == 0.21

    def test_quality_passed_none_pup(self):
        """quality_passed=False olduğunda p_up=None olmalı (kalitesiz tahmin geçmez)."""
        from tradingagents.analytics.probability import ProbabilityResult
        r = ProbabilityResult(ok=True, quality_passed=False, p_up=None)
        assert r.p_up is None

    def test_bss_pozitif_ise_naive_den_iyi(self):
        """BSS > 0 → model naive sınıf oranından daha iyi tahmin ediyor demek."""
        from tradingagents.analytics.probability import _brier_skill_score
        import numpy as np
        y = np.array([1, 0, 1, 0, 1, 1, 0, 1])
        brier_naive_approx = float(np.mean(y)) * (1 - float(np.mean(y)))
        # model naive'den iyi → brier_model < brier_naive → BSS > 0
        bss = _brier_skill_score(brier_naive_approx * 0.8, y)
        assert bss > 0.0

    def test_bss_negatif_ise_naive_den_kotu(self):
        """BSS < 0 → model naive'den kötü."""
        from tradingagents.analytics.probability import _brier_skill_score
        import numpy as np
        y = np.array([1, 0, 1, 0, 1, 1, 0, 1])
        brier_naive_approx = float(np.mean(y)) * (1 - float(np.mean(y)))
        bss = _brier_skill_score(brier_naive_approx * 1.5, y)
        assert bss < 0.0

    def test_yeni_kanit_alanlari_mevcut(self):
        """ModelEvidence sözleşmesi: recent_skill, model_version, prediction_asof,
        trained_until, n_train alanları bulunmalı."""
        from tradingagents.analytics.probability import ProbabilityResult, MODEL_VERSION
        r = ProbabilityResult(
            ok=True, ticker="GARAN", p_up=0.6, recent_skill=0.04,
            n_train=250, prediction_asof="2026-06-29T00:00:00+00:00",
            trained_until="2026-06-27", quality_passed=True,
        )
        assert r.recent_skill == 0.04
        assert r.n_train == 250
        assert r.prediction_asof.startswith("2026")
        assert r.trained_until == "2026-06-27"
        # model_version varsayılanı MODEL_VERSION'a eşit olmalı
        assert ProbabilityResult(ok=True).model_version == MODEL_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# F0.1b  validate_model_evidence — kalite kararı tek saf fonksiyonda
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestValidateModelEvidence:
    """Kalite kapısı ayrı saf fonksiyonda; çıplak boolean yok, neden listesi var."""

    def test_hepsi_gecince_quality_true(self):
        from tradingagents.analytics.probability import validate_model_evidence
        ok, reasons = validate_model_evidence(auc=0.58, n_test=30, brier_skill_score=0.05)
        assert ok is True
        assert reasons == []

    def test_dusuk_auc_reddedilir(self):
        from tradingagents.analytics.probability import validate_model_evidence
        ok, reasons = validate_model_evidence(auc=0.50, n_test=30, brier_skill_score=0.05)
        assert ok is False
        assert any("auc" in r for r in reasons)

    def test_kucuk_test_penceresi_reddedilir(self):
        from tradingagents.analytics.probability import validate_model_evidence
        ok, reasons = validate_model_evidence(auc=0.58, n_test=10, brier_skill_score=0.05)
        assert ok is False
        assert any("test_penceresi" in r for r in reasons)

    def test_bss_olculemedi_reddedilir(self):
        from tradingagents.analytics.probability import validate_model_evidence
        ok, reasons = validate_model_evidence(auc=0.58, n_test=30, brier_skill_score=None)
        assert ok is False
        assert any("bss_olculemedi" in r for r in reasons)

    def test_negatif_bss_reddedilir(self):
        from tradingagents.analytics.probability import validate_model_evidence
        ok, reasons = validate_model_evidence(auc=0.58, n_test=30, brier_skill_score=-0.02)
        assert ok is False
        assert any("bss_negatif" in r for r in reasons)

    def test_birden_cok_red_nedeni_birikir(self):
        """Birden çok şart düşerse hepsi rejection_reasons'ta toplanmalı."""
        from tradingagents.analytics.probability import validate_model_evidence
        ok, reasons = validate_model_evidence(auc=0.40, n_test=5, brier_skill_score=None)
        assert ok is False
        assert len(reasons) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# F0.3  Makro şok entegrasyonu
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMakroSokEntegrasyon:
    """macro_shock=True → AL sinyali gelen hissede karar 'İZLE' olmalı."""

    def test_makro_sok_al_sinyalini_durdurur(self):
        res = unified_confidence(
            agreement_level="güçlü",
            decision_raw="AL",
            macro_shock=True,
        )
        assert res.decision == "İZLE", "Makro şokta AL sinyali İZLE'ye dönmeli"
        assert res.gate_passed is False

    def test_makro_sok_sat_sinyalini_durdurmaz(self):
        res = unified_confidence(
            agreement_level="güçlü",
            decision_raw="SAT",
            macro_shock=True,
        )
        assert res.decision == "SAT", "Makro şok yalnız AL'ı durdurur, SAT'ı değil"

    def test_makro_sok_yok_al_gecer(self):
        res = unified_confidence(
            agreement_level="güçlü",
            decision_raw="AL",
            macro_shock=False,
        )
        assert res.decision in ("AL", "GÜÇLÜ AL"), "Makro şok yokken güçlü AL geçmeli"

    def test_analyze_ticker_macro_shock_parametresi(self):
        """analyze_ticker macro_shock parametresini kabul etmeli (imza testi)."""
        import inspect
        from tradingagents.analysis.run import analyze_ticker
        sig = inspect.signature(analyze_ticker)
        assert "macro_shock" in sig.parameters

    def test_analyze_universe_macro_shock_otomatik(self):
        """analyze_universe macro_shock=None → macro_shock_state() çağırmalı (mock)."""
        from unittest.mock import patch, MagicMock
        from tradingagents.analysis.run import analyze_universe
        with patch("tradingagents.analysis.run.macro_shock_state", return_value=True) as mock_ms, \
             patch("tradingagents.analysis.run.analyze_ticker") as mock_at:
            mock_at.return_value = MagicMock(ok=True)
            analyze_universe(["GARAN"], macro_shock=None)
            mock_ms.assert_called_once()
            _, kwargs = mock_at.call_args
            assert kwargs.get("macro_shock") is True


# ─────────────────────────────────────────────────────────────────────────────
# F0.4  Snapshot sürümleme
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSnapshotSurumleme:
    """to_snapshot_row → signals._meta alanında strateji versiyonu bulunmalı."""

    def _make_outcome(self, model_meta=None):
        from tradingagents.analysis.run import AnalysisOutcome
        return AnalysisOutcome(
            ticker="GARAN", ok=True, status="AL", decision="AL",
            health={"last_bar": "2026-06-27"},
            model_meta=model_meta or {},
        )

    def test_meta_alani_mevcut(self):
        from tradingagents.analysis.run import to_snapshot_row
        row = to_snapshot_row(self._make_outcome())
        assert "_meta" in row["signals"]

    def test_strateji_versiyonu_kayitli(self):
        from tradingagents.analysis.run import to_snapshot_row, STRATEGY_VERSION
        row = to_snapshot_row(self._make_outcome())
        assert row["signals"]["_meta"]["strategy_version"] == STRATEGY_VERSION

    def test_signal_asof_mevcut(self):
        from tradingagents.analysis.run import to_snapshot_row
        row = to_snapshot_row(self._make_outcome())
        assert row["signals"]["_meta"]["signal_asof"] is not None

    def test_last_bar_health_ten_geliyor(self):
        from tradingagents.analysis.run import to_snapshot_row
        row = to_snapshot_row(self._make_outcome())
        assert row["signals"]["_meta"]["last_bar"] == "2026-06-27"

    def test_universe_version_kayitli(self):
        from tradingagents.analysis.run import to_snapshot_row, UNIVERSE_VERSION
        row = to_snapshot_row(self._make_outcome())
        assert row["signals"]["_meta"]["universe_version"] == UNIVERSE_VERSION

    def test_signal_session_mevcut(self):
        from tradingagents.analysis.run import to_snapshot_row
        row = to_snapshot_row(self._make_outcome())
        assert row["signals"]["_meta"]["signal_session"] is not None

    def test_model_kaniti_meta_ya_akar(self):
        """model_meta verilince model_version + red nedenleri snapshot _meta'ya yazılır."""
        from tradingagents.analysis.run import to_snapshot_row
        mm = {"model_version": "calib-v1-sigmoid-bss", "quality_passed": False,
              "rejection_reasons": ["bss_negatif (-0.020)"]}
        row = to_snapshot_row(self._make_outcome(model_meta=mm))
        meta = row["signals"]["_meta"]
        assert meta["model_version"] == "calib-v1-sigmoid-bss"
        assert meta["model_quality_passed"] is False
        assert "bss_negatif (-0.020)" in meta["model_rejection_reasons"]

    def test_model_meta_yoksa_alanlar_none(self):
        """model_meta verilmezse model alanları None (snapshot yine de tutarlı)."""
        from tradingagents.analysis.run import to_snapshot_row
        meta = to_snapshot_row(self._make_outcome())["signals"]["_meta"]
        assert meta["model_version"] is None
        assert meta["model_quality_passed"] is None


# ─────────────────────────────────────────────────────────────────────────────
# F0.3b  analyze_ticker model_meta'yı snapshot'a taşır (orchestrator sözleşmesi)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestModelMetaOrchestrator:
    """analyze_ticker model_meta parametresini kabul edip outcome'a taşımalı."""

    def test_analyze_ticker_model_meta_parametresi(self):
        import inspect
        from tradingagents.analysis.run import analyze_ticker
        sig = inspect.signature(analyze_ticker)
        assert "model_meta" in sig.parameters

    def test_analyze_universe_model_cache_satirini_gecirir(self):
        """analyze_universe model_cache satırından p_up + model_meta türetip geçirmeli."""
        from unittest.mock import patch, MagicMock
        from tradingagents.analysis.run import analyze_universe
        cache = {"GARAN": {"p_up": 0.61, "model_version": "calib-v1-sigmoid-bss",
                           "quality_passed": True, "rejection_reasons": []}}
        with patch("tradingagents.analysis.run.macro_shock_state", return_value=False), \
             patch("tradingagents.analysis.run.index_regime", return_value=None), \
             patch("tradingagents.analysis.run.analyze_ticker") as mock_at:
            mock_at.return_value = MagicMock(ok=True)
            analyze_universe(["GARAN"], model_cache=cache)
            _, kwargs = mock_at.call_args
            assert kwargs.get("p_up") == 0.61
            assert kwargs.get("model_meta", {}).get("model_version") == "calib-v1-sigmoid-bss"

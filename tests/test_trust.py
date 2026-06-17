"""Güven katmanı — mutabakat, sinyal istikrarı, sinyal karnesi (saf testler)."""

from __future__ import annotations

import pytest

from tradingagents.analysis import trust
from tradingagents.analysis.run import AnalysisOutcome, to_snapshot_row


@pytest.mark.unit
class TestAgreement:
    def test_all_same_direction_is_strong(self):
        level, _ = trust.agreement({"teknik": 1, "rasyo": 1, "dip": 1})
        assert level == "güçlü"

    def test_opposing_directions_is_conflict(self):
        level, summary = trust.agreement({"teknik": 1, "rasyo": -1, "dip": 0})
        assert level == "çelişki"
        assert "çeliş" in summary.lower()

    def test_some_flat_no_conflict_is_partial(self):
        level, _ = trust.agreement({"teknik": 1, "rasyo": 0, "dip": 0})
        assert level == "kısmi"

    def test_all_flat_is_neutral(self):
        level, _ = trust.agreement({"teknik": 0, "rasyo": 0})
        assert level == "nötr"


@pytest.mark.unit
class TestSignalStability:
    def test_too_few_observations(self):
        level, score, flips = trust.signal_stability(["AL", "AL"])
        assert level == "yetersiz"

    def test_constant_is_stable(self):
        level, score, flips = trust.signal_stability(["AL", "AL", "AL", "AL"])
        assert flips == 0
        assert score == 1.0
        assert level == "istikrarlı"

    def test_alternating_is_unstable(self):
        level, score, flips = trust.signal_stability(["AL", "SAT", "AL", "SAT"])
        assert flips == 3
        assert score == 0.0
        assert level == "kararsız"

    def test_one_flip_is_medium(self):
        level, score, flips = trust.signal_stability(["AL", "AL", "SAT", "SAT"])
        assert flips == 1
        assert score == pytest.approx(0.67, abs=0.01)
        assert level == "orta"


@pytest.mark.unit
class TestTrackRecord:
    def _rows(self):
        return [
            {"ts": "2026-01-01T10:00:00+00:00", "status": "AL", "close": 100},
            {"ts": "2026-01-02T10:00:00+00:00", "status": "NÖTR", "close": 110},
            {"ts": "2026-01-02T11:00:00+00:00", "status": "SAT", "close": 110},
            {"ts": "2026-01-03T11:00:00+00:00", "status": "NÖTR", "close": 104},
        ]

    def test_forward_return_and_hit_rate(self):
        rec = trust.track_record(self._rows(), horizon_days=1, min_samples=1)
        assert rec["ready"] is True
        assert rec["AL"]["n"] == 1
        assert rec["AL"]["hit_rate"] == 100.0       # +10% sonrası
        assert rec["AL"]["mean_fwd"] == pytest.approx(10.0, abs=0.01)
        assert rec["SAT"]["n"] == 1
        assert rec["SAT"]["hit_rate"] == 100.0       # SAT sonrası düşüş → isabet
        assert rec["SAT"]["mean_fwd"] == pytest.approx(-5.45, abs=0.05)

    def test_not_ready_without_enough_samples(self):
        rec = trust.track_record(self._rows(), horizon_days=1, min_samples=5)
        assert rec["ready"] is False

    def test_signal_without_forward_data_is_skipped(self):
        rows = [{"ts": "2026-01-01T10:00:00+00:00", "status": "AL", "close": 100}]
        rec = trust.track_record(rows, horizon_days=1, min_samples=1)
        assert rec["AL"]["n"] == 0


@pytest.mark.unit
class TestSnapshotRow:
    def test_outcome_maps_to_row(self):
        o = AnalysisOutcome(
            ticker="THYAO.IS", ok=True, status="AL", decision="GÜÇLÜ AL",
            combined_score=72.0, tech_score=44.0, ratio_score=80.0,
            ratio_verdict="AL", confidence="yüksek",
            agreement_level="güçlü", agreement="Tüm yöntemler aynı yönde.",
            close=130.5, smi=12.3, signals={"dip_status": "AL BÖLGESİ"},
            health={"price": "ok"},
        )
        row = to_snapshot_row(o, scope="portfolio", source="on_add")
        assert row["ticker"] == "THYAO.IS"
        assert row["scope"] == "portfolio"
        assert row["source"] == "on_add"
        assert row["status"] == "AL"
        assert row["decision"] == "GÜÇLÜ AL"
        assert row["combined_score"] == 72.0
        assert row["agreement"].startswith("güçlü:")
        assert row["close"] == 130.5
        assert row["signals"] == {"dip_status": "AL BÖLGESİ"}
        assert isinstance(row["ts"], str) and row["ts"].endswith("+00:00")

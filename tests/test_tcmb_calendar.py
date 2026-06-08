"""Tests for the TCMB PPK decision calendar (seed table + EVDS derivation)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from tradingagents.dataflows import tcmb_calendar as cal


@pytest.fixture(autouse=True)
def _clear_cache():
    cal.clear_cache()
    yield
    cal.clear_cache()


@pytest.mark.unit
class TestSeedAndFilter:
    def test_seed_only_when_no_key(self):
        # api_key=None -> EVDS branch skipped entirely, deterministic seed result
        cuts = cal.get_ppk_decisions(direction="cut", api_key=None)
        hikes = cal.get_ppk_decisions(direction="hike", api_key=None)
        assert cuts and all(d["direction"] == "cut" for d in cuts)
        assert hikes and all(d["direction"] == "hike" for d in hikes)
        # sorted ascending by date
        assert [d["date"] for d in cuts] == sorted(d["date"] for d in cuts)

    def test_direction_none_returns_all(self):
        alld = cal.get_ppk_decisions(direction=None, api_key=None)
        assert len(alld) == len(cal.PPK_DECISIONS_SEED)


@pytest.mark.unit
class TestParseAndChangePoints:
    def test_parse_evds_date_formats(self):
        assert cal._parse_evds_date("21-03-2024") == date(2024, 3, 21)
        assert cal._parse_evds_date("03-2024") == date(2024, 3, 1)
        assert cal._parse_evds_date("2024-03-21") == date(2024, 3, 21)
        assert cal._parse_evds_date("garbage") is None

    def test_decisions_from_rate_series_detects_changes(self):
        obs = [
            (date(2024, 1, 1), 45.0),
            (date(2024, 2, 1), 45.0),   # hold -> no decision
            (date(2024, 3, 21), 50.0),  # hike
            (date(2024, 12, 26), 47.5), # cut
        ]
        out = cal._decisions_from_rate_series(obs)
        assert [(d["date"], d["direction"]) for d in out] == [
            ("2024-03-21", "hike"),
            ("2024-12-26", "cut"),
        ]


@pytest.mark.unit
class TestEvdsMerge:
    def test_evds_augments_seed_and_caches(self):
        fake_obs = [
            (date(2025, 3, 6), 45.0),
            (date(2025, 4, 17), 42.5),  # a cut not present in the seed
        ]
        with patch.object(cal, "_fetch_evds_observations", return_value=fake_obs) as m:
            first = cal.get_ppk_decisions(direction="cut", api_key="KEY")
            second = cal.get_ppk_decisions(direction="cut", api_key="KEY")
        dates = {d["date"] for d in first}
        assert "2025-04-17" in dates          # EVDS-derived cut surfaced
        assert m.call_count == 1              # cached: one EVDS call for two queries
        assert first == second

    def test_evds_failure_falls_back_to_seed(self):
        with patch.object(cal, "_fetch_evds_observations", return_value=[]):
            out = cal.get_ppk_decisions(direction="hike", api_key="KEY")
        assert out and all(d["direction"] == "hike" for d in out)

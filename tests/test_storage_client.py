"""Supabase REST istemcisi — yapılandırma/URL kurulumu (ağsız)."""

from __future__ import annotations

import pytest

from tradingagents.storage import supabase_client as sc


@pytest.mark.unit
class TestCredentials:
    def test_missing_config_raises(self, monkeypatch):
        monkeypatch.setattr(sc, "_secret", lambda key: None)
        with pytest.raises(sc.SupabaseError):
            sc.get_credentials()
        assert sc.is_configured() is False

    def test_service_key_preferred_and_url_normalized(self, monkeypatch):
        secrets = {
            "SUPABASE_URL": "https://abc.supabase.co/",
            "SUPABASE_SERVICE_KEY": "service-key",
            "SUPABASE_ANON_KEY": "anon-key",
        }
        monkeypatch.setattr(sc, "_secret", lambda key: secrets.get(key))
        url, key = sc.get_credentials()
        assert url == "https://abc.supabase.co"   # sondaki / kırpılır
        assert key == "service-key"               # service > anon
        assert sc.is_configured() is True

    def test_rest_builds_base_and_headers(self, monkeypatch):
        client = sc.SupabaseREST(url="https://abc.supabase.co", key="k")
        assert client._base == "https://abc.supabase.co/rest/v1"
        headers = client._headers({"Prefer": "return=representation"})
        assert headers["apikey"] == "k"
        assert headers["Authorization"] == "Bearer k"
        assert headers["Prefer"] == "return=representation"

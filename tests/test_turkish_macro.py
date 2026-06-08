"""Tests for the Turkish macro-news fetcher used by the Macro analyst (BIST)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.dataflows import turkish_macro


def _item(title: str, summary: str = "", source: str = "S", pub_date: str = "d") -> dict:
    return {"title": title, "summary": summary, "pub_date": pub_date, "source": source}


@pytest.mark.unit
class TestClassifyTheme:
    def test_rate_policy_wins_priority(self):
        # casefolded text; rate keyword should claim it as the policy theme
        assert turkish_macro._classify_theme("tcmb faiz kararını açıkladı") == "Faiz / Para Politikası"

    def test_inflation(self):
        assert turkish_macro._classify_theme("yıllık enflasyon geriledi, tüfe açıklandı") == "Enflasyon"

    def test_fx(self):
        assert turkish_macro._classify_theme("dolar/tl yeni zirvede") == "Kur / Döviz"

    def test_non_macro_returns_none(self):
        assert turkish_macro._classify_theme("şirket yeni fabrika açılışı yaptı") is None


@pytest.mark.unit
class TestFetchTurkishMacroNews:
    def test_buckets_macro_and_drops_noise(self):
        items = [
            _item("TCMB faiz kararını açıkladı", "politika faizi sabit"),
            _item("Yıllık enflasyon geriledi", "tüfe verisi"),
            _item("THYAO yeni uçak siparişi verdi", "filo genişlemesi"),  # noise, dropped
        ]
        with patch.object(turkish_macro, "_fetch_feed", return_value=items):
            out = turkish_macro.fetch_turkish_macro_news(
                feeds=(("S", "http://x"),), inter_request_delay=0,
            )
        assert "TÜRKİYE MAKRO GÖRÜNÜMÜ" in out
        assert "Faiz / Para Politikası" in out and "TCMB faiz kararını açıkladı" in out
        assert "Enflasyon" in out and "Yıllık enflasyon geriledi" in out
        assert "THYAO" not in out  # company noise filtered out

    def test_dedupes_across_feeds(self):
        dup = [_item("TCMB faizi sabit tuttu")]
        with patch.object(turkish_macro, "_fetch_feed", return_value=dup):
            out = turkish_macro.fetch_turkish_macro_news(
                feeds=(("A", "http://a"), ("B", "http://b")), inter_request_delay=0,
            )
        assert out.count("TCMB faizi sabit tuttu") == 1

    def test_all_feeds_down_returns_placeholder(self):
        with patch.object(turkish_macro, "_fetch_feed", return_value=[]):
            out = turkish_macro.fetch_turkish_macro_news(
                feeds=(("S", "http://x"),), inter_request_delay=0,
            )
        assert "ulaşılamadı" in out

    def test_no_macro_items_returns_distinct_placeholder(self):
        only_noise = [_item("Şirket bilanço açıkladı", "net kâr arttı")]
        with patch.object(turkish_macro, "_fetch_feed", return_value=only_noise):
            out = turkish_macro.fetch_turkish_macro_news(
                feeds=(("S", "http://x"),), inter_request_delay=0,
            )
        assert "tespit edilemedi" in out

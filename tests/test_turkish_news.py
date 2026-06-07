"""Tests for the Turkish-language market news fetcher used by BIST (.IS) tickers."""

from __future__ import annotations

from unittest.mock import patch
from urllib.error import URLError

import pytest

from tradingagents.dataflows import turkish_news
from tradingagents.dataflows.symbol_utils import is_bist_ticker


_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>THYAO yeni uçak siparişi verdi</title>
      <description>&lt;p&gt;Türk Hava Yolları filo &lt;b&gt;genişlemesi&lt;/b&gt; açıkladı.&lt;/p&gt;</description>
      <pubDate>Sun, 07 Jun 2026 08:00:00 +0300</pubDate>
    </item>
    <item>
      <title>TCMB faiz kararını açıkladı</title>
      <description>Merkez Bankası politika faizini sabit tuttu.</description>
      <pubDate>Sun, 07 Jun 2026 07:00:00 +0300</pubDate>
    </item>
    <item>
      <title></title>
      <description>başlıksız öğe atlanmalı</description>
    </item>
  </channel>
</rss>
"""


@pytest.mark.unit
class TestIsBistTicker:
    def test_recognizes_is_suffix(self):
        assert is_bist_ticker("THYAO.IS")
        assert is_bist_ticker("garan.is")  # case-insensitive
        assert is_bist_ticker("  ASELS.IS  ")  # whitespace tolerant

    def test_rejects_non_bist(self):
        assert not is_bist_ticker("AAPL")
        assert not is_bist_ticker("7203.T")
        assert not is_bist_ticker("BTC-USD")
        assert not is_bist_ticker("")
        assert not is_bist_ticker(None)  # type: ignore[arg-type]


@pytest.mark.unit
class TestBuildKeywords:
    def test_includes_bist_code_without_suffix(self):
        kws = turkish_news._build_keywords("THYAO.IS", None)
        assert "thyao" in kws

    def test_adds_company_tokens_and_drops_generic(self):
        kws = turkish_news._build_keywords(
            "THYAO.IS", "Türk Hava Yolları Anonim Ortaklığı"
        )
        assert "thyao" in kws
        assert "yolları" in kws          # length >= 5, kept
        assert "anonim" not in kws       # generic corporate-form token dropped
        assert "ortaklığı" not in kws    # generic, dropped
        assert "hava" not in kws         # length < 5, dropped

    def test_dedupes(self):
        kws = turkish_news._build_keywords("ASELS.IS", "ASELS ASELS")
        assert kws.count("asels") == 1


@pytest.mark.unit
class TestStripHtml:
    def test_strips_tags_and_unescapes(self):
        raw = "<p>Türk Hava Yolları <b>filo</b> &amp; büyüme</p>"
        assert turkish_news._strip_html(raw) == "Türk Hava Yolları filo & büyüme"

    def test_empty(self):
        assert turkish_news._strip_html("") == ""


@pytest.mark.unit
class TestFetchFeed:
    def _patch_response(self, xml_bytes):
        class _Resp:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False
            def read(self_inner):
                return xml_bytes
        return patch.object(turkish_news, "urlopen", return_value=_Resp())

    def test_parses_rss_items_and_skips_titleless(self):
        with self._patch_response(_SAMPLE_RSS.encode("utf-8")):
            items = turkish_news._fetch_feed("Test", "http://x", limit=10, timeout=5.0)
        assert len(items) == 2  # titleless item skipped
        assert items[0]["title"] == "THYAO yeni uçak siparişi verdi"
        assert items[0]["source"] == "Test"
        assert "genişlemesi" in items[0]["summary"]  # html stripped
        assert items[0]["pub_date"].startswith("Sun, 07 Jun 2026")

    def test_network_failure_fails_open(self):
        with patch.object(turkish_news, "urlopen", side_effect=URLError("boom")):
            assert turkish_news._fetch_feed("Test", "http://x", 10, 5.0) == []

    def test_malformed_xml_fails_open(self):
        with self._patch_response(b"<<not xml>>"):
            assert turkish_news._fetch_feed("Test", "http://x", 10, 5.0) == []


@pytest.mark.unit
class TestFetchTurkishMarketNews:
    def test_partitions_company_vs_market(self):
        items = [
            {"title": "THYAO yeni uçak siparişi", "summary": "filo", "pub_date": "d1", "source": "S"},
            {"title": "TCMB faiz kararı", "summary": "makro", "pub_date": "d2", "source": "S"},
        ]
        with patch.object(turkish_news, "_fetch_feed", return_value=items):
            out = turkish_news.fetch_turkish_market_news(
                "THYAO.IS", "Türk Hava Yolları", feeds=(("S", "http://x"),),
                inter_request_delay=0,
            )
        assert "Şirkete özel haberler (THYAO)" in out
        assert "THYAO yeni uçak siparişi" in out
        assert "Genel piyasa" in out
        assert "TCMB faiz kararı" in out

    def test_no_company_match_falls_back_to_context(self):
        items = [
            {"title": "TCMB faiz kararı", "summary": "makro", "pub_date": "d", "source": "S"},
        ]
        with patch.object(turkish_news, "_fetch_feed", return_value=items):
            out = turkish_news.fetch_turkish_market_news(
                "ASELS.IS", None, feeds=(("S", "http://x"),), inter_request_delay=0,
            )
        assert "doğrudan şirket haberi bulunamadı" in out
        assert "TCMB faiz kararı" in out

    def test_dedupes_across_feeds(self):
        dup = [{"title": "Aynı başlık", "summary": "", "pub_date": "", "source": "A"}]
        with patch.object(turkish_news, "_fetch_feed", return_value=dup):
            out = turkish_news.fetch_turkish_market_news(
                "XYZ.IS", None,
                feeds=(("A", "http://a"), ("B", "http://b")),
                inter_request_delay=0,
            )
        assert out.count("Aynı başlık") == 1

    def test_all_feeds_down_returns_placeholder(self):
        with patch.object(turkish_news, "_fetch_feed", return_value=[]):
            out = turkish_news.fetch_turkish_market_news(
                "THYAO.IS", None, feeds=(("S", "http://x"),), inter_request_delay=0,
            )
        assert "ulaşılamadı" in out

"""Tests for the Investing.com ticker-specific Turkish news fetcher (BIST)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.dataflows import investing_news


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


@pytest.mark.unit
class TestBistBase:
    def test_strips_is_suffix_and_uppercases(self):
        assert investing_news._bist_base("THYAO.IS") == "THYAO"
        assert investing_news._bist_base("garan.is") == "GARAN"
        assert investing_news._bist_base("  asels.is ") == "ASELS"


@pytest.mark.unit
class TestResolveEquityUrl:
    def _quotes(self, *quotes):
        return _FakeResp({"quotes": list(quotes)})

    def test_prefers_istanbul_symbol_match(self):
        resp = self._quotes(
            {"symbol": "THYAO", "exchange": "NASDAQ", "flag": "USA", "url": "/equities/other"},
            {"symbol": "THYAO", "exchange": "Istanbul", "flag": "Turkey", "url": "/equities/turk-hava-yollari"},
        )
        with patch("curl_cffi.requests.get", return_value=resp):
            assert investing_news._resolve_equity_url("THYAO.IS", 5) == "/equities/turk-hava-yollari"

    def test_falls_back_to_turkey_flag(self):
        resp = self._quotes(
            {"symbol": "XYZ", "exchange": "London", "flag": "UK", "url": "/equities/foreign"},
            {"symbol": "OTHER", "exchange": "Istanbul", "flag": "Turkey", "url": "/equities/tr-co"},
        )
        with patch("curl_cffi.requests.get", return_value=resp):
            assert investing_news._resolve_equity_url("XYZ.IS", 5) == "/equities/tr-co"

    def test_no_equity_quote_returns_none(self):
        resp = self._quotes({"symbol": "XYZ", "url": "/crypto/whatever"})
        with patch("curl_cffi.requests.get", return_value=resp), \
             patch.object(investing_news.time, "sleep"):
            assert investing_news._resolve_equity_url("XYZ.IS", 5) is None

    def test_retries_once_on_empty_then_succeeds(self):
        empty = _FakeResp({"quotes": []})
        good = self._quotes({"symbol": "GARAN", "exchange": "Istanbul", "flag": "Turkey",
                             "url": "/equities/garanti-bankasi"})
        with patch("curl_cffi.requests.get", side_effect=[empty, good]) as g, \
             patch.object(investing_news.time, "sleep"):
            assert investing_news._resolve_equity_url("GARAN.IS", 5) == "/equities/garanti-bankasi"
        assert g.call_count == 2

    def test_network_error_fails_open(self):
        with patch("curl_cffi.requests.get", side_effect=RuntimeError("boom")), \
             patch.object(investing_news.time, "sleep"):
            assert investing_news._resolve_equity_url("THYAO.IS", 5) is None


@pytest.mark.unit
class TestFetchInvestingNews:
    def test_unresolved_ticker_returns_placeholder(self):
        with patch.object(investing_news, "_resolve_equity_url", return_value=None):
            out = investing_news.fetch_investing_news("ZZZZ.IS")
        assert "eşleşen BIST hissesi bulunamadı" in out

    def test_render_failure_returns_placeholder(self):
        with patch.object(investing_news, "_resolve_equity_url", return_value="/equities/x"), \
             patch.object(investing_news, "_run_in_thread", return_value=[]):
            out = investing_news.fetch_investing_news("THYAO.IS")
        assert "render edilemedi" in out

    def test_formats_titles_with_base_header_and_links(self):
        items = [
            {"title": "THY 2025 Kâr Dağıtımını Erteledi", "url": "/news/global-filings/thy-kar"},
            {"title": "THY CEO değişikliği duyurdu", "url": "https://tr.investing.com/news/x"},
        ]
        with patch.object(investing_news, "_resolve_equity_url", return_value="/equities/turk-hava-yollari"), \
             patch.object(investing_news, "_run_in_thread", return_value=items):
            out = investing_news.fetch_investing_news("THYAO.IS")
        assert "THYAO — Investing.com Türkçe haber" in out
        assert "THY 2025 Kâr Dağıtımını Erteledi" in out
        assert "https://tr.investing.com/news/global-filings/thy-kar" in out  # relative -> absolute
        assert "https://tr.investing.com/news/x" in out  # already absolute, kept


@pytest.mark.unit
class TestRunInThread:
    def test_returns_callable_result(self):
        assert investing_news._run_in_thread(lambda x: x * 2, 21, timeout=5) == 42

    def test_swallows_worker_exception(self):
        def boom():
            raise ValueError("nope")
        assert investing_news._run_in_thread(boom, timeout=5) == []

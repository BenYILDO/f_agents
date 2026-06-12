"""Tests for the Turkish political shock calendar + political news filter."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.political_events import (
    POLITICAL_SHOCKS,
    fetch_political_news,
    format_shock_calendar,
    political_event_dates,
)


@pytest.mark.unit
class TestShockCalendar:
    def test_calendar_is_well_formed(self):
        assert len(POLITICAL_SHOCKS) >= 15
        valid_cats = {"secim", "mb", "jeopolitik", "kur_krizi", "ic_siyaset", "dogal_afet"}
        for e in POLITICAL_SHOCKS:
            # tarih parse edilebilir olmalı (olay etüdüne ham gider)
            pd.Timestamp(e["date"])
            assert e["category"] in valid_cats
            assert e["label"]
            assert e["tone"] in ("pozitif", "negatif", "notr")

    def test_dates_sorted_unique(self):
        dates = [e["date"] for e in POLITICAL_SHOCKS]
        assert dates == sorted(dates), "takvim kronolojik tutulmalı"
        assert len(dates) == len(set(dates))

    def test_category_filter(self):
        all_dates = political_event_dates()
        mb_dates = political_event_dates(categories=["mb"])
        assert set(mb_dates) <= set(all_dates)
        assert 0 < len(mb_dates) < len(all_dates)
        assert "2021-03-22" in mb_dates  # Ağbal'ın görevden alınması

    def test_format_includes_labels(self):
        text = format_shock_calendar()
        assert "ŞOK TAKVİMİ" in text
        assert "Brunson" in text and "darbe girişimi" in text


@pytest.mark.unit
class TestPoliticalNewsFilter:
    def test_filters_political_items(self, monkeypatch):
        items = [
            {"source": "X", "pub_date": "", "title": "Seçim anketinde son durum",
             "summary": "Partilerin oy oranları"},
            {"source": "X", "pub_date": "", "title": "Şirket bilanço açıkladı",
             "summary": "Net kâr arttı"},
            {"source": "X", "pub_date": "", "title": "Kredi notu kararı bekleniyor",
             "summary": "Moody's değerlendirmesi"},
        ]
        monkeypatch.setattr(
            "tradingagents.dataflows.political_events._fetch_feed",
            lambda label, url, limit, timeout: items,
        )
        res = fetch_political_news(feeds=[("Test", "http://example.invalid/rss")])
        assert res.ok
        assert "Seçim anketinde" in res.text
        assert "Kredi notu" in res.text
        assert "bilanço" not in res.text  # siyasi olmayan başlık elenmeli

    def test_empty_feeds_yield_placeholder(self, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.dataflows.political_events._fetch_feed",
            lambda label, url, limit, timeout: [],
        )
        res = fetch_political_news(feeds=[("Test", "http://example.invalid/rss")])
        assert not res.ok
        assert "ulaşılamadı" in res.text

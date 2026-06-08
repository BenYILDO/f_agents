"""Turkish macro-news fetcher — domestic macro regime read for Borsa İstanbul.

The News analyst leans on global/English sources and the Sentiment analyst
folds TCMB/rate/FX headlines in only as company-level *backdrop*. That buries
the single biggest driver of BIST: the domestic macro regime — TCMB policy
rate / PPK decisions, inflation (TÜFE/ÜFE), the lira, growth, budget/country
risk. This module surfaces exactly those headlines as a dedicated, theme-bucketed
prompt block for the Macro analyst.

It is deliberately built on the same plumbing as
:mod:`tradingagents.dataflows.turkish_news` (public RSS, no API key, no
registration) and reuses that module's feed parser/HTML stripper. Like its
sibling it degrades gracefully: every network/parse failure is swallowed
per-feed and the function always returns a formatted plaintext string (real
data or a clear placeholder) so the calling agent never special-cases missing
data.

Selection logic differs from :mod:`turkish_news`: instead of partitioning into
company-vs-market, this fetcher *keeps only* items that match a macro theme
(faiz, enflasyon, kur, büyüme, bütçe/risk, jeopolitik) and drops the rest as
company noise, then groups the survivors by theme so the agent reads the macro
picture by topic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from tradingagents.dataflows.data_health import SourceHealth, OK, EMPTY, any_ok

# Reuse the battle-tested RSS plumbing AND the vetted feed list from the sibling
# module so both fetchers parse identical, already-working sources (the
# sentiment analyst ships against these). We deliberately do NOT invent extra
# economy-desk feed IDs here — unverified URLs would just fail open and look
# like "no macro news". To check/extend live coverage, run
# ``scripts/check_macro_feeds.py`` where outbound network is allowed.
from tradingagents.dataflows.turkish_news import (  # noqa: F401
    DEFAULT_FEEDS as DEFAULT_MACRO_FEEDS,
    _fetch_feed,
    _strip_html,
)


@dataclass
class MacroNewsResult:
    """Themed macro-news block plus per-feed health for the data-health panel."""
    text: str
    sources: list[SourceHealth] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return any_ok(self.sources)

# Theme buckets in priority order. The first bucket whose keyword set hits the
# headline+summary claims the item, so policy-rate items outrank generic FX
# mentions. All keywords are matched case-folded against accent-preserving text.
MACRO_THEMES: tuple[tuple[str, frozenset[str]], ...] = (
    ("Faiz / Para Politikası", frozenset({
        "faiz", "ppk", "tcmb", "merkez bankası", "merkez banka", "politika faizi",
        "faiz kararı", "faiz indirimi", "faiz artırımı", "faiz indir", "faiz artır",
        "sıkılaşma", "gevşeme", "para politikası", "swap", "fed", "ecb",
    })),
    ("Enflasyon", frozenset({
        "enflasyon", "tüfe", "üfe", "çekirdek enflasyon", "fiyat artış",
        "fiyat endeksi", "aylık enflasyon", "yıllık enflasyon",
    })),
    ("Kur / Döviz", frozenset({
        "dolar", "euro", "kur", "döviz", "lira", "parite", "sterlin",
        "altın", "ons", "rezerv",
    })),
    ("Büyüme / İstihdam", frozenset({
        "büyüme", "gsyh", "gsyih", "işsizlik", "istihdam", "sanayi üretimi",
        "pmi", "kapasite kullanım", "ekonomik büyüme", "resesyon",
    })),
    ("Bütçe / Borç / Ülke Riski", frozenset({
        "bütçe", "cari açık", "cari denge", "cari işlemler", "cds", "risk primi",
        "tahvil", "hazine", "borçlanma", "dış borç",
    })),
    ("Politika / Jeopolitik", frozenset({
        "seçim", "jeopolitik", "yaptırım", "kredi notu", "derecelendirme",
        "moody", "fitch", "s&p", "not artır", "not indir",
    })),
)


def _classify_theme(text: str) -> Optional[str]:
    """Return the first macro theme whose keywords appear in ``text`` (or None).

    ``text`` should already be case-folded by the caller. None means the item is
    not macro-relevant and is dropped as company/market noise.
    """
    for label, keywords in MACRO_THEMES:
        if any(kw in text for kw in keywords):
            return label
    return None


def fetch_turkish_macro_news(
    feeds: Iterable[tuple[str, str]] = DEFAULT_MACRO_FEEDS,
    limit_per_feed: int = 15,
    max_items_per_theme: int = 4,
    timeout: float = 10.0,
    inter_request_delay: float = 0.3,
) -> MacroNewsResult:
    """Fetch Turkish macro headlines, bucketed by theme, with per-feed health.

    Pulls the macro feeds, keeps only items matching a macro theme, groups the
    survivors by theme (priority order), and renders a plaintext block. Returns
    a :class:`MacroNewsResult` carrying both the text and a :class:`SourceHealth`
    per feed so the caller can surface data gaps loudly instead of silently
    treating a placeholder as real news. Never raises.
    """
    # Collect + de-duplicate across feeds (same headline can syndicate),
    # recording per-feed health as we go.
    collected: list[dict] = []
    seen_titles: set[str] = set()
    sources: list[SourceHealth] = []
    for i, (label, url) in enumerate(feeds):
        if i > 0:
            time.sleep(inter_request_delay)
        items = _fetch_feed(label, url, limit_per_feed, timeout)
        if items:
            sources.append(SourceHealth(label, OK, count=len(items)))
        else:
            sources.append(SourceHealth(label, EMPTY, "yanıt yok / boş"))
        for item in items:
            key = item["title"].casefold()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            collected.append(item)

    if not collected:
        return MacroNewsResult(
            "<Türkçe makro haber kaynaklarına şu an ulaşılamadı "
            f"({', '.join(label for label, _ in feeds)})>",
            sources,
        )

    # Bucket by theme, preserving feed order within each theme.
    buckets: dict[str, list[dict]] = {label: [] for label, _ in MACRO_THEMES}
    for item in collected:
        haystack = f"{item['title']} {item['summary']}".casefold()
        theme = _classify_theme(haystack)
        if theme is not None:
            buckets[theme].append(item)

    if not any(buckets.values()):
        return MacroNewsResult(
            "<Güncel başlıklarda Türkiye makro temalı (faiz, enflasyon, kur, "
            "büyüme, bütçe) bir haber tespit edilemedi>",
            sources,
        )

    def _fmt(item: dict) -> str:
        line = f"  [{item['source']}"
        if item["pub_date"]:
            line += f" · {item['pub_date']}"
        line += f"] {item['title']}"
        if item["summary"]:
            excerpt = item["summary"]
            if len(excerpt) > 240:
                excerpt = excerpt[:240] + "…"
            line += f"\n    {excerpt}"
        return line

    blocks: list[str] = ["TÜRKİYE MAKRO GÖRÜNÜMÜ — güncel başlıklar (tema bazlı):"]
    for label, _ in MACRO_THEMES:
        items = buckets[label]
        if not items:
            continue
        blocks.append(
            f"▸ {label}:\n"
            + "\n".join(_fmt(it) for it in items[:max_items_per_theme])
        )
    return MacroNewsResult("\n\n".join(blocks), sources)

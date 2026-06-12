"""Türkiye siyasi/jeopolitik şok takvimi + siyasi haber filtresi.

Türkiye piyasası iç ve dış siyasetten dünya ortalamasının çok üzerinde
etkilenir: bir gecede gelen merkez bankası başkanı değişikliği, seçim
sürprizi ya da jeopolitik kriz endeksi günler içinde iki haneli oynatabilir.
Bu modül o hassasiyeti ajanlara İKİ kanaldan "benimsetir":

  1. **Küratörlü şok takvimi** — 2013'ten bugüne piyasayı sert oynatmış
     siyasi/jeopolitik olayların tarihleri. Jeopolitik analist bu tarihleri
     :mod:`macro_event_study` ile birleştirip "geçmiş siyasi şoklarda BU hisse
     ne yaptı?" sorusunun deterministik cevabını üretir (taban oranı).
  2. **Siyasi haber filtresi** — Türkçe RSS akışından yalnız siyaset/jeopolitik
     temalı başlıkları süzer (turkish_macro'nun tek temaya odaklanmış hali,
     daha geniş anahtar kelime setiyle).

Takvim statik ve versiyon kontrolündedir: olay etüdünün tekrarlanabilir
olması için kaynak listesi koddan okunur, ağa bağımlı değildir. Yeni şoklar
yaşandıkça listeye satır eklenir (test: ``tests/test_political_events.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from tradingagents.dataflows.data_health import SourceHealth, OK, EMPTY, any_ok
from tradingagents.dataflows.turkish_news import (
    DEFAULT_FEEDS as POLITICAL_FEEDS,
    _fetch_feed,
)

# Tarih (YYYY-MM-DD), olay etiketi, kategori, beklenen ilk piyasa tepkisi.
# Kriter: olayın açıklandığı/öğrenildiği İLK işlem günü yazılır (etüt forward
# getiri ölçtüğü için). Kategoriler: secim, mb (merkez bankası), jeopolitik,
# kur_krizi, ic_siyaset, dogal_afet.
POLITICAL_SHOCKS: tuple[dict, ...] = (
    {"date": "2013-05-31", "label": "Gezi protestoları başlangıcı", "category": "ic_siyaset", "tone": "negatif"},
    {"date": "2013-12-17", "label": "17 Aralık operasyonları", "category": "ic_siyaset", "tone": "negatif"},
    {"date": "2015-06-08", "label": "7 Haziran seçimi (tek parti çoğunluğu kaybı)", "category": "secim", "tone": "negatif"},
    {"date": "2015-11-02", "label": "1 Kasım seçimi (çoğunluk geri geldi)", "category": "secim", "tone": "pozitif"},
    {"date": "2015-11-24", "label": "Rus uçağının düşürülmesi", "category": "jeopolitik", "tone": "negatif"},
    {"date": "2016-07-18", "label": "15 Temmuz darbe girişimi (ilk işlem günü)", "category": "ic_siyaset", "tone": "negatif"},
    {"date": "2017-04-17", "label": "Anayasa referandumu sonucu", "category": "secim", "tone": "notr"},
    {"date": "2018-06-25", "label": "24 Haziran seçimi (cumhurbaşkanlığı sistemi)", "category": "secim", "tone": "notr"},
    {"date": "2018-08-10", "label": "Brunson krizi / kur şoku zirvesi", "category": "kur_krizi", "tone": "negatif"},
    {"date": "2019-03-22", "label": "Swap krizi / rezerv tartışması", "category": "kur_krizi", "tone": "negatif"},
    {"date": "2019-07-08", "label": "TCMB başkanı Çetinkaya'nın görevden alınması", "category": "mb", "tone": "negatif"},
    {"date": "2019-10-09", "label": "Barış Pınarı Harekâtı başlangıcı", "category": "jeopolitik", "tone": "negatif"},
    {"date": "2020-11-09", "label": "Ağbal ataması + ekonomi yönetimi değişimi", "category": "mb", "tone": "pozitif"},
    {"date": "2021-03-22", "label": "Ağbal'ın görevden alınması (gece kararnamesi)", "category": "mb", "tone": "negatif"},
    {"date": "2021-12-20", "label": "KKM duyurusu / kur şoku tepe noktası", "category": "kur_krizi", "tone": "notr"},
    {"date": "2022-02-24", "label": "Rusya'nın Ukrayna'yı işgali", "category": "jeopolitik", "tone": "negatif"},
    {"date": "2023-02-06", "label": "Kahramanmaraş depremleri (borsa 5 gün kapandı)", "category": "dogal_afet", "tone": "negatif"},
    {"date": "2023-05-15", "label": "14 Mayıs seçimi ilk turu (2. tura kalması)", "category": "secim", "tone": "negatif"},
    {"date": "2023-05-29", "label": "28 Mayıs 2. tur sonucu", "category": "secim", "tone": "notr"},
    {"date": "2023-06-05", "label": "Şimşek'in atanması (rasyonel politikaya dönüş)", "category": "mb", "tone": "pozitif"},
    {"date": "2023-10-09", "label": "İsrail-Hamas savaşı başlangıcı (ilk işlem günü)", "category": "jeopolitik", "tone": "negatif"},
    {"date": "2024-04-01", "label": "31 Mart yerel seçim sonucu", "category": "secim", "tone": "notr"},
    {"date": "2025-03-19", "label": "İmamoğlu'nun gözaltına alınması", "category": "ic_siyaset", "tone": "negatif"},
)

# Siyasi haber filtresi anahtar kelimeleri (turkish_macro'daki jeopolitik
# temasından geniştir: parti/lider/diplomasi/savunma da yakalanır).
POLITICAL_KEYWORDS: frozenset[str] = frozenset({
    "seçim", "anket", "parti", "cumhurbaşkan", "meclis", "kabine", "bakan",
    "muhalefet", "chp", "akp", "ak parti", "mhp", "iyi parti", "dem parti",
    "jeopolitik", "yaptırım", "ambargo", "nato", "ab ", "avrupa birliği",
    "abd ile", "rusya", "ukrayna", "israil", "iran", "suriye", "gazze",
    "savaş", "operasyon", "harekat", "harekât", "gerilim", "diplomasi",
    "kredi notu", "derecelendirme", "moody", "fitch", "s&p",
    "anayasa", "kayyum", "gözaltı", "tutuklama", "protesto", "miting",
})


def political_event_dates(categories: Iterable[str] | None = None) -> list[str]:
    """Şok takviminden (istenirse kategoriye süzülmüş) tarih listesi döndürür."""
    cats = set(categories) if categories else None
    return [e["date"] for e in POLITICAL_SHOCKS
            if cats is None or e["category"] in cats]


def format_shock_calendar() -> str:
    """Şok takvimini LLM prompt'una uygun, kategorili plaintext yapar."""
    lines = ["TÜRKİYE SİYASİ/JEOPOLİTİK ŞOK TAKVİMİ (küratörlü, 2013→):"]
    for e in POLITICAL_SHOCKS:
        lines.append(f"  - {e['date']} · {e['label']} [{e['category']}, ilk tepki: {e['tone']}]")
    return "\n".join(lines)


@dataclass
class PoliticalNewsResult:
    """Siyasi başlık bloğu + besleme sağlığı."""
    text: str
    sources: list[SourceHealth] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return any_ok(self.sources)


def fetch_political_news(
    feeds: Iterable[tuple[str, str]] = POLITICAL_FEEDS,
    limit_per_feed: int = 15,
    max_items: int = 12,
    timeout: float = 10.0,
    inter_request_delay: float = 0.3,
) -> PoliticalNewsResult:
    """Türkçe akışlardan yalnız siyaset/jeopolitik başlıkları süzer. Asla istisna fırlatmaz."""
    collected: list[dict] = []
    seen: set[str] = set()
    sources: list[SourceHealth] = []
    for i, (label, url) in enumerate(feeds):
        if i > 0:
            time.sleep(inter_request_delay)
        items = _fetch_feed(label, url, limit_per_feed, timeout)
        sources.append(SourceHealth(label, OK, count=len(items)) if items
                       else SourceHealth(label, EMPTY, "yanıt yok / boş"))
        for item in items:
            key = item["title"].casefold()
            if key in seen:
                continue
            seen.add(key)
            haystack = f"{item['title']} {item['summary']}".casefold()
            if any(kw in haystack for kw in POLITICAL_KEYWORDS):
                collected.append(item)

    if not collected:
        placeholder = ("<Güncel akışta siyaset/jeopolitik temalı başlık tespit edilemedi>"
                       if any_ok(sources)
                       else "<Türkçe haber kaynaklarına şu an ulaşılamadı>")
        return PoliticalNewsResult(placeholder, sources)

    lines = ["SİYASET / JEOPOLİTİK BAŞLIKLAR (güncel, süzülmüş):"]
    for item in collected[:max_items]:
        line = f"  [{item['source']}"
        if item["pub_date"]:
            line += f" · {item['pub_date']}"
        line += f"] {item['title']}"
        if item["summary"]:
            excerpt = item["summary"][:200]
            line += f"\n    {excerpt}{'…' if len(item['summary']) > 200 else ''}"
        lines.append(line)
    return PoliticalNewsResult("\n".join(lines), sources)

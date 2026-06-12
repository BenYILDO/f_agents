"""Altın & döviz anlık görünümü — ons, gram altın, USDTRY, XU100 ilişkisi.

Türk yatırımcının üç referans varlığı vardır: BIST, dolar, altın. Bu modül
o üçlünün güncel durumunu tek blokta toplar:

  - **Ons altın** (GC=F vadeli) ve USD/TRY → **gram altın** TL fiyatı
    (ons × kur ÷ 31.1035). Gram altın Türkiye'de tasarrufun fiili birimi
    olduğundan BIST'e alternatif maliyetin göstergesidir.
  - 1 ay / 3 ay / 1 yıl getirileri: BIST 100, gram altın ve dolar **TL bazında
    yan yana** — "borsa gerçekten kazandırıyor mu, yoksa kurun gerisinde mi?"
    sorusunun deterministik cevabı.
  - XU100/gram-altın oranı: BIST'in altın cinsinden reel seviyesi.

Çıktı hem Streamlit "Altın & Döviz" sayfasını hem de Makro analistin prompt
bloğunu besler. Asla istisna fırlatmaz; her seri için SourceHealth raporlanır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.data_health import SourceHealth, OK, EMPTY, ERROR

GRAMS_PER_OUNCE = 31.1035

# (etiket, yfinance sembolü)
_SERIES = (
    ("Ons Altın (USD)", "GC=F"),
    ("USD/TRY", "TRY=X"),
    ("BIST 100", "XU100.IS"),
)
_HORIZONS = (("1 ay", 21), ("3 ay", 63), ("1 yıl", 252))


@dataclass
class GoldFxResult:
    ok: bool
    text: str = ""                       # LLM/markdown bloğu
    frames: dict = field(default_factory=dict)    # etiket -> Close serisi (grafik için)
    table: pd.DataFrame | None = None    # getiri karşılaştırma tablosu
    snapshot: dict = field(default_factory=dict)  # etiket -> son değer
    sources: list[SourceHealth] = field(default_factory=list)


def _fetch_close(symbol: str, period: str = "2y") -> pd.Series | None:
    try:
        hist = yf.Ticker(symbol).history(period=period, interval="1d")
    except Exception:  # noqa: BLE001
        return None
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    s = hist["Close"].dropna()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s if len(s) >= 30 else None


def _ret(s: pd.Series, bars: int) -> float | None:
    if s is None or len(s) <= bars:
        return None
    return round(float(s.iloc[-1] / s.iloc[-1 - bars] - 1) * 100, 1)


def fetch_gold_fx_snapshot() -> GoldFxResult:
    """Altın/döviz/BIST görünümünü tek seferde çeker ve formatlar."""
    closes: dict[str, pd.Series] = {}
    sources: list[SourceHealth] = []
    for label, symbol in _SERIES:
        s = _fetch_close(symbol)
        if s is None:
            sources.append(SourceHealth(label, ERROR, "veri alınamadı"))
        else:
            closes[label] = s
            sources.append(SourceHealth(label, OK, count=len(s)))

    ons = closes.get("Ons Altın (USD)")
    usdtry = closes.get("USD/TRY")
    xu100 = closes.get("BIST 100")

    if ons is not None and usdtry is not None:
        # Gram altın TL: iki seriyi ortak günlere hizala
        aligned = pd.concat([ons, usdtry], axis=1, keys=["ons", "kur"]).dropna()
        gram = aligned["ons"] * aligned["kur"] / GRAMS_PER_OUNCE
        closes["Gram Altın (TL)"] = gram
        sources.append(SourceHealth("Gram Altın (TL, türetilmiş)", OK, count=len(gram)))
    else:
        sources.append(SourceHealth("Gram Altın (TL, türetilmiş)", EMPTY, "ons/kur eksik"))

    if not closes:
        return GoldFxResult(False, text="<Altın/döviz verilerine şu an ulaşılamadı>",
                            sources=sources)

    snapshot = {label: round(float(s.iloc[-1]), 2) for label, s in closes.items()}

    rows = []
    for label in ("BIST 100", "Gram Altın (TL)", "USD/TRY"):
        s = closes.get(label)
        if s is None:
            continue
        rows.append({"Varlık": label,
                     **{h: _ret(s, bars) for h, bars in _HORIZONS}})
    table = pd.DataFrame(rows) if rows else None

    lines = ["ALTIN & DÖVİZ GÖRÜNÜMÜ (deterministik, yfinance):"]
    for label, val in snapshot.items():
        lines.append(f"  - {label}: {val}")
    if table is not None and not table.empty:
        lines.append("")
        lines.append("TL bazında getiri karşılaştırması (%):")
        for _, row in table.iterrows():
            parts = [f"{h}: {row[h]:+.1f}" for h, _ in _HORIZONS if pd.notna(row[h])]
            lines.append(f"  - {row['Varlık']}: " + " · ".join(parts))
    if xu100 is not None and "Gram Altın (TL)" in closes:
        aligned = pd.concat([xu100, closes["Gram Altın (TL)"]], axis=1,
                            keys=["xu", "gram"]).dropna()
        if not aligned.empty:
            ratio = aligned["xu"] / aligned["gram"]
            pct = float((ratio <= ratio.iloc[-1]).mean()) * 100
            lines += ["", (f"XU100/gram-altın oranı: {ratio.iloc[-1]:.1f} "
                           f"(2 yıllık dağılımda %{pct:.0f} yüzdelik) — düşük yüzdelik, "
                           "BIST'in altına göre tarihsel ucuz olduğunu gösterir.")]
            closes["XU100/Gram Altın"] = ratio

    return GoldFxResult(True, text="\n".join(lines), frames=closes,
                        table=table, snapshot=snapshot, sources=sources)

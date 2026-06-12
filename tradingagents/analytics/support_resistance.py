"""Destek/Direnç seviyeleri — pivot kümeleme + Fibonacci + klasik pivot noktası.

Üç bağımsız yöntemle seviye üretir ve hepsini tek listede birleştirir:

  1. **Pivot kümeleme** — geçmiş tepe/dip pivotları %1,5 bant içinde kümelenir;
     bir seviyeye ne kadar çok dokunulduysa o kadar güçlüdür (touch count).
  2. **Fibonacci düzeltme** — son 6 ayın majör salınımı üzerinden 23.6/38.2/
     50/61.8/78.6 seviyeleri.
  3. **Klasik pivot noktası** — son barın (P, R1-R2, S1-S2) seviyeleri (gün içi
     referans).

Çıktı hem Streamlit S/R tablosunu hem de kompozit skorun "fiyat desteğe mi
dirence mi yakın" bileşenini besler.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingagents.analytics.patterns import find_pivots


@dataclass
class Level:
    price: float
    kind: str      # "destek" | "direnç"
    source: str    # "pivot" | "fibonacci" | "klasik"
    strength: int  # dokunuş sayısı (pivot) ya da 1
    label: str     # gösterim etiketi


def _cluster_pivots(piv: pd.DataFrame, band: float = 0.015) -> list[tuple[float, int]]:
    """Pivot fiyatlarını ±band içinde kümeler; (küme ortalaması, dokunuş) döner."""
    prices = sorted(float(p) for p in piv["price"])
    clusters: list[list[float]] = []
    for p in prices:
        if clusters and p <= clusters[-1][-1] * (1 + band):
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [(sum(c) / len(c), len(c)) for c in clusters]


def compute_levels(df: pd.DataFrame, order: int = 5, swing_window: int = 126) -> list[Level]:
    """Tüm yöntemlerin seviyelerini, güncel fiyata göre etiketleyip döndürür."""
    if df.empty or len(df) < 30:
        return []
    close = float(df["Close"].iloc[-1])
    levels: list[Level] = []

    # 1) Pivot kümeleri — en az 2 dokunuş olanlar anlamlı
    piv = find_pivots(df.tail(252), order)
    if not piv.empty:
        for price, touches in _cluster_pivots(piv):
            if touches < 2:
                continue
            kind = "destek" if price < close else "direnç"
            levels.append(Level(round(price, 2), kind, "pivot", touches,
                                f"{touches} dokunuş"))

    # 2) Fibonacci — son swing_window barın min/max salınımı
    win = df.tail(swing_window)
    lo, hi = float(win["Low"].min()), float(win["High"].max())
    if hi > lo:
        for ratio in (0.236, 0.382, 0.5, 0.618, 0.786):
            price = hi - (hi - lo) * ratio
            kind = "destek" if price < close else "direnç"
            levels.append(Level(round(price, 2), kind, "fibonacci", 1,
                                f"Fib %{ratio * 100:.1f}".replace(".0", "")))

    # 3) Klasik pivot noktası (son bar)
    last = df.iloc[-1]
    p = (float(last["High"]) + float(last["Low"]) + float(last["Close"])) / 3
    classic = {
        "P": p,
        "R1": 2 * p - float(last["Low"]), "S1": 2 * p - float(last["High"]),
        "R2": p + (float(last["High"]) - float(last["Low"])),
        "S2": p - (float(last["High"]) - float(last["Low"])),
    }
    for name, price in classic.items():
        kind = "destek" if price < close else "direnç"
        levels.append(Level(round(price, 2), kind, "klasik", 1, name))

    levels.sort(key=lambda lv: lv.price)
    return levels


def nearest_levels(levels: list[Level], close: float, n: int = 3) -> tuple[list[Level], list[Level]]:
    """(en yakın n destek, en yakın n direnç) — fiyata yakınlık sırasıyla."""
    supports = sorted((lv for lv in levels if lv.price < close),
                      key=lambda lv: close - lv.price)[:n]
    resistances = sorted((lv for lv in levels if lv.price >= close),
                         key=lambda lv: lv.price - close)[:n]
    return supports, resistances


def sr_score(levels: list[Level], close: float) -> float:
    """Fiyatın S/R konumunu [-1, +1] skora çevirir.

    Güçlü (çok dokunuşlu) desteğin hemen üstünde olmak pozitif (alım bölgesi),
    güçlü direncin hemen altında olmak negatif. %2'den uzak seviyeler etkisiz.
    """
    score = 0.0
    for lv in levels:
        dist = abs(lv.price - close) / max(close, 1e-9)
        if dist > 0.02 or lv.source == "klasik":
            continue
        weight = min(lv.strength, 4) / 4 * (1 - dist / 0.02)
        score += weight if lv.kind == "destek" else -weight
    return max(-1.0, min(1.0, score))

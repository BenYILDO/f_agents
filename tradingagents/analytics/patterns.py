"""Grafik formasyonu tespiti — pivot (zigzag) tabanlı, deterministik.

Klasik fiyat formasyonlarını tespit eder: ikili tepe/dip, (ters) omuz-baş-omuz,
yükselen/alçalan/simetrik üçgen. Yaklaşım: önce yerel tepe/dip pivotları
bulunur (±``order`` bar penceresinde ekstremum), sonra son pivot dizisi
üzerinde geometrik kurallar aranır.

Tasarım ilkeleri:
  - **Muhafazakâr ol.** Şüpheli eşleşme üretmektense formasyon kaçırmak yeğdir;
    LLM'e ve kompozit skora giden her formasyon sinyal ağırlığı taşır.
  - **Teyit ayrı raporlanır.** Boyun çizgisi kırılmamış OBO "oluşum aşamasında"
    diye etiketlenir; kırılmışsa "teyitli". Trader ikisini farklı oynar.
  - Asla istisna fırlatmaz; veri yetersizse boş liste döner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PatternHit:
    name: str           # Türkçe formasyon adı
    direction: str      # "boğa" | "ayı"
    confirmed: bool     # boyun çizgisi / kırılım teyidi geldi mi
    start: str          # formasyonun ilk pivotunun tarihi
    end: str            # son pivot / teyit tarihi
    neckline: float | None  # boyun çizgisi / kırılım seviyesi (varsa)
    note: str = ""      # kısa Türkçe açıklama (hedef fiyat vb.)


def find_pivots(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """Yerel tepe (H) ve dip (L) pivotlarını döndürür.

    Bir bar, ±``order`` bar penceresindeki en yüksek High ise tepe; en düşük
    Low ise dip sayılır. Dönen DataFrame kolonları: ``kind`` ("H"/"L"),
    ``price``, index = tarih.
    """
    if len(df) < order * 2 + 1:
        return pd.DataFrame(columns=["kind", "price"])
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    rows = []
    for i in range(order, len(df) - order):
        win_h = high[i - order: i + order + 1]
        win_l = low[i - order: i + order + 1]
        # Plato tepeler (ardışık eşit değer) gerçek veride olur: penceredeki
        # maksimumun İLK görüldüğü bar pivot sayılır (argmax/argmin ilk indeksi
        # döndürür), böylece eşitlik pivotu tamamen düşürmez ama çift de saymaz.
        if int(np.argmax(win_h)) == order:
            rows.append((df.index[i], "H", float(high[i])))
        elif int(np.argmin(win_l)) == order:
            rows.append((df.index[i], "L", float(low[i])))
    piv = pd.DataFrame(rows, columns=["date", "kind", "price"]).set_index("date")
    # Ardışık aynı tür pivotlarda yalnız ekstrem olanı tut (zigzag temizliği)
    keep, last_kind = [], None
    for date, row in piv.iterrows():
        if row["kind"] == last_kind and keep:
            prev_date, prev = keep[-1]
            better = (row["kind"] == "H" and row["price"] > prev["price"]) or \
                     (row["kind"] == "L" and row["price"] < prev["price"])
            if better:
                keep[-1] = (date, row)
            continue
        keep.append((date, row))
        last_kind = row["kind"]
    return pd.DataFrame([r for _, r in keep], index=[d for d, _ in keep])


def _fmt(d) -> str:
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def detect_patterns(
    df: pd.DataFrame,
    order: int = 5,
    tol: float = 0.03,
    max_age_bars: int = 60,
) -> list[PatternHit]:
    """Son ``max_age_bars`` bar içinde biten formasyonları döndürür.

    ``tol`` iki tepenin/dibin "eşit" sayılması için göreli tolerans (BIST
    oynaklığına göre %3 makul). Eski formasyonlar elenir — bugün üç ay önceki
    ikili tepenin işlem değeri yoktur.
    """
    hits: list[PatternHit] = []
    piv = find_pivots(df, order)
    if len(piv) < 3:  # ikili tepe/dip için 3 pivot yeter; OBO/üçgen kendileri eler
        return hits
    last_close = float(df["Close"].iloc[-1])
    cutoff_pos = max(0, len(df) - max_age_bars)
    cutoff_date = df.index[cutoff_pos]

    seq = [(d, r["kind"], float(r["price"])) for d, r in piv.iterrows()]

    # ── İkili tepe / dip: H-L-H (tepeler ~eşit) / L-H-L (dipler ~eşit) ──────
    for i in range(len(seq) - 2):
        (d1, k1, p1), (d2, k2, p2), (d3, k3, p3) = seq[i], seq[i + 1], seq[i + 2]
        if d3 < cutoff_date:
            continue
        if (k1, k2, k3) == ("H", "L", "H") and _near(p1, p3, tol) and p2 < min(p1, p3) * (1 - 0.01):
            confirmed = last_close < p2
            hits.append(PatternHit(
                "İkili Tepe", "ayı", confirmed, _fmt(d1), _fmt(d3), p2,
                f"boyun {p2:.2f}; hedef ≈ {p2 - (max(p1, p3) - p2):.2f}" if confirmed
                else f"boyun {p2:.2f} henüz kırılmadı",
            ))
        if (k1, k2, k3) == ("L", "H", "L") and _near(p1, p3, tol) and p2 > max(p1, p3) * (1 + 0.01):
            confirmed = last_close > p2
            hits.append(PatternHit(
                "İkili Dip", "boğa", confirmed, _fmt(d1), _fmt(d3), p2,
                f"boyun {p2:.2f}; hedef ≈ {p2 + (p2 - min(p1, p3)):.2f}" if confirmed
                else f"boyun {p2:.2f} henüz kırılmadı",
            ))

    # ── (Ters) Omuz-Baş-Omuz: H-L-H-L-H (baş en yüksek, omuzlar ~eşit) ──────
    for i in range(len(seq) - 4):
        window = seq[i:i + 5]
        kinds = tuple(k for _, k, _ in window)
        (d1, _, s1), (_, _, n1), (_, _, head), (_, _, n2), (d5, _, s2) = window
        if d5 < cutoff_date:
            continue
        if kinds == ("H", "L", "H", "L", "H") and head > max(s1, s2) * (1 + 0.02) and _near(s1, s2, tol * 1.5):
            neckline = (n1 + n2) / 2
            confirmed = last_close < neckline
            hits.append(PatternHit(
                "Omuz-Baş-Omuz (OBO)", "ayı", confirmed, _fmt(d1), _fmt(d5), neckline,
                f"boyun {neckline:.2f}; hedef ≈ {neckline - (head - neckline):.2f}" if confirmed
                else f"boyun {neckline:.2f} henüz kırılmadı",
            ))
        if kinds == ("L", "H", "L", "H", "L") and head < min(s1, s2) * (1 - 0.02) and _near(s1, s2, tol * 1.5):
            neckline = (n1 + n2) / 2
            confirmed = last_close > neckline
            hits.append(PatternHit(
                "Ters Omuz-Baş-Omuz (TOBO)", "boğa", confirmed, _fmt(d1), _fmt(d5), neckline,
                f"boyun {neckline:.2f}; hedef ≈ {neckline + (neckline - head):.2f}" if confirmed
                else f"boyun {neckline:.2f} henüz kırılmadı",
            ))

    # ── Üçgenler: son 4 pivotun (2H + 2L) eğimlerinden ───────────────────────
    recent = [t for t in seq if t[0] >= cutoff_date]
    highs = [(d, p) for d, k, p in recent if k == "H"][-2:]
    lows = [(d, p) for d, k, p in recent if k == "L"][-2:]
    if len(highs) == 2 and len(lows) == 2:
        h_flat = _near(highs[0][1], highs[1][1], tol / 2)
        l_flat = _near(lows[0][1], lows[1][1], tol / 2)
        h_down = highs[1][1] < highs[0][1] * (1 - tol / 2)
        l_up = lows[1][1] > lows[0][1] * (1 + tol / 2)
        start = _fmt(min(highs[0][0], lows[0][0]))
        end = _fmt(max(highs[1][0], lows[1][0]))
        if h_flat and l_up:
            level = max(highs[0][1], highs[1][1])
            confirmed = last_close > level
            hits.append(PatternHit(
                "Yükselen Üçgen", "boğa", confirmed, start, end, level,
                f"direnç {level:.2f}" + (" yukarı kırıldı" if confirmed else " test ediliyor"),
            ))
        elif l_flat and h_down:
            level = min(lows[0][1], lows[1][1])
            confirmed = last_close < level
            hits.append(PatternHit(
                "Alçalan Üçgen", "ayı", confirmed, start, end, level,
                f"destek {level:.2f}" + (" aşağı kırıldı" if confirmed else " test ediliyor"),
            ))
        elif h_down and l_up:
            hits.append(PatternHit(
                "Simetrik Üçgen (sıkışma)", "nötr", False, start, end, None,
                "kırılım yönü beklenmeli — hacimle teyit şart",
            ))

    # En yeni bitenler önce; aynı formasyondan teyitli olan üstte
    hits.sort(key=lambda h: (h.end, h.confirmed), reverse=True)
    return hits


def pattern_score(hits: list[PatternHit]) -> float:
    """Formasyon listesini [-1, +1] tek skora indirger (teyitli 2x ağırlık)."""
    if not hits:
        return 0.0
    raw = 0.0
    for h in hits:
        w = 2.0 if h.confirmed else 0.7
        if h.direction == "boğa":
            raw += w
        elif h.direction == "ayı":
            raw -= w
    return max(-1.0, min(1.0, raw / 3.0))

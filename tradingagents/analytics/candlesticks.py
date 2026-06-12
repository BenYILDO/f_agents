"""Mum (candlestick) formasyonu tespiti — saf pandas, TA-Lib'siz.

Klasik tek/çift/üçlü mum formasyonlarını tespit eder ve her birini Türkçe
adı, yönü (boğa/ayı) ve göreli gücüyle döndürür. Formasyonlar bağlama duyarlı
okunur: çekiç ancak düşüş sonrası (dip bölgesinde) anlamlıdır, kayan yıldız
ancak yükseliş sonrası — bu yüzden kısa vadeli trend filtresi (SMA10 eğimi)
uygulanır ki yatay piyasada her fitilli mum "çekiç" diye işaretlenmesin.

Çıktı iki tüketiciye gider: Streamlit Teknik Analiz sayfasındaki formasyon
tablosu ve kompozit skorun mum bileşeni (son birkaç barın net boğa/ayı yükü).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# (kod, Türkçe ad, yön, güç 1-3) — güç klasik literatürdeki güvenilirlik sırası
PATTERN_META = {
    "hammer":            ("Çekiç", "boğa", 2),
    "inverted_hammer":   ("Ters Çekiç", "boğa", 1),
    "bullish_engulfing": ("Boğa Yutan", "boğa", 3),
    "piercing":          ("Delici Mum", "boğa", 2),
    "morning_star":      ("Sabah Yıldızı", "boğa", 3),
    "three_white":       ("Üç Beyaz Asker", "boğa", 3),
    "bullish_harami":    ("Boğa Harami", "boğa", 1),
    "shooting_star":     ("Kayan Yıldız", "ayı", 2),
    "hanging_man":       ("Asılı Adam", "ayı", 1),
    "bearish_engulfing": ("Ayı Yutan", "ayı", 3),
    "dark_cloud":        ("Kara Bulut", "ayı", 2),
    "evening_star":      ("Akşam Yıldızı", "ayı", 3),
    "three_black":       ("Üç Kara Karga", "ayı", 3),
    "bearish_harami":    ("Ayı Harami", "ayı", 1),
    "doji":              ("Doji (kararsızlık)", "nötr", 1),
}


@dataclass
class CandleHit:
    date: str
    code: str
    name: str       # Türkçe ad
    direction: str  # "boğa" | "ayı" | "nötr"
    strength: int   # 1-3


def detect_candlesticks(df: pd.DataFrame) -> pd.DataFrame:
    """Her bar için formasyon bayraklarını içeren bool DataFrame döndürür.

    Girdi OHLCV; çıktı, ``PATTERN_META`` anahtarlarıyla aynı adlı bool
    kolonlar. Vektörel hesap — uzun seride de hızlıdır.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, pd.NA)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    bull = c > o
    bear = c < o
    avg_body = body.rolling(14).mean()

    # Kısa vadeli bağlam: SMA10 eğimi (çekiç dip ister, kayan yıldız tepe)
    sma10 = c.rolling(10).mean()
    downtrend = sma10 < sma10.shift(3)
    uptrend = sma10 > sma10.shift(3)

    out = pd.DataFrame(index=df.index)

    out["doji"] = (body <= 0.1 * rng).fillna(False)

    pin_low = (lower >= 2 * body) & (upper <= 0.3 * lower) & (body > 0)
    pin_high = (upper >= 2 * body) & (lower <= 0.3 * upper) & (body > 0)
    out["hammer"] = (pin_low & downtrend).fillna(False)
    out["hanging_man"] = (pin_low & uptrend).fillna(False)
    out["inverted_hammer"] = (pin_high & downtrend).fillna(False)
    out["shooting_star"] = (pin_high & uptrend).fillna(False)

    # Yutan: bugünkü gövde dünkü gövdeyi tamamen kapsar, yönler zıt
    prev_o, prev_c = o.shift(), c.shift()
    engulf = (pd.concat([o, c], axis=1).max(axis=1) >= pd.concat([prev_o, prev_c], axis=1).max(axis=1)) & \
             (pd.concat([o, c], axis=1).min(axis=1) <= pd.concat([prev_o, prev_c], axis=1).min(axis=1)) & \
             (body > avg_body * 0.8)
    out["bullish_engulfing"] = (engulf & bull & bear.shift().fillna(False)).fillna(False)
    out["bearish_engulfing"] = (engulf & bear & bull.shift().fillna(False)).fillna(False)

    # Harami: bugünkü gövde dünkü büyük gövdenin içinde kalır
    harami = (pd.concat([o, c], axis=1).max(axis=1) <= pd.concat([prev_o, prev_c], axis=1).max(axis=1)) & \
             (pd.concat([o, c], axis=1).min(axis=1) >= pd.concat([prev_o, prev_c], axis=1).min(axis=1)) & \
             (body.shift() > avg_body.shift())
    out["bullish_harami"] = (harami & bull & bear.shift().fillna(False) & downtrend).fillna(False)
    out["bearish_harami"] = (harami & bear & bull.shift().fillna(False) & uptrend).fillna(False)

    # Delici mum / kara bulut: %50'den fazla geri alma
    mid_prev = (prev_o + prev_c) / 2
    out["piercing"] = (bear.shift().fillna(False) & bull & (o < prev_c) &
                       (c > mid_prev) & (c < prev_o)).fillna(False)
    out["dark_cloud"] = (bull.shift().fillna(False) & bear & (o > prev_c) &
                         (c < mid_prev) & (c > prev_o)).fillna(False)

    # Sabah/akşam yıldızı: büyük gövde + küçük gövde (gap'e bakılmaz, BIST'te
    # gap nadirdir) + zıt yönlü büyük gövde ilk mumun ortasını geçer
    big = body > avg_body
    small = body < 0.5 * avg_body
    out["morning_star"] = (bear.shift(2).fillna(False) & big.shift(2).fillna(False) &
                           small.shift().fillna(False) & bull & big &
                           (c > (o.shift(2) + c.shift(2)) / 2)).fillna(False)
    out["evening_star"] = (bull.shift(2).fillna(False) & big.shift(2).fillna(False) &
                           small.shift().fillna(False) & bear & big &
                           (c < (o.shift(2) + c.shift(2)) / 2)).fillna(False)

    # Üç beyaz asker / üç kara karga: ardışık üç belirgin gövde, her kapanış öncekini aşar
    solid = body > 0.6 * avg_body
    out["three_white"] = (bull & bull.shift().fillna(False) & bull.shift(2).fillna(False) &
                          solid & solid.shift().fillna(False) & solid.shift(2).fillna(False) &
                          (c > c.shift()) & (c.shift() > c.shift(2))).fillna(False)
    out["three_black"] = (bear & bear.shift().fillna(False) & bear.shift(2).fillna(False) &
                          solid & solid.shift().fillna(False) & solid.shift(2).fillna(False) &
                          (c < c.shift()) & (c.shift() < c.shift(2))).fillna(False)
    return out


def recent_candle_hits(df: pd.DataFrame, lookback: int = 10) -> list[CandleHit]:
    """Son ``lookback`` bardaki formasyonları (en yeniden eskiye) listeler."""
    flags = detect_candlesticks(df).tail(lookback)
    hits: list[CandleHit] = []
    for idx, row in flags.iloc[::-1].iterrows():
        date = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        for code, fired in row.items():
            if fired:
                name, direction, strength = PATTERN_META[code]
                hits.append(CandleHit(date, code, name, direction, strength))
    return hits


def candle_score(df: pd.DataFrame, lookback: int = 5) -> float:
    """Son barların net mum yükü: [-1, +1] (boğa pozitif, yakın bar ağır)."""
    hits = recent_candle_hits(df, lookback)
    if not hits:
        return 0.0
    raw, max_possible = 0.0, 0.0
    last_date = hits[0].date
    for hit in hits:
        weight = 1.0 if hit.date == last_date else 0.6
        max_possible += 3 * weight
        if hit.direction == "boğa":
            raw += hit.strength * weight
        elif hit.direction == "ayı":
            raw -= hit.strength * weight
    return max(-1.0, min(1.0, raw / max(max_possible, 1.0) * 2))

"""Sezonsallık analizi — ay ve haftanın günü bazlı tarihsel istatistikler.

"Bu hisse Aralık aylarında tarihsel olarak nasıl performans gösterdi?"
sorusuna deterministik cevap verir: son N yılın aylık getiri ortalaması,
medyanı ve pozitif-ay oranı (win rate). BIST'te sezonsallık belirgin olabilir
(temettü sezonu, bilanço dönemleri, yıl sonu portföy makyajı, Ramazan/bayram
haftaları likidite düşüşü) — ama az örneklemli istatistik yanıltıcıdır, bu
yüzden örnek sayısı her satırda raporlanır ve 5 yıldan az veride skor
sıfıra çekilir.

Çıktılar: Streamlit Sezonsallık sayfası (ısı tablosu) ve kompozit skorun
küçük-ağırlıklı sezonsallık bileşeni.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MONTH_NAMES_TR = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
    7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}
DAY_NAMES_TR = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma"}


@dataclass
class SeasonalityResult:
    ok: bool
    monthly: pd.DataFrame | None = None    # index: ay no — mean/median/win_rate/count
    daily: pd.DataFrame | None = None      # index: gün no — mean/win_rate/count
    years: int = 0
    reason: str = ""


def compute_seasonality(df: pd.DataFrame) -> SeasonalityResult:
    """Günlük OHLCV'den aylık + haftanın günü istatistiklerini çıkarır.

    ``df`` günlük bar olmalı (DatetimeIndex). En az ~2 yıl veri ister; yoksa
    ``ok=False`` döner — yarım yıllık veriden "sezonsallık" uydurulmaz.
    """
    if df is None or df.empty or len(df) < 400:
        return SeasonalityResult(False, reason="Sezonsallık için en az ~2 yıl günlük veri gerekli.")

    close = df["Close"].dropna()
    years = round(len(close) / 252)

    # Aylık getiriler: ay sonu kapanışları üzerinden
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change().dropna() * 100
    grp = monthly_ret.groupby(monthly_ret.index.month)
    monthly = pd.DataFrame({
        "ay": [MONTH_NAMES_TR[m] for m in grp.mean().index],
        "ort_getiri_pct": grp.mean().round(2),
        "medyan_pct": grp.median().round(2),
        "pozitif_oran": (grp.apply(lambda s: (s > 0).mean()) * 100).round(0),
        "ornek": grp.count(),
    })

    daily_ret = close.pct_change().dropna() * 100
    weekday = daily_ret[daily_ret.index.dayofweek < 5]
    dgrp = weekday.groupby(weekday.index.dayofweek)
    daily = pd.DataFrame({
        "gun": [DAY_NAMES_TR[d] for d in dgrp.mean().index],
        "ort_getiri_pct": dgrp.mean().round(3),
        "pozitif_oran": (dgrp.apply(lambda s: (s > 0).mean()) * 100).round(0),
        "ornek": dgrp.count(),
    })
    return SeasonalityResult(True, monthly=monthly, daily=daily, years=years)


def month_edge(result: SeasonalityResult, month: int) -> tuple[float, str]:
    """Verilen ayın tarihsel kenarını (skor [-1, +1], Türkçe özet) döndürür.

    Skor; ortalama getiri ve pozitif-oranın birleşimi. Örneklem < 5 yıl ise
    güven yetersiz → skor 0 ve bunu söyleyen bir özet döner.
    """
    if not result.ok or result.monthly is None or month not in result.monthly.index:
        return 0.0, "Sezonsallık verisi yetersiz."
    row = result.monthly.loc[month]
    n = int(row["ornek"])
    name = row["ay"]
    summary = (f"{name} ayı tarihsel: ort %{row['ort_getiri_pct']:+.1f}, "
               f"pozitif oran %{row['pozitif_oran']:.0f} ({n} örnek)")
    if n < 5:
        return 0.0, summary + " — örneklem küçük, sinyal olarak kullanılmadı."
    # Ortalama getiriyi ±%5'te, pozitif oranı %50 etrafında normalize et
    ret_part = max(-1.0, min(1.0, float(row["ort_getiri_pct"]) / 5.0))
    win_part = max(-1.0, min(1.0, (float(row["pozitif_oran"]) - 50.0) / 25.0))
    return round(0.5 * ret_part + 0.5 * win_part, 2), summary

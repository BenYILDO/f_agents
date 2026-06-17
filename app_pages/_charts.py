"""Paylaşılan grafik yardımcıları — Altair mum (candlestick) grafiği vb.

Portföyüm ve Otomatik Analiz ekranları aynı mum grafiğini kullanır; tek yerde
tutarak tekrarı önler.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def candlestick_chart(df: pd.DataFrame, bars: int = 120, height: int = 340) -> alt.Chart:
    """OHLC verisinden mum grafiği üretir (yeşil=yükseliş, kırmızı=düşüş).

    ``df`` index'i tarih, sütunları Open/High/Low/Close olmalı. Son ``bars``
    bar gösterilir.
    """
    d = df.tail(bars).copy()
    d["date"] = d.index
    d["yön"] = (d["Close"] >= d["Open"]).map({True: "yükseliş", False: "düşüş"})
    color = alt.Color(
        "yön:N",
        scale=alt.Scale(domain=["yükseliş", "düşüş"], range=["#16a34a", "#dc2626"]),
        legend=None,
    )
    base = alt.Chart(d).encode(
        x=alt.X("date:T", title=None),
        color=color,
    )
    # Fitil (High-Low)
    wick = base.mark_rule().encode(
        y=alt.Y("Low:Q", title="Fiyat (TRY)", scale=alt.Scale(zero=False)),
        y2="High:Q",
    )
    # Gövde (Open-Close)
    body = base.mark_bar(size=5).encode(y="Open:Q", y2="Close:Q")
    return (wick + body).properties(height=height)

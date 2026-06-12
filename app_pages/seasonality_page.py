"""📅 Sezonsallık ekranı — ay/gün bazlı tarihsel istatistikler (LLM yok).

"Bu hisse hangi aylarda tarihsel olarak güçlü?" sorusunun deterministik
cevabı: aylık ortalama getiri, pozitif-ay oranı ve haftanın günü etkisi.
Küçük örneklemin yanıltıcılığına karşı her satırda örnek sayısı gösterilir.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from tradingagents.analytics.composite import _fetch_daily
from tradingagents.analytics.seasonality import MONTH_NAMES_TR, compute_seasonality


def render() -> None:
    st.title("📅 Sezonsallık Analizi")
    st.caption("Aylık ve haftanın günü bazlı tarihsel getiri istatistikleri · "
               "Deterministik (LLM yok) · Yatırım tavsiyesi değildir.")

    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("Sembol", value="XU100.IS",
                               help="XU100.IS (endeks), THYAO.IS, GC=F (altın)…",
                               key="season_ticker").strip().upper()
    with c2:
        st.write(""); st.write("")
        run = st.button("📊 Hesapla", type="primary", use_container_width=True,
                        key="season_run")
    if not run:
        st.info("Sembol gir, **Hesapla**'ya bas. En az ~2 yıl veri gerekir; "
                "10 yıla kadar geçmiş kullanılır.")
        return

    with st.spinner(f"{ticker} geçmişi çekiliyor…"):
        df = _fetch_daily(ticker, period="10y")
        res = compute_seasonality(df) if df is not None else None
    if res is None or not res.ok:
        st.error((res.reason if res else "") or "Yeterli veri alınamadı.")
        return

    st.success(f"≈{res.years} yıllık günlük veriyle hesaplandı.")
    cur_month = pd.Timestamp.today().month

    monthly = res.monthly.reset_index(names="ay_no")
    monthly["şu an"] = monthly["ay_no"] == cur_month

    chart = alt.Chart(monthly).mark_bar().encode(
        x=alt.X("ay:N", sort=[MONTH_NAMES_TR[m] for m in range(1, 13)], title=None),
        y=alt.Y("ort_getiri_pct:Q", title="Ortalama aylık getiri (%)"),
        color=alt.condition(alt.datum["ort_getiri_pct"] > 0,
                            alt.value("#16a34a"), alt.value("#dc2626")),
        stroke=alt.condition(alt.datum["şu an"], alt.value("#1d4ed8"), alt.value(None)),
        strokeWidth=alt.condition(alt.datum["şu an"], alt.value(3), alt.value(0)),
        tooltip=["ay", "ort_getiri_pct", "medyan_pct", "pozitif_oran", "ornek"],
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Mavi çerçeve = içinde bulunduğumuz ay.")

    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.markdown("##### Aylık istatistikler")
        view = res.monthly.copy()
        view.columns = ["Ay", "Ort. getiri %", "Medyan %", "Pozitif oran %", "Örnek"]
        st.dataframe(view, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("##### Haftanın günü etkisi")
        dview = res.daily.copy()
        dview.columns = ["Gün", "Ort. getiri %", "Pozitif oran %", "Örnek"]
        st.dataframe(dview, use_container_width=True, hide_index=True)

    st.caption("⚠️ Sezonsallık zayıf bir sinyaldir: kompozit skorda yalnız %5 ağırlık taşır "
               "ve 5 yıldan az örneklemde tamamen yok sayılır. Tek başına işlem gerekçesi yapma.")

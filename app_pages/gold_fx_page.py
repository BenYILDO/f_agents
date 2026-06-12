"""🥇 Altın & Döviz ekranı — ons, gram altın, USDTRY, BIST karşılaştırması.

Türk yatırımcının üç referansı (BIST, dolar, altın) tek ekranda: TL bazında
getiri yarışı, XU100/gram-altın reel seviyesi ve TL stres göstergesi. Tamamı
deterministik (LLM yok); aynı blok Makro analiste de enjekte edilir.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from tradingagents.analytics.regime import fetch_try_stress
from tradingagents.dataflows.gold_fx import fetch_gold_fx_snapshot


def _normalized_chart(frames: dict) -> alt.Chart | None:
    """Seçili serileri 100 bazlı normalize edip tek grafikte çizer."""
    rows = []
    for label in ("BIST 100", "Gram Altın (TL)", "USD/TRY"):
        s = frames.get(label)
        if s is None or s.empty:
            continue
        norm = s / s.iloc[0] * 100
        rows.append(pd.DataFrame({"date": norm.index, "değer": norm.values,
                                  "varlık": label}))
    if not rows:
        return None
    data = pd.concat(rows, ignore_index=True)
    return alt.Chart(data).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("değer:Q", title="2 yıl önce = 100 (TL bazında)",
                scale=alt.Scale(zero=False)),
        color=alt.Color("varlık:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["BIST 100", "Gram Altın (TL)", "USD/TRY"],
                                        range=["#3b82f6", "#eab308", "#16a34a"])),
    ).properties(height=340)


def render() -> None:
    st.title("🥇 Altın & Döviz Görünümü")
    st.caption("Ons/gram altın · USDTRY · BIST'in TL bazlı getiri yarışı ve "
               "altın cinsinden reel seviyesi · Deterministik (LLM yok).")

    if not st.button("🔄 Güncel verileri çek", type="primary", key="gold_run"):
        st.info("**Güncel verileri çek**'e bas — ons altın, USDTRY, gram altın "
                "ve BIST 100 son 2 yıl günlük veriyle karşılaştırılır.")
        return

    with st.spinner("Altın/döviz/endeks serileri çekiliyor…"):
        res = fetch_gold_fx_snapshot()
        stress, stress_note = fetch_try_stress()

    if not res.ok:
        st.error("Altın/döviz verilerine şu an ulaşılamadı.")
        for sh in res.sources:
            st.caption(f"• {sh.name}: {sh.status} {sh.detail}")
        return

    cols = st.columns(len(res.snapshot) or 1)
    for col, (label, val) in zip(cols, res.snapshot.items()):
        col.metric(label, f"{val:,.2f}")

    if stress is not None:
        color = "#16a34a" if stress < 0.3 else ("#ea580c" if stress < 0.6 else "#dc2626")
        level = "düşük" if stress < 0.3 else ("orta" if stress < 0.6 else "YÜKSEK")
        st.markdown(
            f"<div style='padding:10px 16px;border-radius:10px;background:{color}1a;"
            f"border:1.5px solid {color};'><b style='color:{color};'>💵 TL Stres: "
            f"{stress:.2f}/1.00 ({level})</b> — {stress_note}</div>",
            unsafe_allow_html=True,
        )
        st.caption("TL stresi yükseldikçe teknik sinyallerin güveni otomatik kırpılır "
                   "(kompozit skor rejim çarpanı).")

    chart = _normalized_chart(res.frames)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    if res.table is not None and not res.table.empty:
        st.markdown("##### TL bazında getiri karşılaştırması (%)")
        st.dataframe(res.table, use_container_width=True, hide_index=True)
        st.caption("BIST satırı altın/dövizin belirgin gerisindeyse yerli tasarruf "
                   "akımı hisse aleyhine dönmüş demektir (makro analist de bunu okur).")

    ratio = res.frames.get("XU100/Gram Altın")
    if ratio is not None and not ratio.empty:
        st.markdown("##### XU100 / Gram Altın — BIST'in reel (altın) seviyesi")
        rdf = pd.DataFrame({"date": ratio.index, "oran": ratio.values})
        st.altair_chart(
            alt.Chart(rdf).mark_line(color="#8b5cf6").encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("oran:Q", scale=alt.Scale(zero=False), title="XU100 / gram altın"),
            ).properties(height=240),
            use_container_width=True,
        )
        st.caption("Oran tarihsel dibe yaklaştıkça BIST altına göre ucuz, tepeye "
                   "yaklaştıkça pahalı demektir.")

    with st.expander("🩺 Veri Sağlığı"):
        for sh in res.sources:
            st.caption(f"• {sh.name}: {sh.status}" + (f" — {sh.detail}" if sh.detail else ""))

"""🧾 Temel Skor ekranı — video rasyo puanlaması (AL/TUT/SAT) + Piotroski.

Birincil çıktı: videolardaki kesin eşiklere (FD/FAVÖK 5-7, Cari Oran 1.5-2.5,
Net Borç/FAVÖK <2, ROE>enflasyon) göre 0-100 rasyo puanı ve net karar.
İkincil: Piotroski-tarzı sağlamlık kriterleri. Tamamı deterministik (LLM yok).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tradingagents.analytics.fundamental_score import compute_fundamental_score
from tradingagents.analytics.ratio_score import DEFAULT_TR_INFLATION, compute_ratio_score

_VERDICT_STYLE = {
    "AL":       ("#16a34a", "🟢"),
    "TUT":      ("#ea580c", "🟠"),
    "SAT":      ("#dc2626", "🔴"),
    "VERİ YOK": ("#6b7280", "⚪"),
}


def render() -> None:
    st.title("🧾 Temel Skor — Rasyo Puanlaması")
    st.caption("Video kriterleriyle rasyo puanı (FD/FAVÖK · Cari Oran · Net Borç/FAVÖK · "
               "reel ROE · EFK büyümesi) → AL/TUT/SAT · Deterministik (LLM yok).")

    c1, c2, c3 = st.columns([2, 1.2, 1])
    with c1:
        ticker = st.text_input("Hisse (ticker)", value="THYAO.IS",
                               key="fund_ticker").strip().upper()
    with c2:
        inflation = st.number_input("Yıllık enflasyon (%)", min_value=0.0, max_value=200.0,
                                    value=float(DEFAULT_TR_INFLATION), step=1.0,
                                    help="Reel ROE/büyüme eşiği. TÜFE'yi gir.")
    with c3:
        st.write(""); st.write("")
        run = st.button("🧮 Skorla", type="primary", use_container_width=True,
                        key="fund_run")
    if not run:
        st.info("Hisse kodu + güncel enflasyon gir, **Skorla**'ya bas. Veriler yfinance "
                "yıllık finansal tablolarından gelir (anahtar gerekmez).")
        return

    with st.spinner(f"{ticker} finansalları çekiliyor…"):
        res = compute_ratio_score(ticker, inflation_pct=inflation)
        piotroski = compute_fundamental_score(ticker)
    if not res.ok:
        st.error(res.error or "Temel veri alınamadı.")
        return

    color, emoji = _VERDICT_STYLE.get(res.verdict, ("#6b7280", "⚪"))
    title = res.company or ticker
    fin_note = " · finansal şirket (kriterler uyarlandı)" if res.is_financial else ""
    st.markdown(
        f"<div style='padding:16px 20px;border-radius:12px;background:{color}1a;"
        f"border:2px solid {color};'>"
        f"<span style='font-size:13px;color:{color};font-weight:600;'>RASYO KARARI · {title}"
        + (f" · {res.sector}" if res.sector else "") + fin_note + "</span><br>"
        f"<span style='font-size:30px;font-weight:800;color:{color};'>{emoji} {res.verdict} "
        f"<span style='font-size:18px;font-weight:600;'>({res.score:.0f}/100)</span></span></div>",
        unsafe_allow_html=True,
    )
    st.caption("⚠️ Bu bir KALİTE/DEĞER kararıdır (ne alınmalı). Alım ZAMANLAMASI için "
               "🧰 Teknik veya 🎯 Birleşik Karar ekranına bak (video: önce ne, sonra ne zaman).")

    st.markdown("##### Kriter bantları")
    rows = []
    for crit in res.criteria:
        rows.append({
            "Kriter": crit.name,
            "Değer": f"{crit.value:.2f}" if isinstance(crit.value, (int, float)) else "—",
            "Bant / Değerlendirme": crit.band,
            "Puan": f"{crit.score:.2f}",
            "Ağırlık": crit.weight,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Enflasyon eşiği: %{res.inflation_pct:.0f} · puan = Σ(ağırlık×bant skoru) / "
               "mevcut ağırlık. Eksik veri kriteri paydaya girmez.")

    if piotroski.ok:
        with st.expander("📋 Piotroski-tarzı sağlamlık kriterleri (ikincil)"):
            st.caption(f"Skor: {piotroski.score}/{piotroski.max_score} → {piotroski.grade}")
            prows = []
            for name, (ok, desc) in piotroski.criteria.items():
                mark = "✅" if ok else ("❌" if ok is False else "➖")
                prows.append({"Kriter": name, "Durum": mark, "Açıklama": desc})
            st.dataframe(pd.DataFrame(prows), use_container_width=True, hide_index=True)

    st.caption("⚠️ Yatırım tavsiyesi değildir. yfinance BIST finansallarında bazı kalemleri "
               "eksik döndürebilir; eksikler skora dahil edilmez.")

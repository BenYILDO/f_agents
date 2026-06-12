"""🧾 Temel Skor ekranı — Piotroski-tarzı sağlamlık skoru + rasyolar (LLM yok).

yfinance finansal tablolarından deterministik temel skor: kârlılık, nakit
üretimi, kaldıraç yönü, likidite, marj ve verimlilik kriterleri. Aynı motor
AI Fundamentals analistine de brif olarak enjekte edilir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tradingagents.analytics.fundamental_score import compute_fundamental_score

_GRADE_STYLE = {
    "Sağlam":   ("#16a34a", "🟢"),
    "Orta":     ("#ea580c", "🟠"),
    "Zayıf":    ("#dc2626", "🔴"),
    "veri yok": ("#6b7280", "⚪"),
}


def render() -> None:
    st.title("🧾 Temel Analiz Skoru")
    st.caption("Piotroski-tarzı sağlamlık kriterleri + değerleme rasyoları · "
               "Deterministik (LLM yok) · Yatırım tavsiyesi değildir.")

    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("Hisse (ticker)", value="THYAO.IS",
                               key="fund_ticker").strip().upper()
    with c2:
        st.write(""); st.write("")
        run = st.button("🧮 Skorla", type="primary", use_container_width=True,
                        key="fund_run")
    if not run:
        st.info("Hisse kodu gir, **Skorla**'ya bas. Veriler yfinance yıllık "
                "finansal tablolarından gelir (anahtar gerekmez).")
        return

    with st.spinner(f"{ticker} finansalları çekiliyor…"):
        res = compute_fundamental_score(ticker)
    if not res.ok:
        st.error(res.error or "Temel veri alınamadı.")
        return

    color, emoji = _GRADE_STYLE.get(res.grade, ("#6b7280", "⚪"))
    title = res.company or ticker
    st.markdown(
        f"<div style='padding:16px 20px;border-radius:12px;background:{color}1a;"
        f"border:2px solid {color};'>"
        f"<span style='font-size:13px;color:{color};font-weight:600;'>TEMEL SAĞLAMLIK · {title}"
        + (f" · {res.sector}" if res.sector else "") + "</span><br>"
        f"<span style='font-size:30px;font-weight:800;color:{color};'>{emoji} {res.grade} "
        f"<span style='font-size:18px;font-weight:600;'>({res.score}/{res.max_score} kriter)</span></span></div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.markdown("##### Sağlamlık kriterleri")
        rows = []
        for name, (ok, desc) in res.criteria.items():
            mark = "✅" if ok else ("❌" if ok is False else "➖ veri yok")
            rows.append({"Kriter": name, "Durum": mark, "Açıklama": desc})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("➖ olan kriterler skor paydasına dahil edilmez (eksik veri ceza değildir).")
    with col_b:
        st.markdown("##### Değerleme rasyoları")
        ratio_rows = [{"Rasyo": k, "Değer": v if v is not None else "—"}
                      for k, v in res.ratios.items()]
        st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True, hide_index=True)
        st.caption("Rasyolar tek başına 'ucuz/pahalı' demek için yetmez — sektör "
                   "ortalamasıyla karşılaştır. Yüksek enflasyonda nominal büyüme "
                   "yanıltıcıdır; skor bu yüzden marj/kaldıraç YÖNÜNE bakar.")

    st.caption("⚠️ Yatırım tavsiyesi değildir. yfinance BIST finansallarında bazı "
               "kalemleri eksik döndürebilir; eksikler tabloda ➖ ile işaretlenir.")

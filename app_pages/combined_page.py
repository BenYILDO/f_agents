"""🎯 Birleşik Karar ekranı — temel (rasyo) + teknik (kompozit) → AL/SAT.

Video felsefesinin uygulaması: temel "ne alınır"ı, teknik "ne zaman"ı söyler.
Tek hisse için birleşik karar + gerekçe; ayrıca BIST evrenini birleşik skora
göre sıralayan bir tarayıcı. Deterministik (LLM yok).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from tradingagents.analytics.combined import combined_signal
from tradingagents.analytics.ratio_score import DEFAULT_TR_INFLATION
from tradingagents.strategy.dip_signal import BIST_POPULAR

_DECISION_STYLE = {
    "GÜÇLÜ AL": ("#15803d", "🟢"),
    "AL":       ("#16a34a", "🟢"),
    "TUT":      ("#6b7280", "⚪"),
    "SAT":      ("#dc2626", "🔴"),
    "KAÇIN":    ("#991b1b", "🔴"),
    "VERİ YOK": ("#6b7280", "⚪"),
}
_ORDER = {"GÜÇLÜ AL": 0, "AL": 1, "TUT": 2, "SAT": 3, "KAÇIN": 4, "VERİ YOK": 5}


def _single(ticker: str, inflation: float) -> None:
    with st.spinner(f"{ticker} — temel + teknik birleştiriliyor…"):
        res = combined_signal(ticker, inflation_pct=inflation)
    if not res.ok:
        st.error(res.error or "Analiz başarısız.")
        return

    color, emoji = _DECISION_STYLE.get(res.decision, ("#6b7280", "⚪"))
    st.markdown(
        f"<div style='padding:16px 20px;border-radius:12px;background:{color}1a;"
        f"border:2px solid {color};'>"
        f"<span style='font-size:13px;color:{color};font-weight:600;'>BİRLEŞİK KARAR · {ticker}</span><br>"
        f"<span style='font-size:30px;font-weight:800;color:{color};'>{emoji} {res.decision} "
        f"<span style='font-size:18px;font-weight:600;'>({res.combined_score:.0f}/100)</span></span></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        if res.fundamental:
            st.metric("Temel (rasyo)", res.fundamental.verdict,
                      f"{res.fundamental.score:.0f}/100")
        else:
            st.metric("Temel (rasyo)", "veri yok")
    with cols[1]:
        if res.technical:
            st.metric("Teknik (kompozit)", res.technical.verdict,
                      f"{res.technical.score:+.0f}/100")
        else:
            st.metric("Teknik (kompozit)", "veri yok")

    st.markdown("##### Karar gerekçesi")
    for r in res.rationale:
        st.markdown(f"- {r}")

    if res.technical and res.technical.warnings:
        for w in res.technical.warnings:
            st.warning(w)

    st.caption("⚠️ Yatırım tavsiyesi değildir. GÜÇLÜ AL = temel sağlam + teknik alım "
               "bölgesi + tuzak yok. Tuzak (aşırı coşku / düşen bıçak) aktifse alım engellenir.")


def _scan_one(ticker: str, inflation: float) -> dict:
    try:
        res = combined_signal(ticker, inflation_pct=inflation)
        if not res.ok:
            return {"ticker": ticker, "decision": "VERİ YOK", "score": None,
                    "temel": None, "teknik": None}
        return {
            "ticker": ticker, "decision": res.decision, "score": res.combined_score,
            "temel": res.fundamental.score if res.fundamental else None,
            "teknik": res.technical.score if res.technical else None,
        }
    except Exception:  # noqa: BLE001
        return {"ticker": ticker, "decision": "VERİ YOK", "score": None,
                "temel": None, "teknik": None}


def render() -> None:
    st.title("🎯 Birleşik Karar — Temel + Teknik")
    st.caption("Rasyo puanı (ne alınır) + kompozit teknik (ne zaman) → tek AL/SAT kararı · "
               "Tuzak filtreli · Deterministik (LLM yok) · Yatırım tavsiyesi değildir.")

    c1, c2, c3 = st.columns([2, 1.2, 1])
    with c1:
        ticker = st.text_input("Hisse (ticker)", value="THYAO.IS",
                               key="comb_ticker").strip().upper()
    with c2:
        inflation = st.number_input("Enflasyon (%)", min_value=0.0, max_value=200.0,
                                    value=float(DEFAULT_TR_INFLATION), step=1.0,
                                    key="comb_infl")
    with c3:
        st.write(""); st.write("")
        run = st.button("🎯 Karar Ver", type="primary", use_container_width=True,
                        key="comb_run")
    if run:
        _single(ticker, inflation)

    st.divider()
    st.subheader("🔎 BIST Tarayıcı — birleşik skora göre sıralı")
    if st.button(f"📡 {len(BIST_POPULAR)} BIST hissesini tara", key="comb_scan"):
        with st.spinner(f"{len(BIST_POPULAR)} hisse taranıyor (temel+teknik, biraz sürer)…"):
            with ThreadPoolExecutor(max_workers=8) as ex:
                rows = list(ex.map(lambda t: _scan_one(t, inflation), BIST_POPULAR))
        rows.sort(key=lambda r: (_ORDER.get(r["decision"], 9),
                                 -(r["score"] if r["score"] is not None else -1)))
        al = [r for r in rows if r["decision"] in ("GÜÇLÜ AL", "AL")]
        st.success(f"**{len(al)}** hisse AL/GÜÇLÜ AL bölgesinde"
                   + (": " + ", ".join(r["ticker"].replace(".IS", "") for r in al) if al else "."))
        df = pd.DataFrame(rows)
        df["Karar"] = df["decision"].map(lambda d: f"{_DECISION_STYLE.get(d, ('', '⚪'))[1]} {d}")
        view = df[["ticker", "Karar", "score", "temel", "teknik"]].copy()
        view.columns = ["Hisse", "Karar", "Birleşik", "Temel", "Teknik"]
        view["Hisse"] = view["Hisse"].str.replace(".IS", "", regex=False)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("Birleşik = 0.55×teknik + 0.45×temel (0-100). Karar kural tabanlıdır "
                   "(saf ortalama değil): tuzak filtreleri alımı bloklar.")

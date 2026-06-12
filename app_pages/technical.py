"""🧰 Teknik Analiz ekranı — AI'dan tamamen bağımsız, deterministik.

Kompozit skor paneli + grafik formasyonları + mum formasyonları + destek/
direnç + indikatör grafikleri. Tüm hesaplar :mod:`tradingagents.analytics`
motorundan gelir; LLM yok, API anahtarı yok, maliyet yok.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from tradingagents.analytics.composite import WEIGHTS, _fetch_daily, compute_composite
from tradingagents.analytics.indicators import add_core_indicators

_VERDICT_STYLE = {
    "GÜÇLÜ AL":  ("#15803d", "🟢"),
    "AL":        ("#16a34a", "🟢"),
    "NÖTR":      ("#6b7280", "⚪"),
    "SAT":       ("#dc2626", "🔴"),
    "GÜÇLÜ SAT": ("#991b1b", "🔴"),
}
_COMPONENT_TR = {
    "trend": "Trend (SMA hiyerarşisi)",
    "momentum": "Momentum (RSI/MACD/Stokastik)",
    "pattern": "Grafik formasyonları",
    "dip": "Dip-Al stratejisi",
    "candle": "Mum formasyonları",
    "sr": "Destek/Direnç konumu",
    "seasonality": "Sezonsallık",
}


def _price_chart(df: pd.DataFrame, res) -> alt.Chart:
    ind = add_core_indicators(df).tail(180).copy()
    ind["date"] = ind.index
    base = alt.Chart(ind).encode(x=alt.X("date:T", title=None))
    layers = [
        base.mark_line(color="#e5e7eb", strokeDash=[2, 3]).encode(y="bb_upper:Q"),
        base.mark_line(color="#e5e7eb", strokeDash=[2, 3]).encode(y="bb_lower:Q"),
        base.mark_line(color="#f59e0b", strokeDash=[4, 3]).encode(y="sma50:Q"),
        base.mark_line(color="#8b5cf6", strokeDash=[4, 3]).encode(y="sma200:Q"),
        base.mark_line(color="#3b82f6").encode(
            y=alt.Y("Close:Q", title="Fiyat (TRY)", scale=alt.Scale(zero=False))),
    ]
    # Destek/direnç yatay çizgileri
    rules = pd.DataFrame(
        [{"y": lv.price, "tip": "destek"} for lv in res.supports]
        + [{"y": lv.price, "tip": "direnç"} for lv in res.resistances]
    )
    if not rules.empty:
        layers.append(alt.Chart(rules).mark_rule(strokeDash=[6, 4], opacity=0.6).encode(
            y="y:Q",
            color=alt.Color("tip:N", scale=alt.Scale(domain=["destek", "direnç"],
                                                     range=["#16a34a", "#dc2626"]),
                            legend=alt.Legend(title=None, orient="top"))))
    return alt.layer(*layers).properties(height=360)


def render() -> None:
    st.title("🧰 Teknik Analiz — Deterministik Motor")
    st.caption("Formasyon tespiti · destek/direnç · sezonsallık · rejim · kompozit skor — "
               "tamamı yerel hesap (LLM yok) · Yatırım tavsiyesi değildir.")

    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("Sembol", value="THYAO.IS",
                               help="BIST: THYAO.IS · Altın: GC=F · Dolar: TRY=X · Endeks: XU100.IS",
                               key="ta_ticker").strip().upper()
    with c2:
        st.write(""); st.write("")
        run = st.button("🔬 Analiz Et", type="primary", use_container_width=True,
                        key="ta_run")
    if not run:
        st.info("Sembol gir ve **Analiz Et**'e bas. BIST hisseleri, XU100, altın (GC=F) "
                "ve dövizle (TRY=X) çalışır.")
        return

    with st.spinner(f"{ticker} hesaplanıyor…"):
        # Veriyi bir kez çek; hem kompozit skor hem grafik aynı barları kullansın
        df = _fetch_daily(ticker)
        res = compute_composite(ticker, df)
    if not res.ok:
        st.error(res.error or "Analiz başarısız.")
        return

    color, emoji = _VERDICT_STYLE.get(res.verdict, ("#6b7280", "⚪"))
    st.markdown(
        f"<div style='padding:16px 20px;border-radius:12px;background:{color}1a;"
        f"border:2px solid {color};'>"
        f"<span style='font-size:13px;color:{color};font-weight:600;'>KOMPOZİT TEKNİK SKOR · {ticker}</span><br>"
        f"<span style='font-size:30px;font-weight:800;color:{color};'>{emoji} {res.verdict} "
        f"<span style='font-size:18px;font-weight:600;'>({res.score:+.0f}/100 · güven: {res.confidence})</span></span>"
        f"<br><span style='color:#6b7280;font-size:13px;'>kapanış {res.last_close} · "
        f"ham skor {res.raw_score:+.0f} → rejim çarpanı sonrası {res.score:+.0f}</span></div>",
        unsafe_allow_html=True,
    )
    if res.regime is not None:
        st.caption(f"🌡️ Rejim: {res.regime.summary}")
        if res.regime.try_note:
            st.caption(f"💵 {res.regime.try_note}")

    # Bileşen kırılımı
    st.markdown("##### Skor bileşenleri")
    comp_rows = [{
        "Bileşen": _COMPONENT_TR.get(k, k),
        "Ağırlık": WEIGHTS[k],
        "Skor [-1,+1]": res.components[k],
        "Açıklama": res.details[k],
    } for k in WEIGHTS]
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    if df is not None:
        st.altair_chart(_price_chart(df, res), use_container_width=True)
        st.caption("Mavi=kapanış · turuncu=SMA50 · mor=SMA200 · gri=Bollinger · "
                   "yeşil/kırmızı kesikli=destek/direnç")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 📐 Grafik formasyonları")
        if res.patterns:
            st.dataframe(pd.DataFrame([{
                "Formasyon": h.name, "Yön": h.direction,
                "Durum": "✅ teyitli" if h.confirmed else "⏳ oluşum",
                "Aralık": f"{h.start} → {h.end}", "Not": h.note,
            } for h in res.patterns]), use_container_width=True, hide_index=True)
        else:
            st.caption("Son 60 barda aktif formasyon tespit edilmedi.")

        st.markdown("##### 🕯️ Mum formasyonları (son 10 bar)")
        if res.candles:
            st.dataframe(pd.DataFrame([{
                "Tarih": h.date, "Formasyon": h.name,
                "Yön": h.direction, "Güç": "★" * h.strength,
            } for h in res.candles]), use_container_width=True, hide_index=True)
        else:
            st.caption("Son barlarda mum formasyonu yok.")

    with col_b:
        st.markdown("##### 🧱 Destek / Direnç")
        sr_rows = ([{"Seviye": lv.price, "Tip": "🟢 destek", "Kaynak": lv.source,
                     "Not": lv.label} for lv in res.supports]
                   + [{"Seviye": lv.price, "Tip": "🔴 direnç", "Kaynak": lv.source,
                       "Not": lv.label} for lv in res.resistances])
        if sr_rows:
            st.dataframe(pd.DataFrame(sr_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Seviye hesaplanamadı.")
        st.caption("pivot = geçmiş dokunuş kümeleri · fibonacci = son 6 ay salınımı · "
                   "klasik = günlük pivot noktası")

    st.caption("⚠️ Yatırım tavsiyesi değildir. Skorlar geçmiş veriyle hesaplanır; "
               "yüksek volatilite/TL stresi dönemlerinde güven otomatik düşürülür.")

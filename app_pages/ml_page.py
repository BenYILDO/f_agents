"""🤖 ML Sinyal ekranı — hissenin geçmişinden eğitilen yön-tahmin modeli.

Deterministik analiz motorunun özelliklerinden bir gradient-boosting
sınıflandırıcı eğitir, zaman-serisi backtest ile isabetini ölçer ve güncel bar
için yukarı olasılığını + sinyal üretir. Model talep anında eğitilir (önceden
paketlenmiş ağırlık yok). Skill = backtest doğruluğu − taban (çoğunluk sınıfı):
pozitif skill, modelin sadece trendi ezberlemenin ötesinde değer kattığını
gösterir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tradingagents.analytics.composite import _fetch_daily
from tradingagents.ml.model import train_signal_model

_SIGNAL_STYLE = {
    "AL":       ("#16a34a", "🟢"),
    "TUT":      ("#6b7280", "⚪"),
    "SAT":      ("#dc2626", "🔴"),
    "VERİ YOK": ("#6b7280", "⚪"),
}


def render() -> None:
    st.title("🤖 ML Sinyal — Yön Tahmini")
    st.caption("Hissenin kendi geçmişinden eğitilen sınıflandırıcı + zaman-serisi backtest · "
               "Eğitim talep anında yapılır · Yatırım tavsiyesi değildir.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        ticker = st.text_input("Sembol", value="THYAO.IS", key="ml_ticker").strip().upper()
    with c2:
        horizon = st.number_input("Ufuk (gün)", min_value=1, max_value=60, value=10, step=1,
                                  help="Kaç gün sonrasının yönü tahmin edilsin.")
    with c3:
        threshold = st.number_input("Eşik (%)", min_value=0.0, max_value=20.0, value=0.0,
                                    step=0.5, help="Bu getirinin üstü 'yukarı' sayılır.")
    with c4:
        st.write(""); st.write("")
        run = st.button("🧠 Eğit & Tahmin", type="primary", use_container_width=True,
                        key="ml_run")
    if not run:
        st.info("Sembol seç, **Eğit & Tahmin**'e bas. ~3+ yıl günlük veri gerekir; "
                "model birkaç saniyede eğitilir. Skill > 0 ise model trend tabanından "
                "daha isabetli demektir.")
        return

    with st.spinner(f"{ticker} verisi çekiliyor ve model eğitiliyor…"):
        df = _fetch_daily(ticker, period="10y")
        res = train_signal_model(df, ticker=ticker, horizon=int(horizon),
                                 threshold=threshold / 100.0) if df is not None else None

    if res is None or not res.ok:
        st.error((res.reason if res else "") or "Yeterli veri alınamadı.")
        return

    color, emoji = _SIGNAL_STYLE.get(res.signal, ("#6b7280", "⚪"))
    prob_txt = f"%{res.prob_up * 100:.0f}" if res.prob_up is not None else "—"
    st.markdown(
        f"<div style='padding:16px 20px;border-radius:12px;background:{color}1a;"
        f"border:2px solid {color};'>"
        f"<span style='font-size:13px;color:{color};font-weight:600;'>ML SİNYAL · {ticker} · "
        f"{res.horizon} gün</span><br>"
        f"<span style='font-size:30px;font-weight:800;color:{color};'>{emoji} {res.signal} "
        f"<span style='font-size:18px;font-weight:600;'>(yukarı olasılığı {prob_txt})</span></span></div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Backtest doğruluk", f"%{res.accuracy * 100:.1f}")
    m2.metric("Taban (çoğunluk)", f"%{res.baseline * 100:.1f}")
    m3.metric("Skill (ek isabet)", f"{res.skill * 100:+.1f} puan",
              delta_color="normal" if res.skill > 0 else "inverse")
    m4.metric("AUC", f"{res.roc_auc:.2f}" if res.roc_auc is not None else "—")

    if res.skill <= 0:
        st.warning("⚠️ Skill ≤ 0: model bu hisse/ufukta tabandan (hep çoğunluk sınıfı demekten) "
                   "daha iyi değil. Sinyale düşük güvenle yaklaş.")
    else:
        st.caption(f"Model, tabana göre {res.skill * 100:+.1f} puan ek isabet sağlıyor "
                   f"({res.n_samples} örnek, out-of-sample backtest).")

    st.markdown("##### Özellik önemleri (modelin en çok baktığı sinyaller)")
    imp = pd.DataFrame(
        [{"Özellik": k, "Önem": round(v, 3)} for k, v in list(res.feature_importance.items())[:10]]
    )
    st.bar_chart(imp.set_index("Özellik"))

    st.caption("⚠️ Yatırım tavsiyesi değildir. Geçmiş performans geleceği garanti etmez; "
               "ML sinyali diğer ekranlarla (Teknik, Temel, Birleşik) birlikte okunmalıdır.")

"""BIST TradingAgents — Streamlit arayüzü.

Tek dosyalık, tek kullanıcılık bir analiz arayüzü. Bir BIST hissesi (örn.
THYAO.IS) ve tarih girip çok-ajanlı analiz hattını (`TradingAgentsGraph.propagate`)
çalıştırır; sonucu 5 kademeli rating + ajan raporları olarak gösterir.

Çalıştırma:  streamlit run streamlit_app.py
Anahtar:     .env içindeki OPENAI_API_KEY (tradingagents import'unda load_dotenv ile yüklenir)
             ya da kenar çubuğundan girilir.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import streamlit as st

# tradingagents import'u .env'i otomatik yükler (tradingagents/__init__.py)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.utils.rating import parse_rating, RATINGS_5_TIER

st.set_page_config(page_title="BIST TradingAgents", page_icon="📈", layout="wide")

# 5 kademeli rating için renk/emoji eşlemesi (en boğa → en ayı)
_RATING_STYLE = {
    "Buy":        ("#16a34a", "🟢", "Güçlü Al"),
    "Overweight": ("#65a30d", "🟢", "Ağırlık Artır"),
    "Hold":       ("#6b7280", "⚪", "Tut"),
    "Underweight":("#ea580c", "🟠", "Ağırlık Azalt"),
    "Sell":       ("#dc2626", "🔴", "Sat"),
}

_DEPTH = {"Sığ (hızlı)": 1, "Orta": 3, "Derin (kapsamlı)": 5}
_QUICK_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1"]
_DEEP_MODELS = ["gpt-5.4", "gpt-5.5", "gpt-5.2"]
_ANALYSTS = {
    "Teknik (Market)": "market",
    "Duygu/Haber-TR (Sentiment)": "social",
    "Haber (News)": "news",
    "Temel (Fundamentals)": "fundamentals",
}


def _last_weekday() -> date:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Cmt, 6=Paz
        d -= timedelta(days=1)
    return d


def _build_report_markdown(state: dict, ticker: str, trade_date: str) -> str:
    parts = [f"# {ticker} — Analiz Raporu ({trade_date})\n"]
    sections = [
        ("Nihai Karar", state.get("final_trade_decision")),
        ("Temel (Fundamentals)", state.get("fundamentals_report")),
        ("Duygu / TR Haber & KAP (Sentiment)", state.get("sentiment_report")),
        ("Teknik (Market)", state.get("market_report")),
        ("Haber (News)", state.get("news_report")),
        ("Trader Planı", state.get("trader_investment_plan")),
    ]
    for title, body in sections:
        if body:
            parts.append(f"## {title}\n\n{body}\n")
    return "\n".join(parts)


# ── Kenar çubuğu ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ayarlar")

    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        st.success("OpenAI anahtarı .env'den yüklendi ✓")
    else:
        manual_key = st.text_input("OpenAI API Key", type="password",
                                   help="sk-... — sadece bu oturumda kullanılır")
        if manual_key:
            os.environ["OPENAI_API_KEY"] = manual_key

    deep_model = st.selectbox("Derin düşünme modeli", _DEEP_MODELS, index=0,
                              help="Araştırma/karar ajanları. Maliyet/kalite dengesi.")
    quick_model = st.selectbox("Hızlı model", _QUICK_MODELS, index=0)
    depth_label = st.select_slider("Araştırma derinliği", list(_DEPTH.keys()),
                                   value="Sığ (hızlı)")
    analyst_labels = st.multiselect("Analistler", list(_ANALYSTS.keys()),
                                    default=list(_ANALYSTS.keys()))
    st.caption("💡 Her analiz OpenAI kredisi harcar. Derinlik arttıkça maliyet/süre artar.")


# ── Ana panel ───────────────────────────────────────────────────────────────
st.title("📈 BIST TradingAgents")
st.caption("Borsa İstanbul hisseleri için çok-ajanlı yapay zeka analizi · "
           "Yatırım tavsiyesi değildir.")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    ticker = st.text_input("Hisse (ticker)", value="THYAO.IS",
                           help="BIST için .IS ekle: THYAO.IS, GARAN.IS, ASELS.IS").strip().upper()
with c2:
    trade_date = st.date_input("Analiz tarihi", value=_last_weekday())
with c3:
    st.write("")
    st.write("")
    run = st.button("🚀 Analiz Et", type="primary", use_container_width=True)

if run:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API anahtarı yok. Kenar çubuğundan gir ya da .env'e ekle.")
        st.stop()
    if not ticker:
        st.error("Bir hisse kodu gir (örn. THYAO.IS).")
        st.stop()

    selected = [_ANALYSTS[l] for l in analyst_labels] or ["market", "social", "news", "fundamentals"]
    rounds = _DEPTH[depth_label]
    config = {
        **DEFAULT_CONFIG,
        "llm_provider": "openai",
        "deep_think_llm": deep_model,
        "quick_think_llm": quick_model,
        "max_debate_rounds": rounds,
        "max_risk_discuss_rounds": rounds,
    }
    asset_type = "crypto" if ticker.endswith(("-USD", "-USDT", "-USDC")) else "stock"
    date_str = trade_date.strftime("%Y-%m-%d")

    try:
        with st.status(f"**{ticker}** analiz ediliyor — birkaç dakika sürebilir…",
                       expanded=True) as status:
            st.write("Graph kuruluyor, modeller başlatılıyor…")
            ta = TradingAgentsGraph(selected_analysts=selected, debug=False, config=config)
            benchmark = ta._resolve_benchmark(ticker)
            st.write(f"Benchmark (alpha için): **{benchmark}**")
            st.write("Ajanlar çalışıyor: Teknik → Duygu/TR-Haber → Haber → Temel → "
                     "Araştırma tartışması → Trader → Risk → Karar…")
            final_state, decision = ta.propagate(ticker, date_str, asset_type=asset_type)
            status.update(label=f"{ticker} analizi tamamlandı ✓", state="complete", expanded=False)
    except Exception as e:  # noqa: BLE001
        import traceback as _tb
        st.error(f"Analiz sırasında hata: {type(e).__name__}: {e}")
        with st.expander("Teknik ayrıntı (traceback)"):
            st.code(_tb.format_exc())
        st.stop()

    # ── Sonuçlar ────────────────────────────────────────────────────────────
    rating = parse_rating(final_state.get("final_trade_decision", ""))
    color, emoji, tr = _RATING_STYLE.get(rating, ("#6b7280", "⚪", rating))

    st.markdown(
        f"<div style='padding:16px 20px;border-radius:12px;background:{color}1a;"
        f"border:2px solid {color};'>"
        f"<span style='font-size:13px;color:{color};font-weight:600;'>NİHAİ KARAR · {ticker}</span><br>"
        f"<span style='font-size:30px;font-weight:800;color:{color};'>{emoji} {rating} "
        f"<span style='font-size:18px;font-weight:600;'>({tr})</span></span></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Alpha benchmark'ı: {benchmark} · Fiyatlar TRY · Karar metni 5 kademeli "
               f"ölçek ({', '.join(RATINGS_5_TIER)}) kullanır.")

    st.download_button(
        "📥 Tüm raporu indir (.md)",
        _build_report_markdown(final_state, ticker, date_str),
        file_name=f"{ticker}_{date_str}_analiz.md",
        mime="text/markdown",
    )

    st.subheader("Karar gerekçesi")
    st.markdown(final_state.get("final_trade_decision") or "_(boş)_")

    tabs = st.tabs(["💬 Duygu / TR-Haber & KAP", "📊 Temel", "📈 Teknik",
                    "📰 Haber", "🧠 Araştırma tartışması", "💼 Trader planı"])
    with tabs[0]:
        st.markdown(final_state.get("sentiment_report") or "_(seçili değil)_")
    with tabs[1]:
        st.markdown(final_state.get("fundamentals_report") or "_(seçili değil)_")
    with tabs[2]:
        st.markdown(final_state.get("market_report") or "_(seçili değil)_")
    with tabs[3]:
        st.markdown(final_state.get("news_report") or "_(seçili değil)_")
    with tabs[4]:
        deb = final_state.get("investment_debate_state", {}) or {}
        if deb.get("bull_history"):
            st.markdown("#### 🐂 Boğa")
            st.markdown(deb["bull_history"])
        if deb.get("bear_history"):
            st.markdown("#### 🐻 Ayı")
            st.markdown(deb["bear_history"])
        if deb.get("judge_decision"):
            st.markdown("#### ⚖️ Araştırma Yöneticisi")
            st.markdown(deb["judge_decision"])
        if not deb:
            st.markdown("_(yok)_")
    with tabs[5]:
        st.markdown(final_state.get("trader_investment_plan") or "_(yok)_")
else:
    st.info("Soldan ayarları seç, bir hisse kodu (örn. **THYAO.IS**) ve tarih gir, "
            "**Analiz Et**'e bas. İlk çalıştırma birkaç dakika sürebilir.")

"""BIST TradingAgents — Streamlit arayüzü.

Ekranlar:
  🤖 AI Analizi        — çok-ajanlı LLM analizi (TradingAgentsGraph.propagate)
  🧰 Teknik Analiz     — formasyon/destek-direnç/rejim + kompozit skor, LLM yok
  📐 Dip-Al Stratejisi — video.md teknik stratejisi (SMI+VWMA+Bollinger), LLM yok
  📅 Sezonsallık       — ay/gün bazlı tarihsel istatistikler, LLM yok
  🥇 Altın & Döviz     — ons/gram altın, USDTRY, TL stres göstergesi, LLM yok
  🧾 Temel Skor        — Piotroski-tarzı sağlamlık skoru + rasyolar, LLM yok

Çalıştırma:  streamlit run streamlit_app.py
Anahtar:     .env içindeki OPENAI_API_KEY (yalnız AI Analizi ekranı için gerekir)
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

# tradingagents import'u .env'i yükler + bozuk SSL_CERT_FILE'ı onarır (__init__.py)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.utils.rating import parse_rating, RATINGS_5_TIER
from tradingagents.dataflows.symbol_utils import is_bist_ticker
from tradingagents.storage import portfolio
from tradingagents.storage.supabase_client import SupabaseError, SupabaseREST, is_configured
from tradingagents.strategy.dip_signal import (
    analyze as strategy_analyze,
    scan as strategy_scan,
    INTERVALS,
    BIST_POPULAR,
)
from app_pages import (
    arena_page,
    combined_page,
    fundamental_page,
    gold_fx_page,
    ml_page,
    portfolio_page,
    scanner_page,
    seasonality_page,
    technical,
)

st.set_page_config(page_title="BIST TradingAgents", page_icon="📈", layout="wide")

_RATING_STYLE = {
    "Buy":        ("#16a34a", "🟢", "Güçlü Al"),
    "Overweight": ("#65a30d", "🟢", "Ağırlık Artır"),
    "Hold":       ("#6b7280", "⚪", "Tut"),
    "Underweight":("#ea580c", "🟠", "Ağırlık Azalt"),
    "Sell":       ("#dc2626", "🔴", "Sat"),
}
_STATUS_STYLE = {
    "AL BÖLGESİ":  ("#16a34a", "🟢"),
    "SAT UYARISI": ("#dc2626", "🔴"),
    "NÖTR":        ("#6b7280", "⚪"),
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
_LANGUAGES = ["Turkish", "English"]


def _last_weekday() -> date:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _build_report_markdown(state: dict, ticker: str, trade_date: str) -> str:
    parts = [f"# {ticker} — Analiz Raporu ({trade_date})\n"]
    for title, body in [
        ("Nihai Karar", state.get("final_trade_decision")),
        ("Temel (Fundamentals)", state.get("fundamentals_report")),
        ("Duygu / TR Haber & KAP (Sentiment)", state.get("sentiment_report")),
        ("Teknik (Market)", state.get("market_report")),
        ("Haber (News)", state.get("news_report")),
        ("Makro (TR — TCMB/faiz, enflasyon, kur)", state.get("macro_report")),
        ("Siyaset/Jeopolitik (TR — risk primi)", state.get("geopolitics_report")),
        ("Trader Planı", state.get("trader_investment_plan")),
    ]:
        if body:
            parts.append(f"## {title}\n\n{body}\n")
    return "\n".join(parts)


# ── Kenar çubuğu ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ayarlar")
    mode = st.radio("Ekran", ["💼 Portföyüm", "🏟️ Paper Arena", "📡 BIST 30 Tarayıcı",
                              "🤖 AI Analizi", "🎯 Birleşik Karar", "🧰 Teknik Analiz",
                              "🧾 Temel Skor", "🔮 ML Sinyal", "📐 Dip-Al Stratejisi",
                              "📅 Sezonsallık", "🥇 Altın & Döviz"])

    env_key = os.environ.get("OPENAI_API_KEY")
    if mode == "🤖 AI Analizi":
        if env_key:
            st.success("OpenAI anahtarı .env'den yüklendi ✓")
        else:
            manual_key = st.text_input("OpenAI API Key", type="password",
                                       help="sk-... — sadece bu oturumda kullanılır")
            if manual_key:
                os.environ["OPENAI_API_KEY"] = manual_key

        language = st.selectbox("Rapor dili", _LANGUAGES, index=0)
        deep_model = st.selectbox("Derin düşünme modeli", _DEEP_MODELS, index=0)
        quick_model = st.selectbox("Hızlı model", _QUICK_MODELS, index=0)
        depth_label = st.select_slider("Araştırma derinliği", list(_DEPTH.keys()),
                                       value="Sığ (hızlı)")
        analyst_labels = st.multiselect("Analistler", list(_ANALYSTS.keys()),
                                        default=list(_ANALYSTS.keys()))
        st.caption("💡 Her analiz OpenAI kredisi harcar. Derinlik arttıkça maliyet/süre artar.")
    elif mode == "💼 Portföyüm":
        if is_configured():
            st.success("Supabase anahtarları yüklü ✓")
        else:
            st.warning("Supabase bağlı değil — kurulum: docs/SUPABASE_SETUP.md")
        st.caption("LLM yok · veriler Supabase'de saklanır · saat başı otomatik analiz.")
    else:
        st.caption("🔓 Bu ekran tamamen yereldir (LLM yok, ücretsiz, anlıktır).")


# ════════════════════════════════════════════════════════════════════════════
# 🤖 AI ANALİZİ EKRANI
# ════════════════════════════════════════════════════════════════════════════
def _ai_config() -> dict:
    """Kenar çubuğu ayarlarından (globaller) AI graph yapılandırmasını üretir."""
    return {
        **DEFAULT_CONFIG,
        "llm_provider": "openai",
        "deep_think_llm": deep_model,
        "quick_think_llm": quick_model,
        "max_debate_rounds": _DEPTH[depth_label],
        "max_risk_discuss_rounds": _DEPTH[depth_label],
        "output_language": language,
    }


def _ai_selected(ticker: str) -> list[str]:
    """Seçili analistler + BIST (.IS) için otomatik makro/jeopolitik analist."""
    selected = [_ANALYSTS[l] for l in analyst_labels] or list(_ANALYSTS.values())
    if is_bist_ticker(ticker):
        for auto_key in ("macro", "geopolitics"):
            if auto_key not in selected:
                selected.append(auto_key)
    return selected


def _ai_run_one(ticker: str, date_str: str):
    """Tek hisse için graph kurup propagate eder → (final_state, benchmark)."""
    config = _ai_config()
    ta = TradingAgentsGraph(selected_analysts=_ai_selected(ticker), debug=False, config=config)
    benchmark = ta._resolve_benchmark(ticker)
    asset_type = "crypto" if ticker.endswith(("-USD", "-USDT", "-USDC")) else "stock"
    final_state, _ = ta.propagate(ticker, date_str, asset_type=asset_type)
    return final_state, benchmark


def render_ai_basket() -> None:
    """🧺 Sepetime özgü AI analizi — portföyden seç, toplu çok-ajan analizi.

    Tekli AI akışından bağımsız; aynı ekranın altında durur. Her seçili hisse
    ayrı bir çok-ajan koşusudur (kredi + süre harcar). Sonuçlar opsiyonel olarak
    ``ai_runs`` tablosuna yazılır.
    """
    st.subheader("🧺 Sepetime özgü AI analizi")
    st.caption("Portföyündeki hisseleri seç, hepsine birden çok-ajan AI analizi "
               "çalıştır. Her hisse ayrı koşu → kredi/süre harcar.")

    if not is_configured():
        st.info("Supabase bağlı değil — sepet portföyden beslenir. 💼 Portföyüm'ü "
                "bağlayınca burada hisselerini seçebilirsin. (Tekli analiz yukarıda çalışır.)")
        return
    try:
        holdings = portfolio.list_holdings()
    except SupabaseError as e:
        st.error(f"Portföy okunamadı: {e}")
        return
    tickers = sorted({h["ticker"].upper() for h in holdings})
    if not tickers:
        st.info("Portföyün boş. 💼 Portföyüm ekranından hisse ekle, sonra burada seç.")
        return

    sel = st.multiselect("Sepetten hisse seç", tickers,
                         format_func=lambda t: t.replace(".IS", ""))
    b1, b2 = st.columns([1, 2])
    with b1:
        basket_date = st.date_input("Analiz tarihi", value=_last_weekday(),
                                    key="basket_date")
    with b2:
        st.write(""); st.write("")
        go = st.button("🚀 Sepeti AI ile analiz et", type="primary",
                       disabled=not sel, use_container_width=True, key="basket_run")
    if sel:
        st.caption(f"{len(sel)} hisse seçili · her biri birkaç dakika + OpenAI kredisi.")
    if not go:
        return
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API anahtarı yok. Kenar çubuğundan gir ya da .env'e ekle.")
        return

    date_str = basket_date.strftime("%Y-%m-%d")
    results = []
    for tk in sel:
        with st.status(f"**{tk}** analiz ediliyor…", expanded=False) as status:
            try:
                final_state, benchmark = _ai_run_one(tk, date_str)
                status.update(label=f"{tk} tamamlandı ✓", state="complete")
            except Exception as e:  # noqa: BLE001
                status.update(label=f"{tk} — hata", state="error")
                st.error(f"{tk}: {type(e).__name__}: {e}")
                continue
        rating = parse_rating(final_state.get("final_trade_decision", ""))
        results.append((tk, rating, final_state, benchmark, date_str))
        try:  # opsiyonel kalıcılık — başarısızlık analizi engellemesin
            SupabaseREST().insert("ai_runs", {
                "ticker": tk, "rating": rating, "language": language,
                "report_md": _build_report_markdown(final_state, tk, date_str),
                "meta": {"benchmark": benchmark, "source": "basket"},
            })
        except SupabaseError:
            pass

    if not results:
        return
    st.markdown("##### Sepet özeti")
    st.dataframe(pd.DataFrame([{
        "Hisse": tk.replace(".IS", ""),
        "Karar": _RATING_STYLE.get(r, ("", "", r))[2],
        "Rating": r,
    } for tk, r, *_ in results]), use_container_width=True, hide_index=True)

    for tk, r, state, benchmark, dstr in results:
        color, emoji, tr = _RATING_STYLE.get(r, ("#6b7280", "⚪", r))
        with st.expander(f"{emoji} {tk.replace('.IS', '')} — {r} ({tr})"):
            st.download_button("📥 Rapor (.md)",
                               _build_report_markdown(state, tk, dstr),
                               file_name=f"{tk}_{dstr}_analiz.md",
                               mime="text/markdown", key=f"dl_basket_{tk}")
            st.markdown(state.get("final_trade_decision") or "_(boş)_")


def render_ai_screen():
    st.title("📈 BIST TradingAgents — AI Analizi")
    st.caption("Borsa İstanbul hisseleri için çok-ajanlı yapay zeka analizi · "
               "Yatırım tavsiyesi değildir.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ticker = st.text_input("Hisse (ticker)", value="THYAO.IS",
                               help="BIST için .IS ekle: THYAO.IS, GARAN.IS").strip().upper()
    with c2:
        trade_date = st.date_input("Analiz tarihi", value=_last_weekday())
    with c3:
        st.write(""); st.write("")
        run = st.button("🚀 Analiz Et", type="primary", use_container_width=True)

    if run:
        _do_single_run(ticker, trade_date)
    else:
        st.info("Soldan ayarları seç, bir hisse kodu (örn. **THYAO.IS**) ve tarih gir, "
                "**Analiz Et**'e bas. İlk çalıştırma birkaç dakika sürebilir.")

    st.divider()
    render_ai_basket()


def _do_single_run(ticker, trade_date):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API anahtarı yok. Kenar çubuğundan gir ya da .env'e ekle.")
        return
    if not ticker:
        st.error("Bir hisse kodu gir (örn. THYAO.IS).")
        return

    selected = [_ANALYSTS[l] for l in analyst_labels] or list(_ANALYSTS.values())
    # Türk usulü: BIST (.IS) hisselerinde makro rejim analistini (TCMB/faiz,
    # enflasyon, kur, ülke riski) ve jeopolitik/siyaset analistini (siyasi şok
    # takvimi + olay etüdü) otomatik ekle. BIST dışı enstrümanlarda küresel
    # makroyu/jeopolitiği zaten Haber Analisti karşılıyor.
    if is_bist_ticker(ticker):
        for auto_key in ("macro", "geopolitics"):
            if auto_key not in selected:
                selected.append(auto_key)
    config = {
        **DEFAULT_CONFIG,
        "llm_provider": "openai",
        "deep_think_llm": deep_model,
        "quick_think_llm": quick_model,
        "max_debate_rounds": _DEPTH[depth_label],
        "max_risk_discuss_rounds": _DEPTH[depth_label],
        "output_language": language,
    }
    asset_type = "crypto" if ticker.endswith(("-USD", "-USDT", "-USDC")) else "stock"
    date_str = trade_date.strftime("%Y-%m-%d")

    try:
        with st.status(f"**{ticker}** analiz ediliyor — birkaç dakika sürebilir…",
                       expanded=True) as status:
            st.write("Graph kuruluyor, modeller başlatılıyor…")
            ta = TradingAgentsGraph(selected_analysts=selected, debug=False, config=config)
            benchmark = ta._resolve_benchmark(ticker)
            st.write(f"Benchmark (alpha için): **{benchmark}** · Rapor dili: **{language}**")
            macro_note = " → Makro(TR) → Siyaset(TR)" if is_bist_ticker(ticker) else ""
            st.write(f"Ajanlar çalışıyor: Teknik → Duygu/TR-Haber → Haber → Temel{macro_note} → "
                     "Araştırma → Trader → Risk → Karar…")
            final_state, _ = ta.propagate(ticker, date_str, asset_type=asset_type)
            status.update(label=f"{ticker} analizi tamamlandı ✓", state="complete", expanded=False)
    except Exception as e:  # noqa: BLE001
        import traceback as _tb
        st.error(f"Analiz sırasında hata: {type(e).__name__}: {e}")
        with st.expander("Teknik ayrıntı (traceback)"):
            st.code(_tb.format_exc())
        return

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
    st.caption(f"Alpha benchmark'ı: {benchmark} · Fiyatlar TRY · 5 kademe: {', '.join(RATINGS_5_TIER)}")

    st.download_button("📥 Tüm raporu indir (.md)",
                       _build_report_markdown(final_state, ticker, date_str),
                       file_name=f"{ticker}_{date_str}_analiz.md", mime="text/markdown")

    # Fail-loud: if the macro analyst ran on missing data, surface it up top so
    # the decision is never silently read as resting on a full macro picture.
    macro_health = final_state.get("macro_data_health") or ""
    if "VERİ UYARISI" in (final_state.get("macro_report") or ""):
        st.warning("⚠️ Makro veri kaynaklarına ulaşılamadı — makro değerlendirmesi "
                   "eksik veriyle üretildi. 🏛️ Makro-TR sekmesindeki **Veri Sağlığı**'na bak.")

    st.subheader("Karar gerekçesi")
    st.markdown(final_state.get("final_trade_decision") or "_(boş)_")

    tabs = st.tabs(["💬 Duygu / TR-Haber & KAP", "📊 Temel", "📈 Teknik",
                    "📰 Haber", "🏛️ Makro-TR", "🗳️ Siyaset-TR", "🧠 Araştırma",
                    "💼 Trader planı"])
    with tabs[0]:
        st.markdown(final_state.get("sentiment_report") or "_(seçili değil)_")
    with tabs[1]:
        st.markdown(final_state.get("fundamentals_report") or "_(seçili değil)_")
    with tabs[2]:
        st.markdown(final_state.get("market_report") or "_(seçili değil)_")
    with tabs[3]:
        st.markdown(final_state.get("news_report") or "_(seçili değil)_")
    with tabs[4]:
        st.caption("TCMB/faiz · enflasyon · kur · ülke riski → BIST geneli rejim ve sektör etkisi. "
                   "Yalnızca BIST (.IS) hisseleri için otomatik üretilir.")
        if macro_health:
            healthy = "✗" not in macro_health and "⚠️" not in macro_health
            with st.expander("🩺 Veri Sağlığı — kaynak durumları",
                             expanded=not healthy):
                st.caption("✓ canlı veri · ⚠️ boş/fallback · ✗ erişilemedi")
                st.code(macro_health, language=None)
        st.markdown(final_state.get("macro_report") or "_(BIST dışı — makro analizi üretilmedi)_")
    with tabs[5]:
        st.caption("İç siyaset · jeopolitik gerilim · seçim takvimi · siyasi şok olay-etüdü "
                   "→ risk primi ve hisse etkisi. Yalnızca BIST (.IS) hisseleri için üretilir.")
        st.markdown(final_state.get("geopolitics_report")
                    or "_(BIST dışı — siyasi risk analizi üretilmedi)_")
    with tabs[6]:
        deb = final_state.get("investment_debate_state", {}) or {}
        for head, key in [("🐂 Boğa", "bull_history"), ("🐻 Ayı", "bear_history"),
                          ("⚖️ Araştırma Yöneticisi", "judge_decision")]:
            if deb.get(key):
                st.markdown(f"#### {head}")
                st.markdown(deb[key])
        if not deb:
            st.markdown("_(yok)_")
    with tabs[7]:
        st.markdown(final_state.get("trader_investment_plan") or "_(yok)_")


# ════════════════════════════════════════════════════════════════════════════
# 📐 DİP-AL STRATEJİSİ EKRANI  (video.md: SMI + VWMA + Bollinger)
# ════════════════════════════════════════════════════════════════════════════
def _strategy_charts(df: pd.DataFrame):
    d = df.tail(180).copy()
    d["date"] = d.index
    base = alt.Chart(d).encode(x=alt.X("date:T", title=None))
    price = base.mark_line(color="#3b82f6").encode(
        y=alt.Y("Close:Q", title="Fiyat (TRY)", scale=alt.Scale(zero=False)))
    bbmid = base.mark_line(color="#9ca3af", strokeDash=[4, 3]).encode(y="bb_mid:Q")
    bbup = base.mark_line(color="#e5e7eb", strokeDash=[2, 3]).encode(y="bb_upper:Q")
    buys = alt.Chart(d[d["buy"]]).mark_point(
        color="#16a34a", size=120, shape="triangle-up", filled=True).encode(x="date:T", y="Close:Q")
    sells = alt.Chart(d[d["sell"]]).mark_point(
        color="#dc2626", size=120, shape="triangle-down", filled=True).encode(x="date:T", y="Close:Q")
    price_chart = (bbup + bbmid + price + buys + sells).properties(height=320)

    sb = alt.Chart(d).encode(x=alt.X("date:T", title=None))
    smi_line = sb.mark_line(color="#3b82f6").encode(y=alt.Y("smi:Q", title="SMI"))
    sig_line = sb.mark_line(color="#f59e0b").encode(y="smi_signal:Q")
    vwma_line = sb.mark_line(color="#8b5cf6", strokeDash=[3, 2]).encode(y="smi_vwma:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9ca3af").encode(y="y:Q")
    smi_chart = (zero + smi_line + sig_line + vwma_line).properties(height=180)
    return price_chart, smi_chart


def render_strategy_screen():
    st.title("📐 Dip-Al / Tepeden-Sat Stratejisi")
    st.caption("SMI(10,3,3) + SMI üzerine VWMA(7) + Bollinger orta bandı · "
               "Deterministik teknik sinyal (LLM yok) · Yatırım tavsiyesi değildir.")

    c1, c2, c3 = st.columns([2, 1.2, 1])
    with c1:
        ticker = st.text_input("Hisse (ticker)", value="THYAO.IS",
                               help="THYAO.IS, GARAN.IS, ASELS.IS …").strip().upper()
    with c2:
        interval = st.selectbox("Zaman dilimi", list(INTERVALS.keys()), index=0,
                                help="Günlük ve 4 saatlik en kaliteli (video).")
    with c3:
        st.write(""); st.write("")
        scan = st.button("🔍 Tara", type="primary", use_container_width=True)

    if not scan:
        st.info("Bir hisse + zaman dilimi seç, **Tara**'ya bas. "
                "Mavi=SMI, turuncu=sinyal, mor=VWMA(7); 🟢 yukarı üçgen = AL, 🔴 aşağı üçgen = SAT.")
        return

    res = strategy_analyze(ticker, interval)
    if not res.ok:
        st.error(res.error or "Analiz başarısız.")
        return

    color, emoji = _STATUS_STYLE.get(res.status, ("#6b7280", "⚪"))
    last = res.df.iloc[-1]
    close_txt = f"{last['Close']:.2f}" if pd.notna(last["Close"]) else "—"
    smi_txt = f"{last['smi']:.1f}" if pd.notna(last["smi"]) else "—"
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:12px;background:{color}1a;"
        f"border:2px solid {color};'>"
        f"<span style='font-size:13px;color:{color};font-weight:600;'>GÜNCEL DURUM · {ticker} · {interval}</span><br>"
        f"<span style='font-size:26px;font-weight:800;color:{color};'>{emoji} {res.status}</span> "
        f"<span style='color:#6b7280;'>· kapanış {close_txt} TRY · SMI {smi_txt}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### AL kriterleri (güncel)")
    cols = st.columns(len(res.conditions))
    for col, (label, ok) in zip(cols, res.conditions.items()):
        col.metric(label, "✓" if ok else "✗", delta=("sağlandı" if ok else "yok"),
                   delta_color=("normal" if ok else "off"))

    price_chart, smi_chart = _strategy_charts(res.df)
    st.altair_chart(price_chart, use_container_width=True)
    st.altair_chart(smi_chart, use_container_width=True)

    st.markdown("##### Son sinyaller")
    if res.signals:
        df_sig = pd.DataFrame(res.signals)[["date", "type", "price"]]
        df_sig.columns = ["Tarih", "Sinyal", "Fiyat (TRY)"]
        st.dataframe(df_sig.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("Bu pencerede sinyal bulunamadı.")

    st.caption("⚠️ Video notu: AL kombinasyonu fake sinyalleri iyi eler; SAT tarafı "
               "daha az hassastır — kademeli kâr realizasyonu önerilir.")


_STATUS_EMOJI = {"AL BÖLGESİ": "🟢 AL", "SAT UYARISI": "🔴 SAT", "NÖTR": "⚪ Nötr", "—": "⚠️ veri yok"}


def render_scanner():
    st.divider()
    st.subheader("🔎 BIST Tarayıcı — şu an AL bölgesindeki hisseler")
    sc1, sc2 = st.columns([1.2, 1])
    with sc1:
        interval = st.selectbox("Zaman dilimi (tarama)", list(INTERVALS.keys()), index=0,
                                key="scan_interval")
    with sc2:
        st.write(""); st.write("")
        do_scan = st.button(f"📡 {len(BIST_POPULAR)} BIST hissesini tara",
                            use_container_width=True)
    if not do_scan:
        st.caption("Likit BIST evrenini (BIST 30 + popüler) tarar, AL bölgesindekileri en üste sıralar. "
                   "Birkaç saniye sürer (paralel çeker).")
        return

    with st.spinner(f"{len(BIST_POPULAR)} hisse taranıyor…"):
        rows = strategy_scan(BIST_POPULAR, interval)

    al = [r for r in rows if r["status"] == "AL BÖLGESİ"]
    st.success(f"**{len(al)}** hisse şu an AL bölgesinde" + (": " + ", ".join(r["ticker"].replace(".IS","") for r in al) if al else "."))

    df = pd.DataFrame(rows)
    df["Durum"] = df["status"].map(lambda s: _STATUS_EMOJI.get(s, s))
    df["Kriter"] = df["met"].map(lambda m: f"{m}/3")
    view = df[["ticker", "Durum", "Kriter", "close", "smi"]].copy()
    view.columns = ["Hisse", "Durum", "AL kriteri", "Fiyat (TRY)", "SMI"]
    view["Hisse"] = view["Hisse"].str.replace(".IS", "", regex=False)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("⚠️ Yatırım tavsiyesi değildir. Sinyaller geçmiş veriyle hesaplanır; doğrulamadan işlem açma.")


# ── Yönlendirme ─────────────────────────────────────────────────────────────
if mode == "💼 Portföyüm":
    portfolio_page.render()
elif mode == "🏟️ Paper Arena":
    arena_page.render()
elif mode == "📡 BIST 30 Tarayıcı":
    scanner_page.render()
elif mode == "🤖 AI Analizi":
    render_ai_screen()
elif mode == "🎯 Birleşik Karar":
    combined_page.render()
elif mode == "🧰 Teknik Analiz":
    technical.render()
elif mode == "🧾 Temel Skor":
    fundamental_page.render()
elif mode == "🔮 ML Sinyal":
    ml_page.render()
elif mode == "📅 Sezonsallık":
    seasonality_page.render()
elif mode == "🥇 Altın & Döviz":
    gold_fx_page.render()
else:
    render_strategy_screen()
    render_scanner()

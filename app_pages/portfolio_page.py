"""💼 Portföyüm — elindeki hisseler (adet, alış fiyatı, tarih) + canlı P&L.

Hisse eklediğinde **anında** deterministik analiz çalışır ve ilk snapshot DB'ye
yazılır; eklenen hisse böylece saat başı çalışan zamanlayıcının (GitHub Actions)
canlı okuduğu ``holdings`` listesine girer — bir sonraki saat başında otomatik
döngüye dahil olur. Sildiğinde döngüden de düşer.

Her hisse için: güncel fiyat, maliyet ortalaması, kâr/zarar (TL ve %), portföy
ağırlığı; en güncel AL/SAT durumu, birleşik karar ve **çoklu-yöntem mutabakatı**
(güven). Detayda mum grafiği + sinyal istikrarı + sinyal karnesi.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app_pages._charts import candlestick_chart
from app_pages._styles import (
    AGREE_BADGE,
    DECISION_STYLE,
    STATUS_BADGE,
    agree_level_of,
    pnl_text,
)
from tradingagents.analysis import run as analysis_run
from tradingagents.analysis import trust
from tradingagents.analytics.backtest import edge_for_ticker
from tradingagents.analytics.combined import combined_signal
from tradingagents.analytics.composite import WEIGHTS, _fetch_daily
from tradingagents.analytics.confirmation import compute_confirmation
from tradingagents.analytics.multiframe import compute_mtf
from tradingagents.analytics.risk import compute_risk
from tradingagents.storage import portfolio, snapshots
from tradingagents.storage.prices import latest_prices
from tradingagents.storage.supabase_client import SupabaseError, is_configured

@st.cache_data(ttl=600, show_spinner=False)
def _cached_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    return latest_prices(list(tickers))


def _config_warning() -> None:
    st.warning(
        "⚙️ **Supabase bağlı değil.** Portföyü kaydetmek için Supabase kimlik "
        "bilgilerini ayarla:\n\n"
        "1. Ücretsiz proje aç → SQL Editor'de `tradingagents/storage/schema.sql`'i çalıştır.\n"
        "2. `SUPABASE_URL` ve `SUPABASE_SERVICE_KEY`'i Streamlit secrets'a (Cloud) "
        "veya `.env`'e (yerel) ekle.\n\n"
        "Ayrıntılı rehber: **docs/SUPABASE_SETUP.md**"
    )


# ── Ekleme formu ─────────────────────────────────────────────────────────────
def _add_form() -> None:
    st.markdown("##### ➕ Hisse ekle")
    with st.form("add_holding", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.2])
        with c1:
            ticker = st.text_input("Hisse", value="", placeholder="THYAO.IS",
                                   help="BIST için .IS ekle").strip().upper()
        with c2:
            quantity = st.number_input("Adet", min_value=0.0, value=0.0, step=1.0)
        with c3:
            buy_price = st.number_input("Alış fiyatı (TRY)", min_value=0.0,
                                        value=0.0, step=0.01, format="%.2f")
        with c4:
            buy_date = st.date_input("Alış tarihi", value=date.today())
        note = st.text_input("Not (opsiyonel)", value="", placeholder="örn. uzun vade")
        submitted = st.form_submit_button("➕ Ekle ve hemen analiz et",
                                          type="primary", use_container_width=True)

    if not submitted:
        return
    if not ticker or quantity <= 0 or buy_price <= 0:
        st.error("Hisse kodu, adet (>0) ve alış fiyatı (>0) zorunlu.")
        return

    try:
        portfolio.add_holding(ticker, quantity, buy_price, buy_date, note)
    except SupabaseError as e:
        st.error(f"Kaydedilemedi: {e}")
        return

    # Ekler eklemez anında deterministik analiz + ilk snapshot
    with st.spinner(f"{ticker} eklendi — anında analiz ediliyor…"):
        outcome = analysis_run.analyze_ticker(ticker)
        try:
            snapshots.write_snapshot(
                analysis_run.to_snapshot_row(outcome, scope="portfolio", source="on_add")
            )
        except SupabaseError as e:
            st.warning(f"Analiz yapıldı ama snapshot yazılamadı: {e}")

    if outcome.ok:
        color, emoji = DECISION_STYLE.get(outcome.decision, ("#6b7280", "⚪"))
        st.success(
            f"**{ticker}** eklendi · {emoji} **{outcome.decision}** "
            f"({outcome.combined_score:.0f}/100) · {AGREE_BADGE.get(outcome.agreement_level, '')}"
        )
    else:
        st.success(f"**{ticker}** eklendi.")
        st.info(f"Analiz şimdilik yapılamadı: {outcome.error}")
    _cached_prices.clear()
    st.rerun()


# ── Pozisyon tablosu ─────────────────────────────────────────────────────────
def _positions_table(positions: list, snaps: dict[str, dict]) -> None:
    rows = []
    for p in positions:
        snap = snaps.get(p.ticker, {})
        agree = agree_level_of(snap)
        rows.append({
            "Hisse": p.ticker.replace(".IS", ""),
            "Adet": p.quantity,
            "Maliyet": p.avg_cost,
            "Son Fiyat": p.last_price if p.last_price is not None else "—",
            "Değer (TRY)": p.market_value if p.market_value is not None else "—",
            "P&L (TRY)": round(p.pnl, 2) if p.pnl is not None else "—",
            "P&L %": pnl_text(p.pnl_pct),
            "Ağırlık %": p.weight_pct if p.weight_pct is not None else "—",
            "Durum": STATUS_BADGE.get(snap.get("status"), "—"),
            "Karar": snap.get("decision", "—"),
            "Mutabakat": AGREE_BADGE.get(agree, "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _summary_metrics(positions: list) -> None:
    invested = sum(p.invested for p in positions)
    mv = sum(p.market_value for p in positions if p.market_value is not None)
    have_mv = any(p.market_value is not None for p in positions)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pozisyon", f"{len(positions)}")
    c2.metric("Maliyet (TRY)", f"{invested:,.0f}")
    if have_mv:
        pnl = mv - invested
        pnl_pct = (mv / invested - 1) * 100 if invested else 0
        c3.metric("Piyasa değeri (TRY)", f"{mv:,.0f}")
        c4.metric("Toplam P&L (TRY)", f"{pnl:,.0f}", delta=f"{pnl_pct:.2f}%")
    else:
        c3.metric("Piyasa değeri (TRY)", "—")
        c4.metric("Toplam P&L (TRY)", "—")


# ── Lot yönetimi (silme) ─────────────────────────────────────────────────────
def _manage_lots(holdings: list[dict]) -> None:
    with st.expander("🗂️ Alış satırlarını yönet (sil)"):
        if not holdings:
            st.caption("Henüz alış satırı yok.")
            return
        for h in holdings:
            c1, c2 = st.columns([5, 1])
            c1.write(
                f"**{h['ticker'].replace('.IS','')}** · {float(h['quantity']):g} adet "
                f"@ {float(h['buy_price']):.2f} TRY · {h['buy_date']}"
                + (f" · _{h['note']}_" if h.get("note") else "")
            )
            if c2.button("🗑️ Sil", key=f"del_{h['id']}", use_container_width=True):
                try:
                    portfolio.delete_holding(h["id"])
                    _cached_prices.clear()
                    st.rerun()
                except SupabaseError as e:
                    st.error(f"Silinemedi: {e}")


# ── Karar kırılımı (teknik bileşenler + temel rasyolar) ──────────────────────
_COMPONENT_TR = {
    "trend": "Trend (SMA hiyerarşisi)",
    "momentum": "Momentum (RSI/MACD/Stokastik)",
    "pattern": "Grafik formasyonları",
    "money_flow": "Para giriş-çıkışı (hacim)",
    "dip": "Dip-Al stratejisi",
    "candle": "Mum formasyonları",
    "sr": "Destek/Direnç konumu",
    "seasonality": "Sezonsallık",
}


def _factor_breakdown(comb) -> None:
    """Birleşik kararın altındaki tüm faktörleri tek tek gösterir (şeffaflık/güven)."""
    tech, fund = comb.technical, comb.fundamental
    t1, t2 = st.columns(2)
    with t1:
        with st.expander("🧰 Teknik bileşenler (kompozit skor kırılımı)"):
            if tech is None:
                st.caption("Teknik veri yok.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Bileşen": _COMPONENT_TR.get(k, k),
                    "Ağırlık": WEIGHTS[k],
                    "Skor [-1,+1]": round(tech.components.get(k, 0.0), 2),
                    "Açıklama": tech.details.get(k, ""),
                } for k in WEIGHTS]), use_container_width=True, hide_index=True)
                if tech.regime is not None:
                    st.caption(f"🌡️ Rejim: {tech.regime.summary}")
                    if getattr(tech.regime, "try_note", None):
                        st.caption(f"💵 {tech.regime.try_note}")
                for w in (tech.warnings or []):
                    st.caption(f"⚠️ {w}")
                if tech.candles:
                    st.markdown("**🕯️ Mum formasyonları (son barlar)**")
                    st.dataframe(pd.DataFrame([{
                        "Tarih": h.date, "Formasyon": h.name, "Yön": h.direction,
                        "Güç": "★" * h.strength,
                    } for h in tech.candles]), use_container_width=True, hide_index=True)
                if tech.patterns:
                    st.markdown("**📐 Grafik formasyonları**")
                    st.dataframe(pd.DataFrame([{
                        "Formasyon": h.name, "Yön": h.direction,
                        "Durum": "✅ teyitli" if h.confirmed else "⏳ oluşum",
                    } for h in tech.patterns]), use_container_width=True, hide_index=True)
                if tech.supports or tech.resistances:
                    st.markdown("**🧱 Destek / Direnç**")
                    st.dataframe(pd.DataFrame(
                        [{"Seviye": lv.price, "Tip": "🟢 destek"} for lv in tech.supports]
                        + [{"Seviye": lv.price, "Tip": "🔴 direnç"} for lv in tech.resistances]
                    ), use_container_width=True, hide_index=True)
    with t2:
        with st.expander("🧾 Temel rasyolar (karar kırılımı)"):
            criteria = getattr(fund, "criteria", None) if fund else None
            if not criteria:
                st.caption("Temel veri yok.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Rasyo": c.name,
                    "Değer": round(c.value, 2) if c.value is not None else "—",
                    "Bant": c.band,
                    "Skor [0-1]": round(c.score, 2),
                    "Ağırlık": c.weight,
                } for c in criteria]), use_container_width=True, hide_index=True)
                if getattr(fund, "is_financial", False):
                    st.caption("ℹ️ Banka/sigorta için uyarlanmış kriter seti (PD/DD ağırlıklı).")


@st.cache_data(ttl=900, show_spinner=False)
def _cached_benchmark():
    return _fetch_daily("XU100.IS")


def _confidence_v2(sel: str, df) -> None:
    """Güven katmanı v2: risk (ATR/R-R) + teyit satır içi; MTF + backtest talep üzerine."""
    st.markdown("##### 🛡️ Güven katmanı v2")

    rp = compute_risk(df)
    if rp.ok:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stop (ATR×2)", f"{rp.stop:.2f}")
        m2.metric("Hedef", f"{rp.target:.2f}")
        m3.metric("Risk/Ödül", f"{rp.rr:.2f}")
        m4.metric("Likidite", rp.liquidity)
        for n in rp.notes:
            st.caption(f"• {n}")

    conf = compute_confirmation(df, benchmark_df=_cached_benchmark())
    if conf.ok:
        bits = []
        if conf.rsi_divergence != "yok":
            bits.append(f"RSI divergence: **{conf.rsi_divergence}**")
        if conf.macd_divergence != "yok":
            bits.append(f"MACD divergence: **{conf.macd_divergence}**")
        bits.append(f"Hacim teyidi: {'✅' if conf.volume_confirms else '—'}")
        bits.append(f"OBV: {conf.obv_trend}")
        if conf.rel_strength is not None:
            bits.append(conf.rs_note)
        st.caption(" · ".join(bits))

    col_a, col_b = st.columns(2)
    if col_a.button("🔭 Çoklu zaman dilimi teyidi", key=f"mtf_{sel}",
                    use_container_width=True):
        with st.spinner("Haftalık / günlük / 4 saatlik çekiliyor…"):
            mtf = compute_mtf(sel)
        if mtf.ok:
            st.info(f"**Konfluens: {mtf.confluence}** (skor {mtf.score:+.2f})")
            for label, d in mtf.frames.items():
                st.caption(f"• {label}: {d['detail']}")
        else:
            st.caption(mtf.error or "MTF hesaplanamadı.")
    if col_b.button("🧪 Geçmiş backtest (Dip-Al)", key=f"bt_{sel}",
                    use_container_width=True):
        with st.spinner("Geçmiş sinyaller test ediliyor…"):
            edge = edge_for_ticker(sel, df)
        if edge.ok and edge.n_signals:
            eq = edge.equity
            st.caption(edge.note)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Kazanma %", eq.get("win_rate"))
            e2.metric("CAGR %", eq.get("cagr"))
            e3.metric("Max düşüş %", eq.get("max_drawdown"))
            e4.metric("Sharpe", eq.get("sharpe"))
            st.dataframe(pd.DataFrame(
                [{"Ufuk": h, "Örnek": v["n"], "Ort. %": v["mean"],
                  "Medyan %": v["median"], "İsabet %": v["hit_rate"]}
                 for h, v in edge.fwd.items()]),
                use_container_width=True, hide_index=True)
        else:
            st.caption(edge.error or "Geçmişte bu sinyalden örnek yok.")


# ── Detay (mum grafiği + güven) ──────────────────────────────────────────────
def _detail_view(tickers: list[str]) -> None:
    st.markdown("##### 🔍 Hisse detayı")
    sel = st.selectbox("Detayını göster", tickers,
                       format_func=lambda t: t.replace(".IS", ""))
    if not sel:
        return
    recompute = st.button(f"🔄 {sel.replace('.IS','')} — şimdi analiz et",
                          use_container_width=True)

    df = _fetch_daily(sel)
    if df is None or df.empty:
        st.info(f"{sel} için fiyat verisi alınamadı.")
        return

    st.altair_chart(candlestick_chart(df), use_container_width=True)

    comb = combined_signal(sel, df=df)
    if comb.ok:
        color, emoji = DECISION_STYLE.get(comb.decision, ("#6b7280", "⚪"))
        st.markdown(
            f"<div style='padding:12px 16px;border-radius:10px;background:{color}1a;"
            f"border:2px solid {color};'>"
            f"<b style='color:{color};font-size:20px;'>{emoji} {comb.decision}</b> "
            f"<span style='color:#6b7280;'>· birleşik {comb.combined_score:.0f}/100</span></div>",
            unsafe_allow_html=True,
        )
        with st.expander("Gerekçe (şeffaflık)"):
            for r in comb.rationale:
                st.markdown(f"- {r}")
        _factor_breakdown(comb)

    _confidence_v2(sel, df)

    # Güven: geçmişten istikrar + sinyal karnesi
    try:
        hist = snapshots.history(sel, scope="portfolio", limit=300)
    except SupabaseError:
        hist = []
    chrono = list(reversed(hist))  # eski→yeni
    level, score, flips = trust.signal_stability([h.get("status") for h in chrono])
    record = trust.track_record(chrono, horizon_days=1)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📉 Sinyal istikrarı**")
        if level == "yetersiz":
            st.caption("Veri birikiyor — istikrar için en az 3 snapshot gerekli.")
        else:
            st.caption(f"{level} (skor {score}, {flips} değişim / {len(chrono)} gözlem)")
    with c2:
        st.markdown("**🎯 Sinyal karnesi (+1g)**")
        if not record["ready"]:
            st.caption("Veri birikiyor — karne için yeterli geçmiş sinyal yok.")
        else:
            al = record["AL"]
            st.caption(
                f"AL: {al['n']} sinyal · isabet %{al['hit_rate']} · "
                f"ort. getiri {al['mean_fwd']:+.2f}%"
            )

    if recompute:
        with st.spinner(f"{sel} yeniden analiz ediliyor…"):
            outcome = analysis_run.analyze_ticker(sel, df=df)
            try:
                snapshots.write_snapshot(
                    analysis_run.to_snapshot_row(outcome, scope="portfolio", source="manual")
                )
                st.success("Yeni snapshot yazıldı ✓")
            except SupabaseError as e:
                st.error(f"Snapshot yazılamadı: {e}")
        st.rerun()


# ── Sayfa ────────────────────────────────────────────────────────────────────
def render() -> None:
    st.title("💼 Portföyüm")
    st.caption("Elindeki hisseler · canlı kâr/zarar · saat başı otomatik analiz · "
               "Yatırım tavsiyesi değildir.")

    if not is_configured():
        _config_warning()
        _add_form()  # form yine görünsün ki kullanıcı akışı görsün (ekleme hata verir)
        return

    try:
        holdings = portfolio.list_holdings()
    except SupabaseError as e:
        st.error(f"Portföy okunamadı: {e}")
        msg = str(e)
        if "401" in msg or "Invalid API key" in msg or "JWT" in msg:
            st.warning(
                "🔑 **Anahtar reddedildi.** `SUPABASE_SERVICE_KEY` yanlış/eksik "
                "kopyalanmış olabilir. Supabase → Settings → API'den **service_role** "
                "anahtarını (genelde `eyJ…` ile başlayan JWT) **sondaki nokta/boşluk "
                "olmadan** kopyalayıp Streamlit secrets'ı güncelle, sonra **Reboot app**."
            )
        elif "404" in msg or "does not exist" in msg or "relation" in msg:
            st.warning("🗄️ Tablolar yok gibi. `tradingagents/storage/schema.sql`'i "
                       "Supabase SQL Editor'de çalıştırdın mı?")
        else:
            _config_warning()
        return

    _add_form()
    st.divider()

    if not holdings:
        st.info("Portföyün boş. Yukarıdan ilk hisseni ekle — eklenir eklenmez analiz edilir.")
        return

    tickers = sorted({h["ticker"].upper() for h in holdings})
    cols = st.columns([3, 1])
    with cols[1]:
        if st.button("🔄 Fiyatları yenile", use_container_width=True):
            _cached_prices.clear()
            st.rerun()
        manual_all = st.button("📡 Tümünü şimdi analiz et", use_container_width=True,
                               type="primary")

    prices = _cached_prices(tuple(tickers))
    positions = portfolio.compute_positions(holdings, prices)
    try:
        snaps = snapshots.latest_for(tickers, scope="portfolio")
    except SupabaseError:
        snaps = {}

    _summary_metrics(positions)
    _positions_table(positions, snaps)
    st.caption("Durum/Karar/Mutabakat sütunları en güncel snapshot'tan gelir; "
               "saat başı zamanlayıcı bunları otomatik tazeler.")

    _manage_lots(holdings)
    st.divider()
    _detail_view(tickers)

    if manual_all:
        prog = st.progress(0.0, text="Analiz ediliyor…")
        rows = []
        for i, tk in enumerate(tickers, 1):
            outcome = analysis_run.analyze_ticker(tk)
            rows.append(analysis_run.to_snapshot_row(outcome, scope="portfolio", source="manual"))
            prog.progress(i / len(tickers), text=f"{tk} ({i}/{len(tickers)})")
        try:
            snapshots.write_snapshots(rows)
            st.success(f"{len(rows)} hisse analiz edildi ve kaydedildi ✓")
        except SupabaseError as e:
            st.error(f"Snapshot yazılamadı: {e}")
        st.rerun()

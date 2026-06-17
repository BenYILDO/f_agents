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
from tradingagents.analysis import run as analysis_run
from tradingagents.analysis import trust
from tradingagents.analytics.combined import combined_signal
from tradingagents.analytics.composite import _fetch_daily
from tradingagents.storage import portfolio, snapshots
from tradingagents.storage.prices import latest_prices
from tradingagents.storage.supabase_client import SupabaseError, is_configured

_DECISION_STYLE = {
    "GÜÇLÜ AL": ("#15803d", "🟢"),
    "AL": ("#16a34a", "🟢"),
    "TUT": ("#6b7280", "⚪"),
    "SAT": ("#dc2626", "🔴"),
    "KAÇIN": ("#991b1b", "🔴"),
    "VERİ YOK": ("#9ca3af", "⚠️"),
}
_AGREE_BADGE = {
    "güçlü": "🟢 güçlü mutabakat",
    "kısmi": "🟡 kısmi mutabakat",
    "çelişki": "🔴 çelişki",
    "nötr": "⚪ nötr",
}
_STATUS_BADGE = {"AL": "🟢 AL", "SAT": "🔴 SAT", "NÖTR": "⚪ Nötr"}


@st.cache_data(ttl=600, show_spinner=False)
def _cached_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    return latest_prices(list(tickers))


def _pnl_text(pnl_pct: float | None) -> str:
    if pnl_pct is None:
        return "—"
    arrow = "🟢 +" if pnl_pct >= 0 else "🔴 "
    return f"{arrow}{pnl_pct:.2f}%"


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
        color, emoji = _DECISION_STYLE.get(outcome.decision, ("#6b7280", "⚪"))
        st.success(
            f"**{ticker}** eklendi · {emoji} **{outcome.decision}** "
            f"({outcome.combined_score:.0f}/100) · {_AGREE_BADGE.get(outcome.agreement_level, '')}"
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
        agree = (snap.get("agreement") or "").split(":", 1)[0].strip()
        rows.append({
            "Hisse": p.ticker.replace(".IS", ""),
            "Adet": p.quantity,
            "Maliyet": p.avg_cost,
            "Son Fiyat": p.last_price if p.last_price is not None else "—",
            "Değer (TRY)": p.market_value if p.market_value is not None else "—",
            "P&L (TRY)": round(p.pnl, 2) if p.pnl is not None else "—",
            "P&L %": _pnl_text(p.pnl_pct),
            "Ağırlık %": p.weight_pct if p.weight_pct is not None else "—",
            "Durum": _STATUS_BADGE.get(snap.get("status"), "—"),
            "Karar": snap.get("decision", "—"),
            "Mutabakat": _AGREE_BADGE.get(agree, "—"),
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
        color, emoji = _DECISION_STYLE.get(comb.decision, ("#6b7280", "⚪"))
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

"""🏟️ Paper Arena — eşit kasayla yarışan kâğıt hesaplar + XU100 edge kapısı.

İki bölüm:
  1. **Edge kapısı (replay):** Profiller 5 yıllık BIST verisinde, ortak execution
     fiziğiyle simüle edilir ve XU100 al-tut ile kıyaslanır. Planın "önce kâr var mı?"
     sorusunu yanıtlar — arena altyapısına yatırım gerekçeli mi?
  2. **Lig & profiller:** Her hesabın kuralları, getirisi, Sharpe'ı, drawdown'u,
     işlem sayısı; ML-öncelikli hesap OBSERVER olarak işaretli.

Tamamen yerel · LLM yok · ücret harcamaz. Supabase gerekmez (replay anlıktır).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tradingagents.arena.config import DEFAULT_EXECUTION
from tradingagents.arena.profiles import PROFILES
from tradingagents.arena.replay import run_arena_replay


def _equity_chart(result):
    """Tüm profillerin + XU100'ün equity eğrilerini tek grafikte toplar."""
    frames = []
    for code, r in result.per_profile.items():
        if r.ok and len(r.equity_curve):
            prof = PROFILES[code]
            s = r.equity_curve.rename(f"{prof.emoji} {prof.name}")
            frames.append(s)
    if len(result.benchmark_equity):
        frames.append(result.benchmark_equity.rename("📊 XU100 (al-tut)"))
    if not frames:
        return None
    wide = pd.concat(frames, axis=1).ffill()
    return wide


def _render_replay():
    st.subheader("🎯 Edge kapısı — maliyet sonrası XU100'ü geçiyor muyuz?")
    st.caption("Planın 'fail-cheap' kararı: arena altyapısına geçmeden önce sinyalin "
               "gerçekten alfa ürettiğini kanıtla. 5 yıllık BIST verisi · ortak fizik.")

    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        period = st.selectbox("Tarihsel pencere", ["3y", "5y", "10y"], index=1,
                              help="Daha uzun = daha çok rejim, daha yavaş.")
    with c2:
        st.metric("Komisyon", f"{DEFAULT_EXECUTION.commission_bps:.0f} bps")
    with c3:
        st.metric("Slippage", f"{DEFAULT_EXECUTION.slippage_bps:.0f} bps")

    if not st.button("🚀 Edge kapısını çalıştır", type="primary", use_container_width=True):
        st.info("**Edge kapısını çalıştır**'a bas. BIST30 + XU100 verisi çekilir "
                "(birkaç dakika, ilk sefer yavaş), her profil simüle edilir. "
                "Sonuç: hangi profil XU100'ü maliyet sonrası geçiyor?")
        return

    prog = st.progress(0.0, text="Başlatılıyor…")
    result = run_arena_replay(period=period,
                              progress=lambda f, t: prog.progress(min(f, 1.0), text=t))
    prog.empty()

    if not result.ok:
        st.error(f"Replay başarısız: {result.error}")
        return

    st.session_state["arena_replay"] = result
    _show_replay_result(result)


def _show_replay_result(result):
    # Edge kapısı verdiği
    if result.edge_gate_passed:
        st.success(result.edge_summary)
    else:
        st.warning(result.edge_summary)
    st.caption(f"{result.n_tickers} hisse · pencere {result.period} · "
               f"XU100 getiri %{result.benchmark_metrics.total_return*100:+.1f} · "
               f"Sharpe {result.benchmark_metrics.sharpe:.2f}")

    # Lig tablosu
    st.markdown("##### 🏆 Lig tablosu (getiriye göre)")
    rows = []
    for x in result.leaderboard:
        rows.append({
            "": x["emoji"],
            "Hesap": x["name"],
            "Kasa (TL)": f"{x['final_equity']:,.0f}",
            "Getiri": f"%{x['total_return']*100:+.1f}",
            "XU100'e karşı": f"%{x['alpha_vs_xu100']*100:+.1f}",
            "CAGR": f"%{x['cagr']*100:+.1f}",
            "Sharpe": f"{x['sharpe']:.2f}",
            "Maks DD": f"%{x['max_drawdown']*100:.1f}",
            "İşlem": x["n_trades"],
            "İsabet": f"%{x['win_rate']*100:.0f}",
            "XU100>?": "✅" if x["beats_benchmark"] else "—",
        })
    # XU100 referans satırı
    bm = result.benchmark_metrics
    rows.append({
        "": "📊", "Hesap": "XU100 (al-tut)",
        "Kasa (TL)": f"{result.benchmark_equity.iloc[-1]:,.0f}" if len(result.benchmark_equity) else "—",
        "Getiri": f"%{bm.total_return*100:+.1f}", "XU100'e karşı": "—",
        "CAGR": f"%{bm.cagr*100:+.1f}", "Sharpe": f"{bm.sharpe:.2f}",
        "Maks DD": f"%{bm.max_drawdown*100:.1f}", "İşlem": "—", "İsabet": "—", "XU100>?": "—",
    })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Equity eğrileri
    st.markdown("##### 📈 Kasa eğrileri")
    wide = _equity_chart(result)
    if wide is not None:
        st.line_chart(wide, use_container_width=True)

    # Profil detayları
    st.markdown("##### 🔍 Profil detayları")
    for x in result.leaderboard:
        code = x["code"]
        prof = PROFILES[code]
        r = result.per_profile[code]
        with st.expander(f"{prof.emoji} {prof.name} — %{x['total_return']*100:+.1f} · "
                         f"{x['n_trades']} işlem"):
            st.caption(prof.blurb)
            if r.trades:
                last = r.trades[-8:]
                st.dataframe(pd.DataFrame([{
                    "Hisse": t.ticker.replace(".IS", ""), "Giriş": t.entry_date,
                    "Çıkış": t.exit_date, "Adet": t.quantity,
                    "K/Z": f"{t.pnl:+,.0f}", "K/Z %": f"%{t.pnl_pct*100:+.1f}",
                    "Sebep": t.exit_reason,
                } for t in last]), use_container_width=True, hide_index=True)
            else:
                st.caption("Bu profil hiç işlem açmadı (filtreler tuttu).")

    # Sınırlamalar — dürüstlük
    with st.expander("⚠️ Bilimsel sınırlamalar (mutlaka oku)"):
        for c in result.caveats:
            st.markdown(f"- {c}")
        st.caption("Bu arena 'kâr garantisi' değil, **strateji eleme ve operasyon "
                   "doğrulama ortamıdır**. Paper'da kötü olanı eler; iyi olanı ileri "
                   "test için aday yapar (plan §Akademi gerekli ama borsa akademik değil).")


def _render_profiles():
    st.subheader("👥 Yarışan hesaplar")
    st.caption("Hepsi 100.000 TL eşit kasayla, aynı execution fiziğiyle başlar. "
               "Farkları: evren kapısı, sizing, risk, rejim filtresi.")
    for p in PROFILES.values():
        badge = "🟢 PARA" if p.status == "ACTIVE" else "🔵 OBSERVER"
        with st.container(border=True):
            st.markdown(f"### {p.emoji} {p.name} &nbsp; `{badge}`")
            st.caption(p.blurb)
            cols = st.columns(4)
            cols[0].metric("Güven eşiği", f"{p.min_confidence:.0f}")
            cols[1].metric("Maks pozisyon", p.max_positions)
            cols[2].metric("İşlem riski", f"%{p.risk_per_trade*100:.1f}")
            cols[3].metric("Rejim filtresi", "açık" if p.use_regime_filter else "kapalı")
    st.info("🤖 **ML-öncelikli hesap V1'de OBSERVER:** para harcamaz; kalibre p_up "
            "tahminlerini ve karnesini biriktirir. Kalite + hedef-uyumu kanıtlanınca "
            "sonraki sezon parayla girer (plan F0.6 — kilitli karar).")


def render() -> None:
    st.title("🏟️ Paper Arena")
    st.caption("Eşit kasayla yarışan kâğıt hesaplar · XU100 edge kapısı · "
               "LLM yok, ücretsiz · Yatırım tavsiyesi değildir.")

    tab1, tab2 = st.tabs(["🎯 Edge Kapısı & Lig", "👥 Hesaplar"])
    with tab1:
        # Önceki sonucu hatırla (sayfa yenilenince kaybolmasın)
        cached = st.session_state.get("arena_replay")
        if cached is not None and cached.ok:
            with st.expander("↺ Son replay sonucunu göster", expanded=False):
                _show_replay_result(cached)
        _render_replay()
    with tab2:
        _render_profiles()

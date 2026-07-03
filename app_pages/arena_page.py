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


@st.cache_data(ttl=600, show_spinner=False)
def _last_close(ticker: str) -> float | None:
    """Pozisyonların güncel K/Z'si için son kapanış (10 dk önbellekli)."""
    from tradingagents.analytics.composite import _fetch_daily
    try:
        df = _fetch_daily(ticker, period="1mo")
        return float(df["Close"].iloc[-1]) if df is not None and len(df) else None
    except Exception:  # noqa: BLE001
        return None


def _render_model_maintenance():
    """Gecelik model işinin UI'dan koşumu — GitHub Actions kullanılamıyorsa.

    Aynı kod yolu (scripts.run_nightly_models.run_nightly): per-ticker kalibre
    olasılık + havuz modeli + Supabase yazımı. Ağır iştir (birkaç dakika).
    """
    with st.expander("🧠 Model bakımı — gecelik işi şimdi çalıştır (Actions'sız)"):
        st.caption("GitHub Actions kapalıysa (özel repo dakika/ödeme sınırı) aynı "
                   "iş buradan koşar: her hisse için kalibre kazanma olasılığı + "
                   "havuz (pooled) modeli eğitilir, Supabase'e yazılır. "
                   "**Birkaç dakika sürer**; günde bir kez yeterli.")
        if st.button("🧠 Modelleri şimdi eğit", use_container_width=True,
                     key="train_models_now"):
            prog = st.progress(0.0, text="Başlıyor…")
            try:
                from scripts.run_nightly_models import run_nightly
                summary = run_nightly(progress=lambda f, t: prog.progress(f, text=t))
            except Exception as e:  # noqa: BLE001
                prog.empty()
                st.error(f"Eğitim koşamadı: {type(e).__name__}: {e}")
                return
            prog.empty()
            if summary.get("ok"):
                st.success(f"{summary['n_models']}/{summary['n_universe']} hisse modeli "
                           f"yazıldı · {summary['elapsed_s']}s. Havuz karnesi aşağıda.")
            else:
                st.error(summary.get("error") or "Eğitim başarısız.")

        # Son havuz karneleri (varsa) — modelin zamanla iyileşme izi
        try:
            from tradingagents.storage import pooled_models
            cards = pooled_models.list_report_cards(limit=5)
        except Exception:  # noqa: BLE001
            cards = []
        if cards:
            st.markdown("**Son havuz (pooled) model karneleri**")
            st.dataframe(pd.DataFrame([{
                "Eğitim": (c.get("trained_at") or "")[:16].replace("T", " "),
                "Sürüm": c.get("model_version"),
                "AUC": c.get("auc"), "BSS": c.get("brier_skill_score"),
                "Örnek": c.get("n_samples"),
                "Kalite": "✅ GEÇTİ" if c.get("quality_passed") else "❌ geçemedi",
                "Neden": ", ".join(c.get("rejection_reasons") or []) or "—",
            } for c in cards]), use_container_width=True, hide_index=True)
            st.caption("Kalite kapısını geçen model çıkana dek ML meta-filtresi "
                       "pasiftir (bu bir hata değil, emniyettir). AUC ≥ 0.52 ve "
                       "BSS > 0 istikrarlı gelmeye başlarsa filtre kendiliğinden "
                       "devreye girer.")


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
    if len(getattr(result, "equalweight_equity", [])):
        frames.append(result.equalweight_equity.rename("⚖️ Eşit-ağırlık BIST"))
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
        period = st.selectbox("Tarihsel pencere", ["1y", "2y", "3y", "5y", "10y"], index=1,
                              help="İlk %80 eğitim, son %20 görülmemiş test. "
                                   "Daha uzun = daha çok rejim, daha yavaş.")
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
    bm0 = result.benchmark_metrics
    st.caption(f"{result.n_tickers} hisse · pencere {result.period} · "
               f"XU100 getiri %{bm0.total_return*100:+.1f} · Sharpe {bm0.sharpe:.2f} · "
               f"Maks DD %{bm0.max_drawdown*100:.1f}")

    # Dayanıklılık verdisi (overfitting kontrolü) — kullanıcının asıl istediği
    if result.robust_summary:
        if result.robust_summary.startswith("🛡️"):
            st.success(result.robust_summary)
        else:
            st.warning(result.robust_summary)
    st.caption(f"📐 Eğitim/test bölme noktası: {result.split_session} "
               f"(ilk %{int(result.train_frac*100)} eğitim · son "
               f"%{int((1-result.train_frac)*100)} görülmemiş test)")

    st.info("ℹ️ **Edge kapısı risk-ayarlıdır** (Sharpe + drawdown); enflasyonist index'i "
            "mutlak getiride geçmek yanlış bardır. **Dayanıklı** = hem eğitimde hem "
            "görülmemiş testte XU100'ü geçen → overfit değil, izlenecek aday.")

    # Lig tablosu — test (görülmemiş) Sharpe'ına göre sıralı
    st.markdown("##### 🏆 Lig tablosu — görülmemiş test dönemine göre sıralı")
    rows = []
    for x in result.leaderboard:
        rows.append({
            "": x["emoji"],
            "Hesap": x["name"],
            "Tüm dönem getiri": f"%{x['total_return']*100:+.1f}",
            "Eğitim getiri": f"%{x['train_return']*100:+.1f}",
            "Test getiri": f"%{x['test_return']*100:+.1f}",
            "Test Sharpe": f"{x['test_sharpe']:.2f}",
            "Test Maks DD": f"%{x['test_max_drawdown']*100:.1f}",
            "İşlem": str(x["n_trades"]),
            "Eğitim>XU?": "✅" if x["beats_train"] else "—",
            "Test>XU?": "✅" if x["beats_test"] else "—",
            "🛡️ Dayanıklı": "✅" if x["robust"] else "—",
        })
    # XU100 + eşit-ağırlık referans satırları
    bm = result.benchmark_metrics
    ew = result.equalweight_metrics
    rows.append({
        "": "📊", "Hesap": "XU100 (al-tut)",
        "Tüm dönem getiri": f"%{bm.total_return*100:+.1f}", "Eğitim getiri": "—",
        "Test getiri": "—", "Test Sharpe": f"{bm.sharpe:.2f}",
        "Test Maks DD": f"%{bm.max_drawdown*100:.1f}", "İşlem": "—",
        "Eğitim>XU?": "—", "Test>XU?": "—", "🛡️ Dayanıklı": "—",
    })
    rows.append({
        "": "⚖️", "Hesap": "Eşit-ağırlık BIST (böl-tut)",
        "Tüm dönem getiri": f"%{ew.total_return*100:+.1f}", "Eğitim getiri": "—",
        "Test getiri": "—", "Test Sharpe": f"{ew.sharpe:.2f}",
        "Test Maks DD": f"%{ew.max_drawdown*100:.1f}", "İşlem": "—",
        "Eğitim>XU?": "—", "Test>XU?": "—", "🛡️ Dayanıklı": "—",
    })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("**Test getiri/Sharpe** = son %20 (görülmemiş) dönem — overfitting'in "
               "gerçek sınavı. **🛡️ Dayanıklı** = hem eğitimde hem testte XU100'ü geçen.")

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
            # Çıkış-sebebi dökümü: kaç stop / sinyal / hedef / süre / sezon-sonu
            if r.trades:
                from collections import Counter
                cnt = Counter(t.exit_reason for t in r.trades)
                wins = sum(1 for t in r.trades if t.pnl > 0)
                st.caption(
                    f"Çıkışlar → " + " · ".join(f"{k}: {v}" for k, v in cnt.most_common())
                    + f"  |  kazanan {wins}/{len(r.trades)} · "
                    f"Sharpe {x['sharpe']:.2f} · Maks DD %{x['max_drawdown']*100:.1f}"
                )
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


def _render_live():
    from tradingagents.arena.config import DEFAULT_EXECUTION
    from tradingagents.arena.live import build_session_inputs, run_session
    from tradingagents.arena.state import (
        D, load_state, new_state, save_state, storage_backend, supabase_diagnose,
    )

    st.subheader("🔴 Canlı Sezon — bugünün sinyalleriyle ileriye işleyen arena")
    st.caption("Her seansta gerçek analiz (güven+rejim+makro şok+p_up) emir üretir, "
               "emirler ertesi seans açılışında (T+1) dolar, kasa/pozisyon/equity birikir.")

    backend = storage_backend()
    cda, cdb = st.columns([2, 1])
    with cda:
        if backend == "supabase":
            st.success("💾 Kalıcılık: **Supabase** — sezon reboot'a dayanır. "
                       "(Veri yalnız seans çalıştırınca yazılır.)")
        else:
            st.warning("💾 Kalıcılık: **yerel (geçici)** — secrets okunamadı. "
                       "Supabase secrets + `arena_state` tablosu gerekir.")
    with cdb:
        if st.button("🔍 Supabase'i test et", use_container_width=True):
            ok, msg = supabase_diagnose()
            (st.success if ok else st.error)(msg)

    _render_model_maintenance()

    state = load_state() or new_state()

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        run = st.button("▶️ Bugünün seansını çalıştır", type="primary",
                        use_container_width=True)
    with c2:
        st.metric("Sezon", state.season_id)
    with c3:
        st.metric("Son seans", state.last_session or "—")

    if st.button("🔄 Sezonu sıfırla (tüm kasaları 100k'ya döndür)"):
        save_state(new_state())
        st.success("Sezon sıfırlandı.")
        st.rerun()

    if run:
        prog = st.progress(0.0, text="Bugünün sinyalleri üretiliyor (analyze_universe)…")
        try:
            outcomes, prices, session = build_session_inputs()
            prog.progress(0.7, text=f"Seans {session} işleniyor…")
            report = run_session(state, outcomes, prices, session, DEFAULT_EXECUTION)
            save_state(state)
            prog.empty()
            if report.skipped:
                st.info(f"Seans {session} zaten işlenmiş (idempotent — yeni emir üretilmedi).")
            else:
                st.success(f"Seans {session}: {report.filled} emir doldu · "
                           f"{report.new_orders} yeni emir kuyruğa girdi · "
                           f"{report.predictions} observer tahmini.")
        except Exception as e:  # noqa: BLE001
            prog.empty()
            st.error(f"Seans çalıştırılamadı: {type(e).__name__}: {e}")
            return
        state = load_state() or state

    # ── Canlı lig tablosu ──────────────────────────────────────────────────
    init_cap = D(DEFAULT_EXECUTION.initial_capital)
    rows = []
    for code, acc in state.accounts.items():
        prof = PROFILES.get(code)
        last_eq = D(acc.equity_history[-1]["equity"]) if acc.equity_history else acc.cash
        ret = (last_eq / init_cap - 1) * 100 if init_cap else D(0)
        n_fills = sum(1 for e in acc.ledger
                      if e.get("event_type") in ("BUY", "SELL"))
        rows.append({
            "": prof.emoji if prof else "",
            "Hesap": prof.name if prof else code,
            "Tür": "OBSERVER" if acc.status == "OBSERVER" else "PARA",
            "Kasa+Pozisyon (TL)": f"{float(last_eq):,.0f}",
            "Getiri": f"%{float(ret):+.1f}",
            "Gerçekleşen K/Z (TL)": f"{float(acc.realized_pnl):+,.0f}",
            "Nakit (TL)": f"{float(acc.cash):,.0f}",
            "Pozisyon": str(len(acc.positions)),
            "Bekleyen emir": str(len(acc.pending_orders)),
            "İşlem": str(n_fills),
        })
    st.markdown("##### 🏆 Canlı lig")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("⏳ İlk seansta hesaplar emir kuyruğa alır (T+1) — pozisyonlar ertesi "
               "seans açılışında oluşur. Her gün bir kez çalıştır (ya da cron). "
               "Getiri = kasa+pozisyonun 100k'ya göre değişimi; Gerçekleşen K/Z "
               "yalnız kapanan işlemlerin toplamıdır.")

    # ── Hesap detayları ────────────────────────────────────────────────────
    for code, acc in state.accounts.items():
        prof = PROFILES.get(code)
        if acc.status == "OBSERVER":
            continue
        title = f"{prof.emoji if prof else ''} {prof.name if prof else code}"
        with st.expander(f"{title} — {len(acc.positions)} pozisyon · "
                         f"{len(acc.pending_orders)} bekleyen"):
            if acc.positions:
                st.markdown("**Açık pozisyonlar**")
                pos_rows = []
                for p in acc.positions.values():
                    last = _last_close(p.ticker)
                    cost = float(p.avg_cost)
                    pnl_pct = (last / cost - 1) * 100 if last and cost else None
                    pnl_tl = (last - cost) * p.quantity if last else None
                    pos_rows.append({
                        "Hisse": p.ticker.replace(".IS", ""), "Adet": p.quantity,
                        "Maliyet": f"{cost:.2f}",
                        "Güncel": f"{last:.2f}" if last else "—",
                        "K/Z %": f"%{pnl_pct:+.1f}" if pnl_pct is not None else "—",
                        "K/Z TL": f"{pnl_tl:+,.0f}" if pnl_tl is not None else "—",
                        "Stop": f"{float(p.stop):.2f}", "Hedef": f"{float(p.target):.2f}",
                        "Giriş seansı": p.opened_session,
                    })
                st.dataframe(pd.DataFrame(pos_rows), use_container_width=True,
                             hide_index=True)
            if acc.pending_orders:
                st.markdown("**Bugünün kararları — bekleyen emirler (ertesi açılışta dolar)**")
                st.dataframe(pd.DataFrame([{
                    "Hisse": o.ticker.replace(".IS", ""), "Yön": o.side,
                    "Adet": o.quantity, "Sebep": o.reason,
                } for o in acc.pending_orders]), use_container_width=True, hide_index=True)
            fills = [e for e in acc.ledger if e.get("event_type") in ("BUY", "SELL")]
            if fills:
                st.markdown("**Son hareketler (dolan emirler)**")
                st.dataframe(pd.DataFrame([{
                    "Seans": e.get("session", ""), "Yön": e.get("event_type"),
                    "Hisse": (e.get("ticker") or "").replace(".IS", ""),
                    "Fiyat": (e.get("metadata") or {}).get("fill_price", "—"),
                    "Tutar (TL)": f"{float(e.get('amount', 0)):+,.0f}",
                } for e in fills[-10:][::-1]]), use_container_width=True, hide_index=True)
            if not acc.positions and not acc.pending_orders and not fills:
                st.caption("Henüz pozisyon/emir yok — ilk seansı çalıştır.")

    # ── Observer karne ─────────────────────────────────────────────────────
    if state.predictions:
        with st.expander(f"🤖 ML Observer karnesi — {len(state.predictions)} tahmin"):
            st.caption("Para harcamaz; kalibre tahminleri biriktirir. Horizon (10g) "
                       "dolunca gerçekleşen fiyatla değerlendirilir. S4: meta-kapı "
                       "kararı (kaynak model + izin) da karneye girer.")
            recent = state.predictions[-15:]

            def _pct(v):
                return f"%{v*100:.0f}" if v is not None else "—"

            st.dataframe(pd.DataFrame([{
                "Seans": p["session"], "Hisse": p["ticker"].replace(".IS", ""),
                "p_up": _pct(p.get("p_up")), "Karar": p["decision"],
                "Meta p_win": _pct(p.get("meta_p_win")),
                "Meta": (p.get("meta_source") or "—") +
                        ("" if p.get("meta_allow", True) else " · VETO"),
            } for p in recent]), use_container_width=True, hide_index=True)


def render() -> None:
    st.title("🏟️ Paper Arena")
    st.caption("Eşit kasayla yarışan kâğıt hesaplar · XU100 edge kapısı · "
               "LLM yok, ücretsiz · Yatırım tavsiyesi değildir.")

    tab1, tab2, tab3 = st.tabs(["🎯 Edge Kapısı & Lig", "🔴 Canlı Sezon", "👥 Hesaplar"])
    with tab1:
        # Önceki sonucu hatırla (sayfa yenilenince kaybolmasın)
        cached = st.session_state.get("arena_replay")
        if cached is not None and cached.ok:
            with st.expander("↺ Son replay sonucunu göster", expanded=False):
                _show_replay_result(cached)
        _render_replay()
    with tab2:
        _render_live()
    with tab3:
        _render_profiles()

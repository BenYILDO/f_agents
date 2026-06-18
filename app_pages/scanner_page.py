"""📡 BIST 30 Tarayıcı — rasyo + teknik birleşik AL/SAT sinyalleri.

Saat başı zamanlayıcı (GitHub Actions) BIST 30'u sürekli analiz edip
``analysis_snapshots`` (scope='bist30') tablosuna yazar; bu ekran o en güncel
sonuçları gösterir — yani tarayıcıyı açmasan bile sinyaller arkada birikir.
"Şimdi tara" ile anlık yeniden hesaplanır. Supabase bağlı değilse de anlık
tarama çalışır (yalnız geçmişe yazılmaz).

Her hisse: birleşik karar (temel rasyo + teknik kompozit + dip stratejisi),
çoklu-yöntem **mutabakatı** (güven) ve güncel AL/SAT durumu.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from app_pages._styles import (
    AGREE_BADGE,
    STATUS_BADGE,
    agree_level_of,
    decision_rank,
)
from tradingagents.analysis import run as analysis_run
from tradingagents.analysis import trust
from tradingagents.analytics.cross_section import cross_sectional_score
from tradingagents.storage import snapshots
from tradingagents.storage.supabase_client import SupabaseError, is_configured
from tradingagents.strategy.dip_signal import BIST30


def _universe() -> list[str]:
    env = os.environ.get("BIST30_TICKERS", "").strip()
    if env:
        return [t.strip().upper() for t in env.split(",") if t.strip()]
    return list(BIST30)


def _age_text(ts_value) -> str:
    ts = trust._parse_ts(ts_value)
    if ts is None:
        return "—"
    mins = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    if mins < 1:
        return "az önce"
    if mins < 60:
        return f"{mins:.0f} dk önce"
    if mins < 24 * 60:
        return f"{mins / 60:.1f} saat önce"
    return f"{mins / 1440:.0f} gün önce"


def _scan_now(universe: list[str]) -> list[dict]:
    """BIST 30'u anlık analiz eder; snapshot dict'leri döndürür (+ yazmayı dener)."""
    rows: list[dict] = []
    prog = st.progress(0.0, text="BIST 30 taranıyor…")
    for i, tk in enumerate(universe, 1):
        outcome = analysis_run.analyze_ticker(tk)
        rows.append(analysis_run.to_snapshot_row(outcome, scope="bist30", source="manual"))
        prog.progress(i / len(universe), text=f"{tk} ({i}/{len(universe)})")
    prog.empty()
    if is_configured():
        try:
            snapshots.write_snapshots(rows)
            st.success(f"{len(rows)} hisse analiz edildi ve geçmişe yazıldı ✓")
        except SupabaseError as e:
            st.warning(f"Analiz yapıldı ama snapshot yazılamadı: {e}")
    else:
        st.info("Supabase bağlı değil — sonuçlar gösteriliyor ama geçmişe yazılmadı.")
    return rows


def _zscore_map(rows: list[dict]) -> dict[str, float]:
    """Evren için kesitsel birleşik z-skor (Faz D) — ticker→z."""
    if len(rows) < 3:
        return {}
    met = pd.DataFrame(
        [{"combined_score": r.get("combined_score"), "ratio_score": r.get("ratio_score"),
          "tech_score": r.get("tech_score")} for r in rows],
        index=[r["ticker"] for r in rows],
    )
    try:
        ranked = cross_sectional_score(
            met, {"combined_score": 1.0, "ratio_score": 1.0, "tech_score": 1.0})
        return ranked["composite_z"].to_dict()
    except Exception:  # noqa: BLE001
        return {}


def _rows_to_df(rows: list[dict], sort_by_z: bool = False) -> pd.DataFrame:
    zmap = _zscore_map(rows)
    if sort_by_z and zmap:
        ordered = sorted(rows, key=lambda x: -zmap.get(x["ticker"], -99))
    else:
        ordered = sorted(rows, key=lambda x: (decision_rank(x.get("decision")),
                                              -(x.get("combined_score") or 0)))
    out = []
    for r in ordered:
        ratio = r.get("ratio_verdict") or "—"
        rscore = r.get("ratio_score")
        conf = (r.get("signals") or {}).get("confidence") or {}
        score = conf.get("score")
        out.append({
            "Hisse": r["ticker"].replace(".IS", ""),
            "Kesitsel z": round(zmap[r["ticker"]], 2) if r["ticker"] in zmap else "—",
            "Durum": STATUS_BADGE.get(r.get("status"), "—"),
            "Karar (v3)": conf.get("gated") or r.get("decision", "—"),
            "Güven": f"{score:.0f} · {conf.get('grade','')}" if score is not None else "—",
            "Birleşik": round(r["combined_score"], 0) if r.get("combined_score") is not None else "—",
            "Rasyo": f"{ratio} ({rscore:.0f})" if rscore is not None else ratio,
            "Mutabakat": AGREE_BADGE.get(agree_level_of(r), "—"),
            "Fiyat (TRY)": r.get("close") if r.get("close") is not None else "—",
            "Güncellenme": _age_text(r.get("ts")),
        })
    return pd.DataFrame(out)


def render() -> None:
    st.title("📡 BIST 30 Tarayıcı")
    st.caption("Rasyo + teknik birleşik AL/SAT sinyalleri · saat başı otomatik "
               "güncellenir · Yatırım tavsiyesi değildir.")

    from tradingagents.analytics import market_calendar as mcal
    ms = mcal.market_status()
    st.caption(f"{'🟢' if ms.open else '🔴'} **BIST {ms.status}** — {ms.detail}")

    universe = _universe()
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        scan = st.button(f"🔄 Şimdi tara ({len(universe)} hisse)",
                         type="primary", use_container_width=True)
    with c2:
        only_buy = st.toggle("Sadece AL sinyalleri", value=False)
    with c3:
        only_agree = st.toggle("Sadece mutabakatlı", value=False,
                               help="Yöntemler çelişen/nötr olanları gizle")

    if scan:
        rows = _scan_now(universe)
    elif is_configured():
        try:
            rows = snapshots.latest("bist30")
        except SupabaseError as e:
            st.error(f"Snapshot okunamadı: {e}")
            rows = []
        if not rows:
            st.info("Henüz snapshot yok. **Şimdi tara**'ya bas ya da saat başı "
                    "zamanlayıcının ilk çalışmasını bekle.")
            return
    else:
        st.warning("Supabase bağlı değil — kayıtlı sonuç yok. **Şimdi tara** ile "
                   "anlık tarama yapabilirsin (geçmişe yazılmaz). "
                   "Kurulum: docs/SUPABASE_SETUP.md")
        return

    # Tazelik (güven): en güncel snapshot ne kadar eski?
    newest = max((trust._parse_ts(r.get("ts")) for r in rows
                  if trust._parse_ts(r.get("ts"))), default=None)
    if newest is not None:
        age_min = (datetime.now(timezone.utc) - newest).total_seconds() / 60
        if age_min > 120:
            st.warning(f"⚠️ Veriler {_age_text(newest)} güncellendi — zamanlayıcı "
                       "gecikmiş olabilir. Tazelemek için **Şimdi tara**.")
        else:
            st.caption(f"🩺 En son güncelleme: {_age_text(newest)}")

    buys = [r for r in rows if r.get("status") == "AL"]
    if buys:
        st.success(f"🟢 **{len(buys)}** hisse AL bölgesinde: "
                   + ", ".join(sorted(r["ticker"].replace(".IS", "") for r in buys)))

    view = rows
    if only_buy:
        view = [r for r in view if r.get("status") == "AL"
                or r.get("decision") in ("GÜÇLÜ AL", "AL")]
    if only_agree:
        view = [r for r in view if agree_level_of(r) in ("güçlü", "kısmi")]

    if not view:
        st.caption("Filtreye uyan hisse yok.")
        return

    sort_z = st.toggle("📐 Kesitsel z-skora göre sırala (evreni güce göre diz)",
                       value=False, help="Faz D: winsorize + z-skor birleşik sıralama")
    st.dataframe(_rows_to_df(view, sort_by_z=sort_z),
                 use_container_width=True, hide_index=True)
    st.caption("Varsayılan sıra: en boğa karardan en ayıya. 'Kesitsel z' = evren "
               "içinde göreli güç (yüksek=iyi). 'Mutabakat' = teknik+rasyo+dip+teyit "
               "uyumu. Çelişki → temkin.")

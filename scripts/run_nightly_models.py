"""Gecelik model işi (Faz H + S2) — kalibre olasılık + DSR + havuz modeli.

Ağır hesaplar (5-kat GBDT kalibrasyonu) burada, günde bir kez yapılır; saatlik
iş yalnız sonucu okur → saatlik cron hafif kalır. Portföy + BIST 30 evrenini işler.

S2 havuz adımı: tüm evren tek panelde eğitilir (üçlü-bariyer etiket, purged
walk-forward), artefakt ``pooled_models`` tablosuna yazılır ve ticker başına
challenger tahmini ``model_cache.p_up_pooled``'a konur — per-ticker şampiyonun
davranışını DEĞİŞTİRMEZ (champion/challenger).

Kullanım:
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python scripts/run_nightly_models.py
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tradingagents.analytics.backtest import daily_strategy_returns  # noqa: E402
from tradingagents.analytics.composite import _fetch_daily  # noqa: E402
from tradingagents.analytics.probability import calibrated_probability  # noqa: E402
from tradingagents.analytics.significance import deflated_sharpe_ratio  # noqa: E402
from tradingagents.storage import model_cache, portfolio  # noqa: E402
from tradingagents.storage.supabase_client import SupabaseError, is_configured  # noqa: E402
from tradingagents.strategy.dip_signal import BIST30, compute_signals  # noqa: E402


def _universe() -> list[str]:
    env = os.environ.get("BIST30_TICKERS", "").strip()
    bist = [t.strip().upper() for t in env.split(",") if t.strip()] if env else list(BIST30)
    try:
        port = [h["ticker"].upper() for h in portfolio.list_holdings()]
    except SupabaseError:
        port = []
    return sorted(set(bist) | set(port))


def _model_row(ticker: str, benchmark=None) -> tuple[dict | None, object]:
    """(model_cache satırı | None, günlük OHLCV df) — df havuz adımında yeniden
    kullanılır (ikinci kez indirilmez)."""
    df = _fetch_daily(ticker)
    if df is None or df.empty:
        return None, None
    pr = calibrated_probability(ticker, df, horizon=10, benchmark=benchmark)
    dsr = None
    try:
        sig = compute_signals(df)
        strat = daily_strategy_returns(sig["Close"], sig["buy"].astype(bool),
                                       sig["sell"].astype(bool))
        if (strat != 0).sum() > 30:
            dsr = round(deflated_sharpe_ratio(strat, n_trials=20, sr_variance=0.5), 3)
    except Exception:  # noqa: BLE001
        pass
    if not pr.ok and dsr is None:
        return None, df
    return {
        "ticker": ticker,
        # eski/temel kolonlar (geriye uyumlu)
        "p_up": pr.p_up, "brier": pr.brier, "auc": pr.auc,
        "horizon": pr.horizon, "n_samples": pr.n_samples, "dsr": dsr,
        # F0.2 zengin kanıt — model kalitesi/red nedenleri snapshot'a kadar taşınır
        "brier_raw": pr.brier_raw, "brier_calibrated": pr.brier_calibrated,
        "brier_skill_score": pr.brier_skill_score, "recent_skill": pr.recent_skill,
        "n_calibration": pr.n_calibration, "n_test": pr.n_test,
        "quality_passed": pr.quality_passed, "rejection_reasons": pr.rejection_reasons,
        "model_version": pr.model_version, "trained_until": pr.trained_until or None,
    }, df


def _pooled_step(frames: dict, benchmark) -> None:
    """S2: havuz modelini eğit, artefaktı Supabase'e yaz, challenger tahminlerini
    model_cache'in pooled kolonlarına koy. Başarısızlık gecelik işi KIRMAZ."""
    try:
        from tradingagents.ml.pooled import (
            predict_pooled,
            serialize_bundle,
            train_pooled_model,
        )
        from tradingagents.storage import pooled_models

        print(f"  Havuz modeli eğitiliyor ({len(frames)} hisse, tek panel)…",
              flush=True)
        res = train_pooled_model(frames, benchmark)
        if not res.ok:
            print(f"  ! Havuz modeli eğitilemedi: {res.reason}", flush=True)
            return
        print(f"    karne: AUC={res.auc} BSS={res.brier_skill_score} "
              f"n={res.n_samples} test={res.n_test} kalite="
              f"{'GEÇTİ' if res.quality_passed else 'GEÇEMEDİ'} "
              f"({', '.join(res.rejection_reasons) or '—'})", flush=True)

        pooled_models.save_model({
            "model_version": res.model_version,
            "trained_until": res.trained_until, "horizon": res.horizon,
            "threshold": res.threshold, "universe_size": res.n_tickers,
            "universe": res.universe, "n_samples": res.n_samples,
            "n_test": res.n_test, "auc": res.auc, "brier_raw": res.brier_raw,
            "brier_calibrated": res.brier_calibrated,
            "brier_skill_score": res.brier_skill_score,
            "quality_passed": res.quality_passed,
            "rejection_reasons": res.rejection_reasons,
            "feature_importance": res.feature_importance,
            "artifact": serialize_bundle(res),
        })
        preds = predict_pooled(
            {"model": res.model, "calibrator": res.calibrator,
             "medians": res.medians}, frames, benchmark)
        if preds:
            model_cache.upsert_models([
                {"ticker": tk, "p_up_pooled": p,
                 "pooled_version": res.model_version,
                 "pooled_quality": res.quality_passed}
                for tk, p in preds.items()])
            print(f"    {len(preds)} challenger tahmini model_cache'e yazıldı.",
                  flush=True)
    except SupabaseError as exc:
        print(f"  ! Havuz modeli yazılamadı ({exc}). pooled_models tablosu için "
              "schema.sql'i yeniden çalıştır.", flush=True)
    except Exception as exc:  # noqa: BLE001 — challenger adımı şampiyonu kırmasın
        print(f"  ! Havuz adımı hata: {exc}", flush=True)


# Migrate edilmemiş model_cache tablosu (yeni kolonlar yok) için güvenli alt küme.
_LEGACY_COLS = {"ticker", "p_up", "brier", "auc", "horizon", "n_samples", "dsr"}


def _legacy_row(row: dict) -> dict:
    """Yeni kolonları olmayan eski şema için satırı temel kolonlara indirger."""
    return {k: v for k, v in row.items() if k in _LEGACY_COLS}


def main() -> int:
    print("== Gecelik model işi ==", flush=True)
    if not is_configured():
        print("  Supabase yapılandırılmamış — atlanıyor.", flush=True)
        return 0
    started = time.time()
    universe = _universe()
    print(f"  {len(universe)} hisse işleniyor…", flush=True)
    # XU100 bir kez çekilir; S1 üçlü-bariyer etiketi endeks-relatif ölçülür.
    xu = _fetch_daily("XU100.IS")
    benchmark = xu["Close"] if xu is not None and not xu.empty else None
    if benchmark is None:
        print("  ! XU100 alınamadı — etiketler mutlak getiriyle (relatif değil).",
              flush=True)
    rows = []
    frames: dict = {}
    for i, tk in enumerate(universe, 1):
        row, df = _model_row(tk, benchmark)
        if df is not None:
            frames[tk] = df
        if row:
            rows.append(row)
            print(f"    {tk:<12} p_up={row['p_up']} dsr={row['dsr']} ({i}/{len(universe)})",
                  flush=True)
        else:
            print(f"    {tk:<12} atlandı ({i}/{len(universe)})", flush=True)
    try:
        model_cache.upsert_models(rows)
    except SupabaseError as exc:
        # Büyük olasılıkla model_cache tablosu henüz F0.2 kolonlarıyla migrate
        # edilmemiş; temel kolonlarla tekrar dene (kalite metrikleri yazılmaz).
        print(f"  ! Zengin kanıt yazılamadı ({exc}). schema.sql'i yeniden çalıştır; "
              "şimdilik temel kolonlarla yazılıyor.", flush=True)
        try:
            model_cache.upsert_models([_legacy_row(r) for r in rows])
        except SupabaseError as exc2:
            print(f"  ! Yazılamadı: {exc2}", flush=True)
            return 1
    # S2 — havuz (pooled) challenger: per-ticker şampiyon yazıldıktan sonra
    _pooled_step(frames, benchmark)
    print(f"== Bitti · {len(rows)} model yazıldı · {time.time()-started:.0f}s ==", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Gecelik model işi (Faz H) — kalibre olasılık + DSR'yi Supabase'e yazar.

Ağır hesaplar (5-kat GBDT kalibrasyonu) burada, günde bir kez yapılır; saatlik
iş yalnız sonucu okur → saatlik cron hafif kalır. Portföy + BIST 30 evrenini işler.

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


def _model_row(ticker: str) -> dict | None:
    df = _fetch_daily(ticker)
    if df is None or df.empty:
        return None
    pr = calibrated_probability(ticker, df, horizon=10)
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
        return None
    return {
        "ticker": ticker,
        "p_up": pr.p_up, "brier": pr.brier, "auc": pr.auc,
        "horizon": pr.horizon, "n_samples": pr.n_samples, "dsr": dsr,
    }


def main() -> int:
    print("== Gecelik model işi ==", flush=True)
    if not is_configured():
        print("  Supabase yapılandırılmamış — atlanıyor.", flush=True)
        return 0
    started = time.time()
    universe = _universe()
    print(f"  {len(universe)} hisse işleniyor…", flush=True)
    rows = []
    for i, tk in enumerate(universe, 1):
        row = _model_row(tk)
        if row:
            rows.append(row)
            print(f"    {tk:<12} p_up={row['p_up']} dsr={row['dsr']} ({i}/{len(universe)})",
                  flush=True)
        else:
            print(f"    {tk:<12} atlandı ({i}/{len(universe)})", flush=True)
    try:
        model_cache.upsert_models(rows)
    except SupabaseError as exc:
        print(f"  ! Yazılamadı: {exc}", flush=True)
        return 1
    print(f"== Bitti · {len(rows)} model yazıldı · {time.time()-started:.0f}s ==", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Saat başı deterministik analiz işi — GitHub Actions zamanlayıcısı çağırır.

Portföydeki (Supabase ``holdings``) ve BIST 30 evrenindeki her hisse için
LLM'siz/ücretsiz analizi (teknik + rasyo + dip stratejisi + güven) çalıştırır ve
``analysis_snapshots`` tablosuna yazar. Böylece geçmiş birikir; Portföyüm/Otomatik
Analiz ekranları ve güven katmanı (karne/istikrar) bu geçmişe dayanır.

Kullanım:
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python scripts/run_hourly_analysis.py
    python scripts/run_hourly_analysis.py --scope portfolio
    python scripts/run_hourly_analysis.py --prune --retain-days 90

Yalnız deterministik motorları kullanır — OpenAI/LLM anahtarı GEREKMEZ, kredi
harcamaz. Hisse başı hatalar tolere edilir (o hisse için 'veri yok' snapshot'ı
yine yazılır ki veri tazeliği izlenebilsin).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Bu script repo kökünden ya da scripts/ içinden çağrılsa da tradingagents
# paketini bulabilsin diye repo kökünü yola ekle (kurulum gerektirmez).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tradingagents.analysis import run as analysis_run  # noqa: E402
from tradingagents.storage import portfolio, snapshots  # noqa: E402
from tradingagents.storage.supabase_client import (  # noqa: E402
    SupabaseError,
    SupabaseREST,
    is_configured,
)
from tradingagents.strategy.dip_signal import BIST30  # noqa: E402


def _bist30_universe() -> list[str]:
    env = os.environ.get("BIST30_TICKERS", "").strip()
    if env:
        return [t.strip().upper() for t in env.split(",") if t.strip()]
    return list(BIST30)


def _portfolio_tickers() -> list[str]:
    try:
        holdings = portfolio.list_holdings()
    except SupabaseError as exc:
        print(f"  ! Portföy okunamadı: {exc}", flush=True)
        return []
    return sorted({h["ticker"].upper() for h in holdings})


def _run_scope(scope: str, tickers: list[str], source: str) -> tuple[int, int]:
    """Bir evreni analiz edip snapshot'ları yazar. (yazılan, AL_sinyali) döndürür."""
    if not tickers:
        print(f"  [{scope}] hisse yok, atlandı.", flush=True)
        return 0, 0
    print(f"  [{scope}] {len(tickers)} hisse analiz ediliyor…", flush=True)
    rows, buys = [], 0
    for tk in tickers:
        outcome = analysis_run.analyze_ticker(tk)
        rows.append(analysis_run.to_snapshot_row(outcome, scope=scope, source=source))
        flag = ""
        if outcome.ok and outcome.status == "AL":
            buys += 1
            flag = "  🟢 AL"
        elif not outcome.ok:
            flag = f"  ⚠️ {outcome.error}"
        print(f"    {tk:<12} {outcome.decision:<10} "
              f"({outcome.agreement_level}){flag}", flush=True)
    try:
        snapshots.write_snapshots(rows)
    except SupabaseError as exc:
        print(f"  ! [{scope}] snapshot yazılamadı: {exc}", flush=True)
        return 0, buys
    return len(rows), buys


def main() -> int:
    parser = argparse.ArgumentParser(description="Saat başı BIST analiz işi")
    parser.add_argument("--scope", choices=["all", "portfolio", "bist30"], default="all")
    parser.add_argument("--prune", action="store_true",
                        help="Eski snapshot'ları seyrelt (retention)")
    parser.add_argument("--retain-days", type=int, default=90)
    args = parser.parse_args()

    started = time.time()
    print(f"== Saat başı analiz · scope={args.scope} ==", flush=True)

    # Henüz Supabase yapılandırılmadıysa (ilk kurulumdan önce) işi kırmızı
    # göstermek yerine sessizce çık — gerçek bir hata değil, eksik kurulum.
    if not is_configured():
        print("  Supabase yapılandırılmamış (SUPABASE_URL/SUPABASE_SERVICE_KEY yok) "
              "— atlanıyor. Kurulum: docs/SUPABASE_SETUP.md", flush=True)
        return 0

    total_written = 0
    if args.scope in ("all", "portfolio"):
        written, buys = _run_scope("portfolio", _portfolio_tickers(), source="cron")
        total_written += written
        print(f"  [portfolio] {written} snapshot, {buys} AL sinyali", flush=True)
    if args.scope in ("all", "bist30"):
        written, buys = _run_scope("bist30", _bist30_universe(), source="cron")
        total_written += written
        print(f"  [bist30] {written} snapshot, {buys} AL sinyali", flush=True)

    if args.prune:
        try:
            deleted = SupabaseREST().rpc("prune_old_snapshots",
                                         {"retain_days": args.retain_days})
            print(f"  retention: {deleted} eski satır seyreltildi "
                  f"(>{args.retain_days} gün)", flush=True)
        except SupabaseError as exc:
            print(f"  ! retention atlandı: {exc}", flush=True)

    elapsed = time.time() - started
    print(f"== Bitti · {total_written} snapshot · {elapsed:.0f}s ==", flush=True)
    # Hiç snapshot yazılamadıysa (örn. Supabase down) işi kırmızı göster.
    return 0 if total_written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

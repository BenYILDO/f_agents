"""Canlı arena günlük seansı — bugünün sinyalleriyle emir üret/doldur/equity yaz.

Her gün bir kez çalışır (terminal ya da GitHub Actions cron). İdempotenttir:
aynı seans iki kez işlenmez. Yerel JSON durumunu günceller (Supabase opsiyonel).

Kullanım:
    python scripts/run_arena_session.py
    python scripts/run_arena_session.py --tickers GARAN.IS,THYAO.IS,ASELS.IS
    python scripts/run_arena_session.py --reset      # sezonu sıfırla
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tradingagents.arena.config import DEFAULT_EXECUTION  # noqa: E402
from tradingagents.arena.live import build_session_inputs, run_session  # noqa: E402
from tradingagents.arena.state import D, load_state, new_state, save_state  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Canlı arena günlük seansı")
    ap.add_argument("--tickers", default="", help="Virgüllü liste; boşsa BIST30")
    ap.add_argument("--reset", action="store_true", help="Sezonu sıfırla ve çık")
    args = ap.parse_args()

    if args.reset:
        save_state(new_state())
        print("Sezon sıfırlandı (tüm kasalar 100k).", flush=True)
        return 0

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    state = load_state() or new_state()
    print(f"== Canlı arena seansı · sezon {state.season_id} · son {state.last_session or '—'} ==",
          flush=True)

    print("  Bugünün sinyalleri üretiliyor (analyze_universe)…", flush=True)
    outcomes, prices, session = build_session_inputs(tickers)
    report = run_session(state, outcomes, prices, session, DEFAULT_EXECUTION)
    save_state(state)

    if report.skipped:
        print(f"  Seans {session} zaten işlenmiş (idempotent).", flush=True)
        return 0

    print(f"  Seans {session}: {report.filled} emir doldu · {report.new_orders} "
          f"yeni emir · {report.predictions} observer tahmini.\n", flush=True)
    init = D(DEFAULT_EXECUTION.initial_capital)
    for code, acc in state.accounts.items():
        last_eq = D(acc.equity_history[-1]["equity"]) if acc.equity_history else acc.cash
        ret = (last_eq / init - 1) * 100 if init else 0
        tag = "OBS" if acc.status == "OBSERVER" else "   "
        print(f"  {tag} {code:<14} equity {float(last_eq):>12,.0f}  "
              f"({float(ret):+.1f}%)  poz {len(acc.positions)}  bekleyen {len(acc.pending_orders)}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

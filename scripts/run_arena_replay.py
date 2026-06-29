"""Arena edge kapısı — CLI replay (terminalde hızlı doğrulama).

Planın "fail-cheap" kapısını terminalden çalıştırır: profiller maliyet sonrası
XU100'ü geçiyor mu? Supabase GEREKMEZ; tamamen yerel + ücretsiz.

Kullanım:
    python scripts/run_arena_replay.py
    python scripts/run_arena_replay.py --period 10y
    python scripts/run_arena_replay.py --tickers GARAN.IS,THYAO.IS,ASELS.IS
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tradingagents.arena.replay import run_arena_replay  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper Arena edge kapısı (replay)")
    ap.add_argument("--period", default="5y", help="Tarihsel pencere (3y/5y/10y)")
    ap.add_argument("--tickers", default="", help="Virgüllü liste; boşsa BIST30")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None

    def _progress(frac, text):
        bar = "█" * int(frac * 30)
        print(f"\r  [{bar:<30}] {text:<40}", end="", flush=True)

    print(f"== Paper Arena edge kapısı · pencere={args.period} ==", flush=True)
    result = run_arena_replay(tickers=tickers, period=args.period, progress=_progress)
    print()

    if not result.ok:
        print(f"  ! Başarısız: {result.error}", flush=True)
        return 1

    bm = result.benchmark_metrics
    print(f"\n  XU100 (al-tut): getiri %{bm.total_return*100:+.1f} · "
          f"Sharpe {bm.sharpe:.2f} · maks DD %{bm.max_drawdown*100:.1f}\n")
    print(f"  {'Hesap':<16}{'Getiri':>10}{'vsXU100':>10}{'Sharpe':>8}{'MaksDD':>9}"
          f"{'İşlem':>7}{'XU100>?':>9}")
    print("  " + "-" * 69)
    for x in result.leaderboard:
        flag = "✅" if x["beats_benchmark"] else "—"
        print(f"  {x['name']:<16}{x['total_return']*100:>+9.1f}%"
              f"{x['alpha_vs_xu100']*100:>+9.1f}%{x['sharpe']:>8.2f}"
              f"{x['max_drawdown']*100:>+8.1f}%{x['n_trades']:>7}{flag:>9}")

    print(f"\n  {result.edge_summary}\n")
    print("  Sınırlamalar:")
    for c in result.caveats:
        print(f"    · {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

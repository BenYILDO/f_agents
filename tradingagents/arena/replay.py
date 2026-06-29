"""Tarihsel edge kapısı — profiller maliyet sonrası XU100'ü geçiyor mu?

Planın "fail-cheap" kararı (Codex): arena için Supabase/UI/cron yazmadan önce
**önce kâr var mı?** Bu modül o soruyu yanıtlar: her aktif profili 5 yıllık BIST
verisinde, ortak execution fiziğiyle simüle eder ve XU100 al-tut ile kıyaslar.

Dürüst sınırlamalar (UI'da gösterilir):
  - **Survivorship:** bugünkü BIST30 listesi geçmişe uygulanır (dönemsel üyelik yok).
  - **Rejim proxy'si:** boğa = XU100 200 günlük ortalamanın üstünde (HMM değil, nedensel).
  - **Sinyal:** dip-stratejisi nedensel buy/sell (compute_signals) + ATR stop/hedef.
  - **Güven/Kelly/p_up:** canlı kavramlar; tarihsel her gün yeniden hesaplanmaz.
  - ML-öncelikli hesap OBSERVER olduğundan tarihsel para replay'ine girmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.arena.config import DEFAULT_EXECUTION, ExecutionConfig
from tradingagents.arena.engine import BacktestResult, run_backtest
from tradingagents.arena.metrics import PerfMetrics, benchmark_buy_hold, compute_metrics
from tradingagents.arena.profiles import active_profiles

_ATR_LEN = 14
_ATR_STOP_MULT = 2.0
_TARGET_RR = 2.0
_TREND_MA = 50
_REGIME_MA = 200


@dataclass
class ArenaReplayResult:
    ok: bool = False
    per_profile: dict = field(default_factory=dict)        # code -> BacktestResult
    benchmark_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_metrics: PerfMetrics = field(default_factory=PerfMetrics)
    leaderboard: list = field(default_factory=list)        # sıralı [{...}]
    edge_gate_passed: bool = False
    edge_summary: str = ""
    n_tickers: int = 0
    period: str = ""
    caveats: list = field(default_factory=list)
    error: str = ""


def _atr(df: pd.DataFrame, n: int = _ATR_LEN) -> pd.Series:
    """Wilder ATR yaklaşımı (basit rolling mean of true range)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift()
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n // 2)).mean()


def _prepare_ticker(df: pd.DataFrame) -> pd.DataFrame | None:
    """OHLCV → buy/sell + per-bar stop/target + trend_ok. Hata → None."""
    from tradingagents.strategy.dip_signal import compute_signals
    if df is None or len(df) < _REGIME_MA + _TREND_MA:
        return None
    try:
        sig = compute_signals(df)
    except Exception:  # noqa: BLE001
        return None
    atr = _atr(sig)
    sig["stop"] = sig["Close"] - _ATR_STOP_MULT * atr
    sig["target"] = sig["Close"] + _ATR_STOP_MULT * _TARGET_RR * atr
    sig["trend_ok"] = sig["Close"] > sig["Close"].rolling(_TREND_MA).mean()
    keep = ["Open", "High", "Low", "Close", "buy", "sell", "stop", "target", "trend_ok"]
    return sig[keep].copy()


def run_arena_replay(
    tickers: list[str] | None = None,
    period: str = "5y",
    cfg: ExecutionConfig = DEFAULT_EXECUTION,
    progress=None,
) -> ArenaReplayResult:
    """Aktif profilleri tarihsel veride koşturup XU100 ile kıyaslar.

    ``progress(frac, text)`` opsiyonel geri-çağrısı UI ilerleme çubuğu içindir.
    Asla istisna fırlatmaz; başarısız ticker'lar atlanır.
    """
    from tradingagents.analytics.composite import _fetch_daily
    from tradingagents.strategy.dip_signal import BIST30

    universe = tickers or list(BIST30)

    def _tick(frac, text):
        if progress is not None:
            try:
                progress(frac, text)
            except Exception:  # noqa: BLE001
                pass

    # ── Benchmark (XU100) — ana takvim de buradan ──────────────────────────
    _tick(0.02, "XU100 endeksi çekiliyor…")
    xu = _fetch_daily(cfg.benchmark_ticker, period=period)
    if xu is None or len(xu) < _REGIME_MA + 10:
        return ArenaReplayResult(ok=False, error="XU100 verisi alınamadı.")
    xu = xu[~xu.index.duplicated(keep="last")].sort_index()
    dates = xu.index
    regime = (xu["Close"] > xu["Close"].rolling(_REGIME_MA).mean())

    # ── Hisse verisi + sinyaller ───────────────────────────────────────────
    data: dict[str, pd.DataFrame] = {}
    for j, tk in enumerate(universe, 1):
        _tick(0.05 + 0.55 * j / len(universe), f"{tk} ({j}/{len(universe)})")
        try:
            df = _fetch_daily(tk, period=period)
        except Exception:  # noqa: BLE001
            df = None
        prepared = _prepare_ticker(df) if df is not None else None
        if prepared is not None and len(prepared) > _REGIME_MA:
            data[tk] = prepared
    if not data:
        return ArenaReplayResult(ok=False, error="Hiç hisse verisi hazırlanamadı.")

    # ── Benchmark equity ───────────────────────────────────────────────────
    bench_eq = benchmark_buy_hold(xu["Close"], cfg.initial_capital, cfg.slippage_bps)
    bench_metrics = compute_metrics(bench_eq)

    # ── Her aktif profili koştur ───────────────────────────────────────────
    profiles = active_profiles()
    per_profile: dict[str, BacktestResult] = {}
    for k, prof in enumerate(profiles, 1):
        _tick(0.6 + 0.35 * k / len(profiles), f"Profil: {prof.name}…")
        per_profile[prof.code] = run_backtest(prof, data, dates, cfg, regime)

    # ── Lig tablosu + edge kapısı ──────────────────────────────────────────
    bench_ret = bench_metrics.total_return
    leaderboard = []
    for prof in profiles:
        r = per_profile[prof.code]
        alpha = r.metrics.total_return - bench_ret
        beats = (r.metrics.total_return > bench_ret) and (r.metrics.sharpe >= bench_metrics.sharpe)
        leaderboard.append({
            "code": prof.code, "name": prof.name, "emoji": prof.emoji,
            "final_equity": r.final_equity, "total_return": r.metrics.total_return,
            "cagr": r.metrics.cagr, "sharpe": r.metrics.sharpe,
            "max_drawdown": r.metrics.max_drawdown, "n_trades": r.n_trades,
            "win_rate": r.win_rate, "alpha_vs_xu100": round(alpha, 4),
            "beats_benchmark": beats,
        })
    leaderboard.sort(key=lambda x: x["total_return"], reverse=True)

    winners = [x for x in leaderboard if x["beats_benchmark"]]
    edge_passed = len(winners) > 0
    if edge_passed:
        best = winners[0]
        edge_summary = (
            f"✅ EDGE KAPISI GEÇİLDİ — {best['emoji']} {best['name']} maliyet sonrası "
            f"XU100'ü getiri (%{best['total_return']*100:+.1f} vs %{bench_ret*100:+.1f}) "
            f"ve Sharpe'ta geçti. Arena altyapısına yatırım gerekçeli."
        )
    else:
        best = leaderboard[0] if leaderboard else None
        edge_summary = (
            "❌ EDGE KAPISI GEÇİLMEDİ — hiçbir profil maliyet sonrası XU100'ü hem "
            "getiri hem Sharpe'ta geçemedi. Plan gereği arena altyapısına geçmeden "
            "önce sinyal/parametre iyileştirilmeli (fail-cheap)."
            + (f" En iyi: {best['name']} %{best['total_return']*100:+.1f} "
               f"(XU100 %{bench_ret*100:+.1f})." if best else "")
        )

    caveats = [
        "Survivorship: bugünkü BIST30 listesi geçmişe uygulandı (dönemsel üyelik yok).",
        "Rejim filtresi = XU100 200 günlük ortalama proxy'si (HMM değil).",
        "Sinyal = dip-stratejisi nedensel buy/sell + ATR(14) stop/hedef.",
        "Güven skoru/Kelly/p_up canlı kavramlar; tarihsel replay'de uygulanmadı.",
        "ML-öncelikli hesap OBSERVER — tarihsel para replay'ine girmez.",
        f"Ortak fizik: {cfg.cost_summary()}.",
    ]

    return ArenaReplayResult(
        ok=True, per_profile=per_profile, benchmark_equity=bench_eq,
        benchmark_metrics=bench_metrics, leaderboard=leaderboard,
        edge_gate_passed=edge_passed, edge_summary=edge_summary,
        n_tickers=len(data), period=period, caveats=caveats,
    )

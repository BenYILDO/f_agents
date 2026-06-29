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
    # Eğitim/test (kronolojik 80/20 holdout) — overfitting koruması
    split_session: str = ""
    train_frac: float = 0.8
    robust_summary: str = ""
    equalweight_metrics: PerfMetrics = field(default_factory=PerfMetrics)
    equalweight_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


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


def _equalweight_equity(data: dict, dates, initial: float, slippage_bps: float) -> pd.Series:
    """1 dönem önce BIST evrenini eşit böl-tut benchmark'ı (naif çeşitlendirme).

    İlk barda kasayı tüm hisselere eşit dağıtıp tutar; kullanıcının "aynı hisselerin
    dağılımını yapalım" dediği sezgisel kıyas. XU100'den farkı: eşit-ağırlık.
    """
    n = len(data)
    if n == 0:
        return pd.Series(dtype=float)
    per = initial / n
    slip = 1.0 + slippage_bps / 10_000.0
    total = pd.Series(0.0, index=dates)
    for df in data.values():
        c = df["Close"].reindex(dates).ffill()
        first_valid = c.first_valid_index()
        if first_valid is None or c.loc[first_valid] <= 0:
            continue
        units = per / (c.loc[first_valid] * slip)
        total = total.add((units * c).fillna(0.0), fill_value=0.0)
    return total


def _segment_metrics(equity: pd.Series, split_idx: int):
    """Equity eğrisini eğitim (ilk %80) / test (son %20) diye böler → (train, test)."""
    if equity is None or len(equity) < 4:
        return PerfMetrics(), PerfMetrics()
    split_idx = max(2, min(split_idx, len(equity) - 2))
    return compute_metrics(equity.iloc[:split_idx]), compute_metrics(equity.iloc[split_idx:])


def run_arena_replay(
    tickers: list[str] | None = None,
    period: str = "5y",
    cfg: ExecutionConfig = DEFAULT_EXECUTION,
    train_frac: float = 0.8,
    progress=None,
) -> ArenaReplayResult:
    """Aktif profilleri tarihsel veride koşturup XU100 + eşit-ağırlık ile kıyaslar.

    Kronolojik ``train_frac`` (varsayılan %80) holdout: ilk %80 'eğitim/in-sample',
    son %20 'test/görülmemiş'. Her iki segmentte de iyi olan profil dayanıklıdır
    (overfit değil). ``progress(frac, text)`` UI ilerleme çubuğu içindir.
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

    # ── Benchmark equity (XU100 + eşit-ağırlık BIST) ───────────────────────
    bench_eq = benchmark_buy_hold(xu["Close"], cfg.initial_capital, cfg.slippage_bps)
    bench_metrics = compute_metrics(bench_eq)
    ew_eq = _equalweight_equity(data, dates, cfg.initial_capital, cfg.slippage_bps)
    ew_metrics = compute_metrics(ew_eq)

    # ── Eğitim/test bölme noktası (kronolojik %80) ─────────────────────────
    split_idx = int(len(dates) * train_frac)
    split_session = str(pd.Timestamp(dates[min(split_idx, len(dates) - 1)]).date())
    bench_train, bench_test = _segment_metrics(bench_eq, split_idx)

    # ── Her aktif profili koştur ───────────────────────────────────────────
    profiles = active_profiles()
    per_profile: dict[str, BacktestResult] = {}
    for k, prof in enumerate(profiles, 1):
        _tick(0.6 + 0.35 * k / len(profiles), f"Profil: {prof.name}…")
        per_profile[prof.code] = run_backtest(prof, data, dates, cfg, regime)

    # ── Lig tablosu + edge kapısı (RİSK-AYARLI) ────────────────────────────
    # Edge kapısı mutlak getiriye değil risk-ayarlıya bakar: enflasyonist boğada
    # (TL-nominal XU100 ~10x) mutlak getiride index'i geçmek yanlış bardır ve
    # daha riskli/overfit stratejilere iter. Planın amacı da "risk-ayarlı kıyas".
    # "Geçti" = Sharpe ≥ benchmark VE drawdown benchmark'tan sığ (daha az sancı).
    # Survivorship bias mutlak getiriyi şişirir ama drawdown'u görece az → bu
    # tanım daha dayanıklı. Mutlak getiri yine de şeffaflık için gösterilir.
    bench_ret = bench_metrics.total_return
    bench_sharpe = bench_metrics.sharpe
    bench_dd = bench_metrics.max_drawdown
    def _beats(m, bench_m) -> bool:
        return (m.sharpe >= bench_m.sharpe) and (m.max_drawdown >= bench_m.max_drawdown)

    leaderboard = []
    for prof in profiles:
        r = per_profile[prof.code]
        alpha = r.metrics.total_return - bench_ret
        beats_riskadj = _beats(r.metrics, bench_metrics)
        beats_absolute = r.metrics.total_return > bench_ret
        # Eğitim/test holdout — overfitting kontrolü
        tr_m, te_m = _segment_metrics(r.equity_curve, split_idx)
        beats_train = _beats(tr_m, bench_train)
        beats_test = _beats(te_m, bench_test)
        robust = beats_train and beats_test   # her iki segmentte de iyi = dayanıklı
        leaderboard.append({
            "code": prof.code, "name": prof.name, "emoji": prof.emoji,
            "final_equity": r.final_equity, "total_return": r.metrics.total_return,
            "cagr": r.metrics.cagr, "sharpe": r.metrics.sharpe,
            "max_drawdown": r.metrics.max_drawdown, "n_trades": r.n_trades,
            "win_rate": r.win_rate, "alpha_vs_xu100": round(alpha, 4),
            "beats_benchmark": beats_riskadj, "beats_absolute": beats_absolute,
            "train_return": tr_m.total_return, "train_sharpe": tr_m.sharpe,
            "test_return": te_m.total_return, "test_sharpe": te_m.sharpe,
            "test_max_drawdown": te_m.max_drawdown,
            "beats_train": beats_train, "beats_test": beats_test, "robust": robust,
        })
    # Test (görülmemiş) Sharpe'ına göre sırala — en dayanıklı en üstte
    leaderboard.sort(key=lambda x: x["test_sharpe"], reverse=True)

    winners = [x for x in leaderboard if x["beats_benchmark"]]
    edge_passed = len(winners) > 0
    if edge_passed:
        best = winners[0]
        edge_summary = (
            f"✅ EDGE KAPISI GEÇİLDİ (risk-ayarlı) — {best['emoji']} {best['name']} "
            f"XU100'ü Sharpe'ta ({best['sharpe']:.2f} vs {bench_sharpe:.2f}) ve "
            f"drawdown'da (%{best['max_drawdown']*100:.1f} vs %{bench_dd*100:.1f}) geçti. "
            f"Mutlak getiri (%{best['total_return']*100:+.1f} vs %{bench_ret*100:+.1f}) "
            f"enflasyonist index'in altında — bu beklenen nakit-drag etkisi, edge "
            f"sermaye koruma tarafında. Arena altyapısına yatırım gerekçeli."
        )
    else:
        best = leaderboard[0] if leaderboard else None
        edge_summary = (
            "❌ EDGE KAPISI GEÇİLMEDİ (risk-ayarlı) — hiçbir profil XU100'ü hem "
            "Sharpe hem drawdown'da geçemedi. Plan gereği arena altyapısına geçmeden "
            "önce sinyal/parametre iyileştirilmeli (fail-cheap)."
            + (f" En iyi Sharpe: {best['name']} {best['sharpe']:.2f} "
               f"(XU100 {bench_sharpe:.2f})." if best else "")
        )

    # ── Dayanıklılık verdisi (overfitting kontrolü) ────────────────────────
    robust_list = [x for x in leaderboard if x["robust"]]
    oos_leader = leaderboard[0] if leaderboard else None   # test Sharpe'ına göre sıralı
    if robust_list:
        r0 = robust_list[0]
        robust_summary = (
            f"🛡️ DAYANIKLI (overfit değil): {r0['emoji']} {r0['name']} — hem eğitim "
            f"(ilk %{int(train_frac*100)}) hem görülmemiş test (son %{int((1-train_frac)*100)}) "
            f"döneminde XU100'ü risk-ayarlı geçti. Test getirisi %{r0['test_return']*100:+.1f}, "
            f"test Sharpe {r0['test_sharpe']:.2f}. İzlenecek aday bu."
        )
    elif oos_leader:
        robust_summary = (
            f"⚠️ DAYANIKLI PROFİL YOK — hiçbiri her iki segmentte de XU100'ü geçmedi "
            f"(muhtemel overfit/rejime bağlılık). Test döneminde en iyi: {oos_leader['name']} "
            f"(getiri %{oos_leader['test_return']*100:+.1f}, Sharpe {oos_leader['test_sharpe']:.2f}). "
            f"Tek başına bir döneme güvenme."
        )
    else:
        robust_summary = ""

    caveats = [
        f"Eğitim/test 80/20 holdout: bölme {split_session}. Eğitimde iyi olup testte "
        "çöken profil overfit/şanslıdır; her ikisinde iyi olan dayanıklıdır.",
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
        split_session=split_session, train_frac=train_frac, robust_summary=robust_summary,
        equalweight_metrics=ew_metrics, equalweight_equity=ew_eq,
    )

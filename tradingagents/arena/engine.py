"""Portföy backtest motoru — sinyalleri maliyet-gerçekçi equity eğrisine çevirir.

Bu, planın "edge kapısı"nın kalbidir: bir profilin kuralları, ortak execution
fiziği altında (komisyon + slippage + T+1 fill + OHLC stop/hedef) tarihsel olarak
nasıl bir kasa eğrisi üretirdi? Çıktı XU100 al-tut ile kıyaslanır.

Konvansiyon (plan §Günlük OHLC ile stop/hedef simülasyonu):
  - Sinyal gün ``d`` kapanışında bilinir → emir gün ``d+1`` **açılışında** dolar (T+1).
  - Alış ``Open*(1+slippage)``, satış ``Open*(1-slippage)``; komisyon iki yönde bps.
  - Gap kuralı: ``Open <= stop`` ise stop fiyatından değil **açılıştan** çıkılır.
  - Gün içi ``Low <= stop`` → stop tetiklendi; ``High >= target`` → hedef tetiklendi.
  - Aynı barda hem stop hem hedef ve sıra bilinmiyorsa **muhafazakâr: stop önce**.
  - Sizing risk-bazlı: ``risk_per_trade * equity / (giriş - stop)`` adet; ağırlık,
    nakit rezervi ve ``max_positions`` ile sınırlı.

Para muhasebesi backtest'te float'tır (hız); canlı ledger Decimal/numeric kullanır
(plan §Tasarım kuralları — bu ayrım :mod:`state` katmanına aittir).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.arena.config import ExecutionConfig
from tradingagents.arena.metrics import PerfMetrics, compute_metrics
from tradingagents.arena.profiles import Profile


@dataclass
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str          # sinyal | stop | hedef | süre | sezon-sonu
    ambiguous_bar: bool = False


@dataclass
class BacktestResult:
    profile_code: str
    ok: bool = False
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list = field(default_factory=list)
    metrics: PerfMetrics = field(default_factory=PerfMetrics)
    final_equity: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    error: str = ""


def _bps(x: float) -> float:
    return x / 10_000.0


def run_backtest(
    profile: Profile,
    data: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    cfg: ExecutionConfig,
    regime: pd.Series | None = None,
) -> BacktestResult:
    """Bir profili tarihsel veride simüle eder. Asla istisna fırlatmaz.

    ``data[ticker]`` sütunları: Open, High, Low, Close, buy(bool), sell(bool),
    stop(float), target(float), trend_ok(bool). ``dates`` ana işlem takvimi
    (genelde XU100). ``regime`` (date→bool) boğa bayrağı; None ise filtre yok.
    """
    if not data or len(dates) < 2:
        return BacktestResult(profile.code, ok=False, error="Yetersiz veri")

    # Her ticker'ı ana takvime hizala (ileriye-bakış yok; eksik bar = işlemsiz gün).
    cols = ["Open", "High", "Low", "Close", "buy", "sell", "stop", "target", "trend_ok"]
    aligned: dict[str, pd.DataFrame] = {}
    for tk, df in data.items():
        d = df.reindex(dates)
        for c in cols:
            if c not in d.columns:
                d[c] = np.nan
        aligned[tk] = d

    cash = float(cfg.initial_capital)
    positions: dict[str, dict] = {}   # ticker -> {qty, entry_price, stop, target, entry_i}
    pending_buys: list[str] = []      # gün d kapanışında karar, d+1 açılışında fill
    pending_sells: list[str] = []
    equity_curve: list[float] = []
    trades: list[Trade] = []
    peak_equity = float(cfg.initial_capital)

    res_bps = _bps(cfg.commission_bps)
    slip = _bps(cfg.slippage_bps)

    for i, d in enumerate(dates):
        # ── 1) Bekleyen SATIŞLAR (sinyal çıkışı) — açılışta ────────────────
        for tk in pending_sells:
            pos = positions.get(tk)
            if pos is None:
                continue
            op = aligned[tk]["Open"].iloc[i]
            if pd.isna(op):
                continue
            cash += _close_position(tk, pos, float(op) * (1 - slip), i, dates,
                                    "sinyal", res_bps, trades)
            del positions[tk]
        pending_sells = []

        # ── 2) Bekleyen ALIŞLAR — açılışta, bütçe topluca dağıtılır ────────
        if pending_buys:
            equity_now = _mark_to_market(cash, positions, aligned, i)
            kill = equity_now < (1 - cfg.max_account_drawdown) * peak_equity
            if not kill:
                cash = _execute_buys(pending_buys, positions, aligned, i, cash,
                                     equity_now, profile, cfg, res_bps, slip)
        pending_buys = []

        # ── 3) Gün içi stop/hedef/süre çıkışları (açık pozisyonlar) ────────
        for tk in list(positions.keys()):
            pos = positions[tk]
            bar = aligned[tk]
            o, h, l = bar["Open"].iloc[i], bar["High"].iloc[i], bar["Low"].iloc[i]
            if pd.isna(h) or pd.isna(l):
                continue
            exit_price, reason, ambiguous = _check_exit(
                pos, float(o) if pd.notna(o) else None, float(h), float(l),
                i, profile,
            )
            if exit_price is not None:
                cash += _close_position(tk, pos, exit_price, i, dates, reason,
                                        res_bps, trades, ambiguous)
                del positions[tk]

        # ── 4) Kapanışta mark-to-market → equity ──────────────────────────
        equity = _mark_to_market(cash, positions, aligned, i)
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

        # ── 5) Sonraki bar için emir üret (bu kapanıştaki sinyallerle) ─────
        if i < len(dates) - 1:
            if regime is None or d not in regime.index:
                regime_ok = True
            else:
                rv = regime.loc[d]
                regime_ok = True if pd.isna(rv) else bool(rv)
            for tk, bar in aligned.items():
                held = tk in positions
                if held:
                    pos = positions[tk]
                    sell = bool(bar["sell"].iloc[i]) is True
                    aged = (i - pos["entry_i"]) >= profile.max_hold_days
                    if (sell or aged) and tk not in pending_sells:
                        pending_sells.append(tk)
                elif bool(bar["buy"].iloc[i]) is True:
                    if _entry_allowed(tk, bar, i, profile, regime_ok):
                        pending_buys.append(tk)

    # ── Sezon sonu: açık pozisyonları son kapanıştan kapat ────────────────
    last_i = len(dates) - 1
    for tk, pos in list(positions.items()):
        c = aligned[tk]["Close"].iloc[last_i]
        if pd.isna(c):
            continue
        cash += _close_position(tk, pos, float(c), last_i, dates, "sezon-sonu",
                                res_bps, trades)
    if equity_curve:
        equity_curve[-1] = cash  # son gün tamamen nakde döndü

    eq = pd.Series(equity_curve, index=dates[:len(equity_curve)], dtype=float)
    wins = sum(1 for t in trades if t.pnl > 0)
    return BacktestResult(
        profile_code=profile.code, ok=True, equity_curve=eq, trades=trades,
        metrics=compute_metrics(eq), final_equity=float(eq.iloc[-1]) if len(eq) else 0.0,
        n_trades=len(trades),
        win_rate=round(wins / len(trades), 4) if trades else 0.0,
    )


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _mark_to_market(cash: float, positions: dict, aligned: dict, i: int) -> float:
    """Nakit + açık pozisyonların güncel kapanış değeri. Bayat fiyat → son geçerli."""
    value = cash
    for tk, pos in positions.items():
        close = aligned[tk]["Close"].iloc[:i + 1].ffill().iloc[-1] if i >= 0 else np.nan
        if pd.isna(close):
            close = pos["entry_price"]
        value += pos["qty"] * float(close)
    return float(value)


def _entry_allowed(tk: str, bar: pd.DataFrame, i: int, profile: Profile,
                   regime_ok: bool) -> bool:
    """Giriş kapısı: rejim filtresi + trend-takip kendi-MA şartı + geçerli stop."""
    if profile.use_regime_filter and not regime_ok:
        return False
    if profile.require_trend_behavior and not bool(bar["trend_ok"].iloc[i]):
        return False
    stop = bar["stop"].iloc[i]
    close = bar["Close"].iloc[i]
    if pd.isna(stop) or pd.isna(close) or stop <= 0 or stop >= close:
        return False
    return True


def _execute_buys(candidates: list[str], positions: dict, aligned: dict, i: int,
                  cash: float, equity_now: float, profile: Profile,
                  cfg: ExecutionConfig, res_bps: float, slip: float) -> float:
    """Adayları ortak skorla (momentum) sırala, portföy bütçesini topluca dağıt.

    Plan §: "Kasa bütün adaylara yetmiyorsa ticker sırası kullanılmamalı; önce tüm
    adaylar ortak skorla sıralanmalı, sonra bütçe topluca dağıtılmalı."
    """
    slots = profile.max_positions - len(positions)
    if slots <= 0:
        return cash

    # Ortak skor: 20 günlük momentum (nedensel) — yüksek = öncelikli.
    ranked = []
    for tk in candidates:
        if tk in positions:
            continue
        op = aligned[tk]["Open"].iloc[i]
        if pd.isna(op) or float(op) <= 0:
            continue
        closes = aligned[tk]["Close"].iloc[max(0, i - 20):i + 1].dropna()
        mom = float(closes.iloc[-1] / closes.iloc[0] - 1.0) if len(closes) >= 2 else 0.0
        ranked.append((mom, tk, float(op)))
    ranked.sort(reverse=True)

    reserve = equity_now * profile.min_cash_reserve
    investable = max(0.0, cash * (1 - cfg.buying_power_buffer) - reserve)

    for _mom, tk, op in ranked[:slots]:
        entry = op * (1 + slip)
        stop = float(aligned[tk]["stop"].iloc[i])
        target = aligned[tk]["target"].iloc[i]
        target = float(target) if pd.notna(target) else entry * 1.10
        stop_dist = entry - stop
        if stop_dist <= 0:
            continue
        risk_budget = profile.risk_per_trade * equity_now
        qty_risk = risk_budget / stop_dist
        qty_weight = (profile.max_position_weight * equity_now) / entry
        qty_cash = investable / (entry * (1 + res_bps))
        qty = int(min(qty_risk, qty_weight, qty_cash))
        if qty <= 0:
            continue
        gross = qty * entry
        commission = gross * res_bps
        cost = gross + commission
        if cost > cash or cost > investable:
            continue
        cash -= cost
        investable -= cost
        positions[tk] = {"qty": qty, "entry_price": entry, "stop": stop,
                         "target": target, "entry_i": i}
    return cash


def _check_exit(pos: dict, o: float | None, h: float, l: float, i: int,
                profile: Profile):
    """OHLC stop/hedef/süre kontrolü → (exit_price | None, reason, ambiguous)."""
    stop, target, entry_i = pos["stop"], pos["target"], pos["entry_i"]
    hit_stop = profile.use_stop and l <= stop
    hit_target = profile.use_target and h >= target
    same_bar_entry = (entry_i == i)

    if hit_stop and hit_target:
        # Sıra bilinmiyor → muhafazakâr: stop önce (plan §ambiguous_bar)
        price = stop
        if (not same_bar_entry) and o is not None and o <= stop:
            price = o  # gap: açılıştan
        return price, "stop", True
    if hit_stop:
        price = stop
        if (not same_bar_entry) and o is not None and o <= stop:
            price = o
        return price, "stop", False
    if hit_target:
        price = target
        if (not same_bar_entry) and o is not None and o >= target:
            price = o
        return price, "hedef", False
    # Maksimum tutma süresi çıkışı, sinyal üretim adımında (5) pending_sells'e
    # eklenir; burada yalnız gün-içi stop/hedef ele alınır.
    return None, "", False


def _close_position(tk: str, pos: dict, price: float, i: int, dates,
                    reason: str, res_bps: float, trades: list,
                    ambiguous: bool = False) -> float:
    """Pozisyonu kapatır, Trade kaydı üretir, net nakit girişini döndürür."""
    qty = pos["qty"]
    gross = qty * price
    commission = gross * res_bps
    proceeds = gross - commission
    cost_basis = qty * pos["entry_price"]
    pnl = proceeds - cost_basis
    trades.append(Trade(
        ticker=tk,
        entry_date=str(pd.Timestamp(dates[pos["entry_i"]]).date()),
        exit_date=str(pd.Timestamp(dates[i]).date()),
        entry_price=round(pos["entry_price"], 4), exit_price=round(price, 4),
        quantity=qty, pnl=round(pnl, 2),
        pnl_pct=round(pnl / cost_basis, 4) if cost_basis > 0 else 0.0,
        exit_reason=reason, ambiguous_bar=ambiguous,
    ))
    return proceeds

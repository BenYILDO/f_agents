"""Tarihsel kanıt — sinyal edge'i + strateji backtest metrikleri.

"Bu sinyali geçmişte takip etseydin ne olurdu?" güvenin en somut kanıtıdır.
İki çıktı:

  - **İleri getiri dağılımı**: her giriş sinyalinden sonra +5/+10/+20 günlük
    getirinin ortalaması, medyanı ve **isabet oranı** (örneklem sayısıyla).
  - **Strateji metrikleri**: sinyali takip eden uzun-only bir stratejinin
    kazanma oranı, bileşik getirisi (CAGR), **maksimum düşüşü (drawdown)** ve
    Sharpe'ı.

Saf/ağsız: OHLCV + giriş/çıkış boolean serileri ister. ``edge_for_ticker`` ise
sistemdeki Dip-Al sinyalini (SMI/VWMA/Bollinger) kullanarak hisse için çalıştırır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


@dataclass
class EdgeResult:
    ok: bool
    n_signals: int = 0
    fwd: dict = field(default_factory=dict)      # horizon -> {mean,median,hit_rate,n}
    equity: dict = field(default_factory=dict)   # win_rate,cagr,max_drawdown,sharpe,...
    note: str = ""
    error: str = ""


def _forward_returns(close: pd.Series, entries: pd.Series, horizons) -> dict:
    c = close.to_numpy(dtype=float)
    idx = np.where(entries.to_numpy())[0]
    out: dict = {}
    for h in horizons:
        rets = [(c[i + h] / c[i] - 1) * 100 for i in idx if i + h < len(c) and c[i] > 0]
        if rets:
            arr = np.array(rets)
            out[f"+{h}g"] = {
                "n": int(arr.size),
                "mean": round(float(arr.mean()), 2),
                "median": round(float(np.median(arr)), 2),
                "hit_rate": round(float((arr > 0).mean() * 100), 1),
            }
        else:
            out[f"+{h}g"] = {"n": 0, "mean": None, "median": None, "hit_rate": None}
    return out


def _strategy_metrics(close: pd.Series, entries: pd.Series, exits: pd.Series,
                      max_hold: int = 40) -> dict:
    """Uzun-only: giriş sinyalinde gir, çıkış sinyalinde (ya da max_hold sonra) çık."""
    c = close.to_numpy(dtype=float)
    en = entries.to_numpy()
    ex = exits.to_numpy()
    n = len(c)
    position = np.zeros(n)
    trades: list[float] = []
    in_pos = False
    entry_i = 0
    for i in range(n):
        if in_pos:
            position[i] = 1.0
            if ex[i] or (i - entry_i) >= max_hold or i == n - 1:
                if c[entry_i] > 0:
                    trades.append(c[i] / c[entry_i] - 1)
                in_pos = False
        elif en[i] and i < n - 1:
            in_pos = True
            entry_i = i

    daily_ret = pd.Series(c, index=close.index).pct_change().fillna(0).to_numpy()
    strat_ret = np.roll(position, 1) * daily_ret
    strat_ret[0] = 0.0
    equity = np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1).min()) if equity.size else 0.0
    std = strat_ret.std()
    sharpe = float(strat_ret.mean() / std * np.sqrt(_TRADING_DAYS)) if std > 0 else 0.0
    years = n / _TRADING_DAYS
    final_mult = float(equity[-1]) if equity.size else 1.0
    cagr = (final_mult ** (1 / years) - 1) * 100 if years > 0 and final_mult > 0 else 0.0
    wins = sum(1 for t in trades if t > 0)
    return {
        "trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1) if trades else None,
        "avg_trade": round(float(np.mean(trades)) * 100, 2) if trades else None,
        "final_mult": round(final_mult, 2),
        "cagr": round(cagr, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "sharpe": round(sharpe, 2),
    }


def signal_edge(close: pd.Series, entries: pd.Series, exits: pd.Series,
                horizons=(5, 10, 20)) -> EdgeResult:
    """İleri getiri dağılımı + strateji metrikleri (saf)."""
    if close is None or len(close) < 60:
        return EdgeResult(False, error="Yeterli veri yok.")
    entries = entries.reindex(close.index).fillna(False).astype(bool)
    exits = exits.reindex(close.index).fillna(False).astype(bool)
    n_sig = int(entries.sum())
    if n_sig == 0:
        return EdgeResult(True, n_signals=0, note="Geçmişte bu sinyalden örnek yok.")
    return EdgeResult(
        ok=True,
        n_signals=n_sig,
        fwd=_forward_returns(close, entries, horizons),
        equity=_strategy_metrics(close, entries, exits),
        note=f"{n_sig} geçmiş giriş sinyali üzerinden.",
    )


def edge_for_ticker(ticker: str, df: pd.DataFrame | None = None) -> EdgeResult:
    """Sistemdeki Dip-Al sinyalini kullanarak hisse için backtest (ağ: yfinance)."""
    from tradingagents.analytics.composite import _fetch_daily
    from tradingagents.strategy.dip_signal import compute_signals

    if df is None:
        df = _fetch_daily(ticker)
    if df is None or df.empty or len(df) < 60:
        return EdgeResult(False, error="Yeterli fiyat verisi yok.")
    try:
        sig = compute_signals(df)
    except Exception as exc:  # noqa: BLE001
        return EdgeResult(False, error=f"Sinyal hesaplanamadı: {exc}")
    return signal_edge(sig["Close"], sig["buy"].astype(bool), sig["sell"].astype(bool))

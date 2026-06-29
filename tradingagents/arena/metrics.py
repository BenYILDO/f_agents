"""Performans metrikleri — equity eğrisinden getiri/Sharpe/drawdown (saf).

Tüm fonksiyonlar numpy/pandas üzerinde çalışır, istisna fırlatmaz; yetersiz veride
nötr/None döner. Yıllıklandırma 252 işlem günü varsayar (BIST günlük).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


@dataclass
class PerfMetrics:
    total_return: float = 0.0      # toplam getiri (ör. 0.23 = %23)
    cagr: float = 0.0              # yıllık bileşik getiri
    sharpe: float = 0.0            # yıllık Sharpe (rf=0)
    max_drawdown: float = 0.0      # en derin tepe-dip düşüş (negatif)
    volatility: float = 0.0        # yıllık volatilite
    n_days: int = 0

    def as_row(self) -> dict:
        return {
            "Toplam getiri": f"%{self.total_return*100:+.1f}",
            "CAGR": f"%{self.cagr*100:+.1f}",
            "Sharpe": f"{self.sharpe:.2f}",
            "Maks DD": f"%{self.max_drawdown*100:.1f}",
            "Yıllık vol": f"%{self.volatility*100:.1f}",
        }


def max_drawdown(equity: pd.Series) -> float:
    """En derin tepe-dip düşüş oranı (negatif değer; -0.2 = %20 düşüş)."""
    if equity is None or len(equity) < 2:
        return 0.0
    arr = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak > 0, peak, np.nan)
    return float(np.nanmin(dd))


def compute_metrics(equity: pd.Series) -> PerfMetrics:
    """Equity eğrisinden tam metrik seti. Boş/yetersiz → sıfır metrik."""
    if equity is None or len(equity) < 2:
        return PerfMetrics()
    eq = pd.Series(equity, dtype=float).dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return PerfMetrics(n_days=len(eq))

    rets = eq.pct_change().dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max(len(eq) / _TRADING_DAYS, 1e-9)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
    vol = float(rets.std() * np.sqrt(_TRADING_DAYS)) if len(rets) > 1 else 0.0
    mean = float(rets.mean() * _TRADING_DAYS)
    sharpe = float(mean / vol) if vol > 1e-12 else 0.0
    return PerfMetrics(
        total_return=round(total, 4), cagr=round(cagr, 4), sharpe=round(sharpe, 2),
        max_drawdown=round(max_drawdown(eq), 4), volatility=round(vol, 4), n_days=len(eq),
    )


def benchmark_buy_hold(close: pd.Series, initial_capital: float,
                       slippage_bps: float = 0.0) -> pd.Series:
    """XU100 al-tut equity eğrisi — ilk barda tüm kasayla alıp tutar.

    Aktif stratejilerle aynı T+1/maliyet konvansiyonu için ilk alımda slippage
    uygulanır (plan §benchmark aynı konvansiyonla başlar).
    """
    c = pd.Series(close, dtype=float).dropna()
    if len(c) < 2 or c.iloc[0] <= 0:
        return pd.Series(dtype=float)
    entry = c.iloc[0] * (1.0 + slippage_bps / 10_000.0)
    units = initial_capital / entry
    return units * c

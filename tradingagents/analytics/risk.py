"""Risk-farkında sinyal — ATR tabanlı stop/hedef + risk/ödül + likidite.

Bir AL sinyaline güven, "nerede yanıldığını bilmekle" artar. Bu modül son fiyata
göre ATR tabanlı bir **stop** ve **hedef** üretir, **risk/ödül (R/R)** oranını
hesaplar ve hissenin **likiditesini** (ortalama günlük TL hacmi) ölçer — düşük
likiditeli (BIST'te kayma/tuzak riski yüksek) sinyalleri işaretler.

Saf/ağsız: yalnız OHLCV DataFrame'i ister, asla istisna fırlatmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from tradingagents.analytics.indicators import atr

# Likidite eşikleri (ortalama günlük TL hacmi) — BIST için kabaca.
_LIQ_HIGH = 50_000_000.0
_LIQ_MED = 10_000_000.0


@dataclass
class RiskPlan:
    ok: bool
    close: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0           # ATR / fiyat * 100 (volatilite)
    stop: float = 0.0              # long stop = close - mult*ATR
    target: float = 0.0           # hedef (R/R * risk ya da verilen direnç)
    risk_per_share: float = 0.0
    reward_per_share: float = 0.0
    rr: float = 0.0               # ödül / risk
    avg_tl_volume: float = 0.0     # ortalama günlük TL hacmi
    liquidity: str = "—"          # 'yüksek' | 'orta' | 'düşük'
    notes: list = field(default_factory=list)
    error: str = ""


def _liquidity_label(avg_tl: float) -> str:
    if avg_tl >= _LIQ_HIGH:
        return "yüksek"
    if avg_tl >= _LIQ_MED:
        return "orta"
    return "düşük"


def compute_risk(
    df: pd.DataFrame,
    atr_mult: float = 2.0,
    target_rr: float = 2.0,
    atr_n: int = 14,
    target_price: Optional[float] = None,
    vol_lookback: int = 20,
) -> RiskPlan:
    """ATR tabanlı stop/hedef + R/R + likidite. ``target_price`` verilirse
    (örn. en yakın direnç) hedef o olur ve R/R ona göre hesaplanır."""
    if df is None or df.empty or len(df) < atr_n + 2:
        return RiskPlan(False, error="Yeterli veri yok.")
    try:
        a = float(atr(df, atr_n).iloc[-1])
        close = float(df["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        return RiskPlan(False, error="ATR/fiyat hesaplanamadı.")
    if not (a > 0 and close > 0):
        return RiskPlan(False, error="Geçersiz ATR/fiyat.")

    stop = close - atr_mult * a
    risk = close - stop
    if target_price is not None and target_price > close:
        target = float(target_price)
    else:
        target = close + target_rr * risk
    reward = target - close
    rr = reward / risk if risk > 0 else 0.0
    atr_pct = a / close * 100

    notes: list[str] = []
    if atr_pct >= 6:
        notes.append(f"Yüksek volatilite (ATR %{atr_pct:.1f}) — stop geniş, pozisyonu küçült.")
    if rr < 1.5:
        notes.append(f"Zayıf risk/ödül ({rr:.2f}) — kurulum cazip değil.")
    elif rr >= 2.5:
        notes.append(f"Güçlü risk/ödül ({rr:.2f}).")

    avg_tl = 0.0
    if "Volume" in df.columns:
        tl = (df["Close"] * df["Volume"]).tail(vol_lookback).dropna()
        if not tl.empty:
            avg_tl = float(tl.mean())
    liquidity = _liquidity_label(avg_tl)
    if liquidity == "düşük":
        notes.append("Düşük likidite — kayma/manipülasyon riski; sinyale temkinli yaklaş.")

    return RiskPlan(
        ok=True, close=round(close, 4), atr=round(a, 4), atr_pct=round(atr_pct, 2),
        stop=round(stop, 4), target=round(target, 4),
        risk_per_share=round(risk, 4), reward_per_share=round(reward, 4),
        rr=round(rr, 2), avg_tl_volume=round(avg_tl, 0), liquidity=liquidity, notes=notes,
    )

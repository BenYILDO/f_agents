"""Teyit katmanı — RSI/MACD divergence + hacim teyidi + göreceli güç.

Tek bir indikatör yanıltır; bağımsız teyitler güveni yükseltir:

  - **Divergence**: fiyat yeni dip yaparken RSI/MACD daha yüksek dip yapıyorsa
    boğa uyumsuzluğu (dönüş sinyali); tersi ayı uyumsuzluğu. Klasik, güçlü teyit.
  - **Hacim teyidi**: son hareket ortalama-üstü hacimle mi geldi (gerçek) yoksa
    cılız hacimle mi (şüpheli)? OBV trendi yön teyidi verir.
  - **Göreceli güç**: hisse, endekse (XU100) karşı son dönemde güçlü mü? Endeksi
    yenen hisse, alış tezini teyit eder.

Saf/ağsız: OHLCV (+ opsiyonel benchmark) DataFrame'i ister; asla istisna fırlatmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.analytics.indicators import macd, obv, rsi


@dataclass
class ConfirmationResult:
    ok: bool
    rsi_divergence: str = "yok"      # 'yok' | 'pozitif (boğa)' | 'negatif (ayı)'
    macd_divergence: str = "yok"
    volume_confirms: bool = False
    volume_note: str = ""
    obv_trend: str = "yatay"          # 'yukarı' | 'aşağı' | 'yatay'
    rel_strength: Optional[float] = None  # hisse − benchmark getirisi (%, lookback)
    rs_note: str = ""
    score: float = 0.0                # [-1,+1] toplam teyit skoru
    notes: list = field(default_factory=list)
    error: str = ""


def _swing_idx(series: pd.Series, k: int, kind: str) -> list[int]:
    """Pivot indeksleri: bir bar, ±k komşusunun min(low)/max(high)'i ise swing."""
    vals = series.to_numpy()
    n = len(vals)
    out = []
    for i in range(k, n - k):
        window = vals[i - k : i + k + 1]
        if kind == "low" and vals[i] == window.min():
            out.append(i)
        elif kind == "high" and vals[i] == window.max():
            out.append(i)
    return out


def _divergence(price: pd.Series, osc: pd.Series, k: int = 3) -> str:
    """Son iki swing'i karşılaştırır → 'pozitif (boğa)'/'negatif (ayı)'/'yok'."""
    lows = _swing_idx(price, k, "low")
    highs = _swing_idx(price, k, "high")
    p, o = price.to_numpy(), osc.to_numpy()
    # Boğa: fiyat daha düşük dip, osilatör daha yüksek dip
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if np.isfinite(o[a]) and np.isfinite(o[b]) and p[b] < p[a] and o[b] > o[a]:
            return "pozitif (boğa)"
    # Ayı: fiyat daha yüksek tepe, osilatör daha düşük tepe
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if np.isfinite(o[a]) and np.isfinite(o[b]) and p[b] > p[a] and o[b] < o[a]:
            return "negatif (ayı)"
    return "yok"


def compute_confirmation(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    lookback: int = 60,
    rs_window: int = 60,
) -> ConfirmationResult:
    """Divergence + hacim teyidi + göreceli güç. Asla istisna fırlatmaz."""
    if df is None or df.empty or len(df) < 35:
        return ConfirmationResult(False, error="Yeterli veri yok.")

    res = ConfirmationResult(ok=True)
    notes: list[str] = []
    score = 0.0

    window = df.tail(lookback)
    close = window["Close"]
    try:
        rsi_s = rsi(df["Close"]).tail(lookback)
        macd_line, _, _ = macd(df["Close"])
        macd_s = macd_line.tail(lookback)
        res.rsi_divergence = _divergence(close, rsi_s)
        res.macd_divergence = _divergence(close, macd_s)
    except Exception:  # noqa: BLE001
        pass
    for dv in (res.rsi_divergence, res.macd_divergence):
        if dv.startswith("pozitif"):
            score += 0.25
            notes.append("Boğa uyumsuzluğu (divergence) — dönüş/teyit lehte.")
        elif dv.startswith("negatif"):
            score -= 0.25
            notes.append("Ayı uyumsuzluğu (divergence) — zayıflama uyarısı.")

    # Hacim teyidi
    if "Volume" in df.columns:
        vol = df["Volume"]
        avg = float(vol.tail(20).mean()) if vol.tail(20).notna().any() else 0.0
        last_vol = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else 0.0
        last_up = df["Close"].iloc[-1] >= df["Close"].iloc[-2]
        if avg > 0:
            ratio = last_vol / avg
            res.volume_confirms = bool(ratio >= 1.2 and last_up)
            if ratio >= 1.2 and last_up:
                res.volume_note = f"Yükseliş ortalama-üstü hacimle (×{ratio:.1f}) — teyitli."
                score += 0.2
            elif ratio < 0.8 and last_up:
                res.volume_note = f"Yükseliş cılız hacimle (×{ratio:.1f}) — şüpheli."
                score -= 0.1
            else:
                res.volume_note = f"Hacim ×{ratio:.1f} (ortalamaya göre)."
        # OBV trend
        try:
            obv_s = obv(df)
            if len(obv_s) > 20 and pd.notna(obv_s.iloc[-1]) and pd.notna(obv_s.iloc[-21]):
                if obv_s.iloc[-1] > obv_s.iloc[-21]:
                    res.obv_trend = "yukarı"
                    score += 0.1
                elif obv_s.iloc[-1] < obv_s.iloc[-21]:
                    res.obv_trend = "aşağı"
                    score -= 0.1
        except Exception:  # noqa: BLE001
            pass

    # Göreceli güç (XU100 vb.)
    if benchmark_df is not None and not benchmark_df.empty and len(df) > rs_window:
        try:
            stock_ret = df["Close"].iloc[-1] / df["Close"].iloc[-rs_window] - 1
            b = benchmark_df["Close"].dropna()
            if len(b) > rs_window:
                bench_ret = b.iloc[-1] / b.iloc[-rs_window] - 1
                rs = (stock_ret - bench_ret) * 100
                res.rel_strength = round(rs, 2)
                if rs > 0:
                    res.rs_note = f"Endeksi yeniyor (+%{rs:.1f}, {rs_window}g) — teyit lehte."
                    score += 0.15
                else:
                    res.rs_note = f"Endeksin gerisinde (%{rs:.1f}, {rs_window}g)."
                    score -= 0.1
        except Exception:  # noqa: BLE001
            pass

    res.score = round(max(-1.0, min(1.0, score)), 2)
    res.notes = notes
    return res

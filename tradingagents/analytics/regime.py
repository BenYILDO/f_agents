"""Piyasa rejimi tespiti — volatilite, trend ve TL stres göstergesi.

Türkiye piyasası dış/iç siyasi şoklara hızlı tepki verir: kur krizi, seçim
belirsizliği, jeopolitik gerilim dönemlerinde teknik sinyallerin isabeti
düşer. Bu modül o "rejimi" sayısallaştırır ki kompozit skor oynak dönemde
sinyal GÜVENİNİ otomatik kıssın (yönü değil — yön bilgisi diğer bileşenlerin
işi). Üç bacak:

  1. **Volatilite rejimi** — 20 günlük gerçekleşen volatilitenin 2 yıllık
     dağılımdaki yüzdeliği: Düşük / Normal / Yüksek / Aşırı.
  2. **Trend rejimi** — ADX(14) + SMA eğimi: Güçlü Trend / Zayıf Trend / Yatay.
  3. **TL stres** (yalnız BIST) — USDTRY'nin 20 günlük ivmesi ve volatilite
     yüzdeliği: kur sakinse 0, kur koparsa 1'e yaklaşır. 2018 Brunson, 2021
     Aralık, 2023 seçim sonrası gibi dönemlerde bu gösterge tek başına
     "teknik sinyale güvenme" demektir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.analytics.indicators import adx, sma


@dataclass
class RegimeResult:
    ok: bool
    vol_regime: str = ""        # "Düşük" | "Normal" | "Yüksek" | "Aşırı"
    vol_percentile: float = 0.0
    trend_regime: str = ""      # "Güçlü Yükseliş" | "Güçlü Düşüş" | "Zayıf Trend" | "Yatay"
    adx: float = 0.0
    try_stress: float | None = None   # 0-1; None = BIST dışı / veri yok
    try_note: str = ""
    confidence_mult: float = 1.0      # kompozit skor güven çarpanı (0.4-1.0)
    summary: str = ""
    reason: str = ""


def _realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Yıllıklaştırılmış gerçekleşen volatilite (%)."""
    return close.pct_change().rolling(window).std() * np.sqrt(252) * 100


def fetch_try_stress() -> tuple[float | None, str]:
    """USDTRY stres skoru [0, 1] ve Türkçe not döndürür. Asla istisna fırlatmaz."""
    try:
        fx = yf.Ticker("TRY=X").history(period="2y", interval="1d")
    except Exception:  # noqa: BLE001
        return None, "USDTRY verisi alınamadı."
    if fx is None or fx.empty or len(fx) < 60:
        return None, "USDTRY verisi yetersiz."
    close = fx["Close"].dropna()
    mom20 = float(close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0.0
    vol = _realized_vol(close)
    cur_vol = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else 0.0
    vol_pct = float((vol.dropna() <= cur_vol).mean())
    # Aylık %3+ TL değer kaybı ya da vol dağılımının üst ucunda olmak stres demektir
    mom_part = max(0.0, min(1.0, mom20 / 6.0))
    vol_part = max(0.0, (vol_pct - 0.5) * 2)
    stress = round(max(0.0, min(1.0, 0.6 * mom_part + 0.4 * vol_part)), 2)
    note = (f"USDTRY 20 günlük değişim %{mom20:+.1f}, volatilite yüzdeliği "
            f"%{vol_pct * 100:.0f} → TL stres {stress:.2f}/1.00")
    return stress, note


def compute_regime(df: pd.DataFrame, is_bist: bool = False,
                   try_stress: float | None = None, try_note: str = "") -> RegimeResult:
    """OHLCV'den rejim okuması yapar; ``try_stress`` verilmezse ve BIST ise çeker.

    ``confidence_mult`` kompozit skorun güvenini çarpar: sakin trendli piyasada
    1.0; aşırı volatilite + TL stresi üst üste binince 0.4'e kadar düşer.
    """
    if df is None or df.empty or len(df) < 250:
        return RegimeResult(False, reason="Rejim için en az ~1 yıl günlük veri gerekli.")

    close = df["Close"].dropna()
    vol = _realized_vol(close)
    cur_vol = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else 0.0
    vol_pct = float((vol.dropna() <= cur_vol).mean())
    if vol_pct < 0.25:
        vol_regime = "Düşük"
    elif vol_pct < 0.75:
        vol_regime = "Normal"
    elif vol_pct < 0.92:
        vol_regime = "Yüksek"
    else:
        vol_regime = "Aşırı"

    cur_adx = adx(df)
    cur_adx = float(cur_adx.iloc[-1]) if pd.notna(cur_adx.iloc[-1]) else 0.0
    sma50 = sma(close, 50)
    slope_up = bool(pd.notna(sma50.iloc[-1]) and pd.notna(sma50.iloc[-11])
                    and sma50.iloc[-1] > sma50.iloc[-11])
    if cur_adx >= 25:
        trend_regime = "Güçlü Yükseliş" if slope_up else "Güçlü Düşüş"
    elif cur_adx >= 20:
        trend_regime = "Zayıf Trend"
    else:
        trend_regime = "Yatay"

    if is_bist and try_stress is None:
        try_stress, try_note = fetch_try_stress()

    # Güven çarpanı: volatilite cezası + TL stres cezası (yalnız BIST)
    mult = 1.0
    if vol_regime == "Yüksek":
        mult -= 0.15
    elif vol_regime == "Aşırı":
        mult -= 0.3
    if try_stress is not None:
        mult -= 0.3 * try_stress
    mult = round(max(0.4, mult), 2)

    parts = [f"Volatilite: {vol_regime} (yüzdelik %{vol_pct * 100:.0f})",
             f"Trend: {trend_regime} (ADX {cur_adx:.0f})"]
    if try_stress is not None:
        parts.append(f"TL stres: {try_stress:.2f}")
    parts.append(f"Sinyal güven çarpanı: {mult:.2f}")

    return RegimeResult(
        ok=True, vol_regime=vol_regime, vol_percentile=round(vol_pct, 2),
        trend_regime=trend_regime, adx=round(cur_adx, 1),
        try_stress=try_stress, try_note=try_note,
        confidence_mult=mult, summary=" · ".join(parts),
    )

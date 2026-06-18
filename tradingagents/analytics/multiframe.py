"""Çoklu zaman dilimi teyidi (MTF confluence) — haftalık + günlük + 4 saatlik.

Tek zaman diliminde iyi görünen bir sinyal, üst zaman dilimi trendine ters
olabilir ("trende karşı işlem"). Bu modül aynı hisseyi birkaç zaman diliminde
değerlendirip yönlerinin **hizasına** bakar: üçü de yukarı → yüksek güven; günlük
yukarı ama haftalık aşağı → karşı-trend uyarısı.

Her zaman dilimi için yön, trend (SMA hiyerarşisi) + momentum (RSI) ile belirlenir
(tam kompozitten daha hafif; MTF'nin amacı hizadır). ``_frame_direction`` saf ve
test edilebilir; ``compute_mtf`` ağ (yfinance) kullanır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from tradingagents.analytics.indicators import rsi, sma

# Zaman dilimi etiketi -> (yfinance interval, period, resample kuralı, ağırlık)
_FRAMES = {
    "Haftalık": ("1wk", "5y", None, 1.2),
    "Günlük": ("1d", "2y", None, 1.0),
    "4 Saatlik": ("1h", "720d", "4h", 0.8),
}


@dataclass
class MTFResult:
    ok: bool
    frames: dict = field(default_factory=dict)   # etiket -> {'dir','detail'}
    net: float = 0.0                              # ağırlıklı yön toplamı
    confluence: str = "nötr"                      # 'güçlü AL'…'güçlü SAT'
    score: float = 0.0                            # [-1,+1]
    note: str = ""
    error: str = ""


def _frame_direction(df: pd.DataFrame) -> tuple[int, str]:
    """Bir zaman diliminin yönü: +1 yukarı / -1 aşağı / 0 yatay (+ açıklama)."""
    if df is None or df.empty or len(df) < 60:
        return 0, "veri yetersiz"
    close = df["Close"]
    last = float(close.iloc[-1])
    s50 = float(sma(close, 50).iloc[-1])
    s200 = float(sma(close, 200).iloc[-1]) if len(close) >= 200 else s50
    r = float(rsi(close).iloc[-1])
    up = last > s50 and s50 >= s200 and r >= 50
    down = last < s50 and s50 <= s200 and r <= 50
    if up:
        return 1, f"trend yukarı (fiyat>SMA50>SMA200, RSI {r:.0f})"
    if down:
        return -1, f"trend aşağı (fiyat<SMA50<SMA200, RSI {r:.0f})"
    return 0, f"karışık (RSI {r:.0f})"


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    cols = [c for c in agg if c in df.columns]
    return df[cols].resample(rule).agg({c: agg[c] for c in cols}).dropna(how="any")


def _fetch_frame(ticker: str, interval: str, period: str, rule: str | None):
    try:
        raw = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception:  # noqa: BLE001
        return None
    if raw is None or raw.empty:
        return None
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    if rule:
        raw = _resample(raw, rule)
    return raw


def aggregate(frames: dict) -> MTFResult:
    """Zaman dilimi yönlerinden mutabakat seviyesini üretir (saf)."""
    if not frames:
        return MTFResult(False, error="Zaman dilimi verisi yok.")
    weights = {label: _FRAMES.get(label, (None, None, None, 1.0))[3] for label in frames}
    net = sum(frames[l]["dir"] * weights[l] for l in frames)
    total_w = sum(weights.values()) or 1.0
    score = round(net / total_w, 2)
    ups = sum(1 for l in frames if frames[l]["dir"] > 0)
    downs = sum(1 for l in frames if frames[l]["dir"] < 0)
    n = len(frames)
    if ups == n:
        conf = "güçlü AL"
    elif downs == n:
        conf = "güçlü SAT"
    elif ups > downs and downs == 0:
        conf = "AL"
    elif downs > ups and ups == 0:
        conf = "SAT"
    elif ups > 0 and downs > 0:
        conf = "karışık (karşı-trend riski)"
    else:
        conf = "nötr"
    note = " · ".join(f"{l}: {'↑' if frames[l]['dir']>0 else '↓' if frames[l]['dir']<0 else '→'}"
                      for l in frames)
    return MTFResult(True, frames=frames, net=round(net, 2), confluence=conf,
                     score=score, note=note)


def compute_mtf(ticker: str) -> MTFResult:
    """Hisseyi haftalık/günlük/4 saatlik değerlendirir (ağ: yfinance)."""
    frames: dict = {}
    for label, (interval, period, rule, _w) in _FRAMES.items():
        df = _fetch_frame(ticker, interval, period, rule)
        direction, detail = _frame_direction(df)
        if df is not None and not df.empty:
            frames[label] = {"dir": direction, "detail": detail}
    return aggregate(frames)

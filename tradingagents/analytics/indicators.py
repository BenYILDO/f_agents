"""Ortak indikatör çekirdeği — saf pandas/numpy, deterministik.

Diğer analytics modülleri (rejim, kompozit skor) ve Streamlit teknik analiz
sayfası buradaki hesapları paylaşır; böylece RSI/ADX gibi değerler her yerde
birebir aynı çıkar. ``stockstats`` yerine elle yazılmıştır çünkü ADX ve
Wilder-yumuşatmalı RSI gibi varyantların TradingView ile uyumlu olması
istenir (trader'ın grafikte gördüğüyle ajanın okuduğu sayı aynı olmalı).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder yumuşatması (RMA) — RSI/ATR/ADX'in TradingView-uyumlu temeli."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), n)
    loss = _wilder((-delta).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """(macd, sinyal, histogram) üçlüsünü döndürür."""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift()
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ADX(14) — trend gücü (25+ trendli, 20- yatay piyasa kabul edilir)."""
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_n = _wilder(true_range(df), n)
    plus_di = 100 * _wilder(plus_dm, n) / tr_n.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, n) / tr_n.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, n)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """(orta, üst, alt) bantlarını döndürür."""
    mid = sma(close, n)
    std = close.rolling(n).std()
    return mid, mid + k * std, mid - k * std


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    """Stokastik %K ve %D."""
    low_k = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    return pct_k, pct_k.rolling(d).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — hacim teyidi için."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame'ine kompozit skorun kullandığı tüm kolonları ekler."""
    out = df.copy()
    close = out["Close"]
    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)
    out["sma200"] = sma(close, 200)
    out["rsi14"] = rsi(close)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["atr14"] = atr(out)
    out["adx14"] = adx(out)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bollinger(close)
    out["stoch_k"], out["stoch_d"] = stochastic(out)
    out["obv"] = obv(out)
    return out

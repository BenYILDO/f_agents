"""Özellik mühendisliği — OHLCV'den ML modeli için sayısal özellik matrisi.

Deterministik indikatör çekirdeğini (``analytics.indicators``) yeniden kullanır
ve fiyat-bağımsız, ölçeklenmiş özellikler üretir: ham fiyat seviyeleri yerine
oranlar/yüzdelikler kullanılır (model bir hissede öğrendiğini başka rejime
taşıyabilsin ve fiyat ölçeğine aşırı uyum yapmasın).

Etiketler ``make_labels`` ile ileriye dönük getiriden üretilir: ufuk (horizon)
gün sonra fiyat eşiğin üzerindeyse 1 (yukarı), değilse 0. Sızıntıyı (look-ahead)
önlemek için etiket ileri kaydırılır ve son ``horizon`` bar eğitimden düşülür.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.analytics.indicators import add_core_indicators

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_20",
    "rsi14", "macd_hist_norm", "adx14",
    "px_vs_sma20", "px_vs_sma50", "px_vs_sma200",
    "sma50_vs_sma200", "bb_pos", "stoch_k",
    "atr_pct", "vol_ratio", "obv_slope", "dist_from_high252",
]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → ölçeklenmiş özellik DataFrame'i (``FEATURE_COLUMNS``).

    NaN satırlar (ilk ~200 bar, uzun SMA ısınması) çağıran tarafından düşülür.
    """
    ind = add_core_indicators(df)
    close = ind["Close"]
    out = pd.DataFrame(index=ind.index)

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)
    out["rsi14"] = ind["rsi14"] / 100.0
    # MACD histogramını fiyata göre normalize et (ölçek-bağımsız)
    out["macd_hist_norm"] = ind["macd_hist"] / close.replace(0, np.nan)
    out["adx14"] = ind["adx14"] / 100.0
    out["px_vs_sma20"] = close / ind["sma20"] - 1
    out["px_vs_sma50"] = close / ind["sma50"] - 1
    out["px_vs_sma200"] = close / ind["sma200"] - 1
    out["sma50_vs_sma200"] = ind["sma50"] / ind["sma200"] - 1
    # Bollinger içi konum: 0 = alt band, 1 = üst band
    band = (ind["bb_upper"] - ind["bb_lower"]).replace(0, np.nan)
    out["bb_pos"] = (close - ind["bb_lower"]) / band
    out["stoch_k"] = ind["stoch_k"] / 100.0
    out["atr_pct"] = ind["atr14"] / close.replace(0, np.nan)
    vol_ma = ind["Volume"].rolling(20).mean().replace(0, np.nan)
    out["vol_ratio"] = ind["Volume"] / vol_ma
    obv = ind["obv"]
    out["obv_slope"] = (obv - obv.shift(10)) / obv.rolling(20).std().replace(0, np.nan)
    high252 = ind["High"].rolling(252, min_periods=60).max()
    out["dist_from_high252"] = close / high252 - 1

    return out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, horizon: int = 10, threshold: float = 0.0) -> pd.Series:
    """İleriye dönük ikili etiket: ``horizon`` gün sonraki getiri > eşik → 1.

    ``threshold`` oransal (0.02 = %2). Son ``horizon`` bar geleceği bilinmediği
    için NaN olur ve eğitim tarafında düşülür.
    """
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    return (fwd > threshold).astype("float64").where(fwd.notna())


def build_training_set(
    df: pd.DataFrame, horizon: int = 10, threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Hizalı (X, y) döndürür; NaN özellik/etiket satırları düşülmüş."""
    X = build_feature_frame(df)
    y = make_labels(df, horizon, threshold)
    data = X.join(y.rename("_label")).dropna()
    return data[FEATURE_COLUMNS], data["_label"]

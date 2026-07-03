"""Özellik mühendisliği — OHLCV'den ML modeli için sayısal özellik matrisi.

Deterministik indikatör çekirdeğini (``analytics.indicators``) yeniden kullanır
ve fiyat-bağımsız, ölçeklenmiş özellikler üretir: ham fiyat seviyeleri yerine
oranlar/yüzdelikler kullanılır (model bir hissede öğrendiğini başka rejime
taşıyabilsin ve fiyat ölçeğine aşırı uyum yapmasın).

İki etiketleme yolu vardır:

- ``make_labels`` (eski, sabit-ufuk): horizon gün sonra fiyat eşiğin üzerindeyse 1.
  Basit ama motorun gerçek işlem kurallarıyla (stop/hedef) hizasızdır.
- ``make_labels_triple_barrier`` (S1, varsayılan): López de Prado'nun üçlü-bariyer
  etiketi, arena motorunun **birebir aynı fiziğiyle** — T+1 açılış fill + slippage,
  ATR stop / R·ATR hedef bariyerleri, gap kuralı, "aynı barda ikisi de → stop önce"
  muhafazakârlığı, süre bariyeri ve çift-yön komisyon. Etiket böylece "10 gün sonra
  yukarı mı?" değil, **"bu kurulumla açılan işlem maliyet-sonrası para kazanır mı?"**
  sorusuna cevap verir; ``benchmark`` (XU100 kapanış serisi) verilirse getiri
  endekse göre relatif ölçülür.

Sızıntı (look-ahead) yok: geleceği henüz bilinmeyen son barların etiketi NaN olur
ve eğitimden düşülür.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.analytics.indicators import add_core_indicators, atr

# ── Üçlü-bariyer fiziği — arena motoruyla AYNI kalmalı ────────────────────────
# (arena/replay.py: _ATR_STOP_MULT/_TARGET_RR · arena/config.py: bps değerleri)
TB_ATR_N = 14
TB_STOP_MULT = 2.0        # stop  = Close − 2·ATR   (analytics.risk ile aynı)
TB_TARGET_RR = 2.0        # hedef = Close + 2·2·ATR (risk/ödül = 2)
TB_COMMISSION_BPS = 5.0   # tek-yön komisyon (ExecutionConfig ile aynı)
TB_SLIPPAGE_BPS = 10.0    # tek-yön slippage proxy'si

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


CANDLE_COLUMNS = ["candle_net", "candle_net_3"]

# Formasyon yönü/gücü ağırlıkları — analytics.candlesticks.PATTERN_META ile uyumlu
_CANDLE_SIGN = {"boğa": 1.0, "ayı": -1.0, "nötr": 0.0}


def candle_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mum formasyonu özellikleri (S3): bar başına net boğa/ayı yükü.

    Literatür mum formasyonlarının tek başına AL/SAT üretmesini desteklemez ama
    bazı formasyonlarda kısa vadeli sinyal bulur (plan §3a) — bu yüzden formasyon
    burada **özellik bitine** çevrilir; ağırlığını veri (model) belirler.

    - ``candle_net``: o barın formasyon yükü, [-1, +1] (güç 1-3 / 3, yön işaretli;
      birden çok formasyon toplanıp kırpılır).
    - ``candle_net_3``: son 3 barın toplam yükü / 3 (kısa pencere teyidi).
    """
    from tradingagents.analytics.candlesticks import PATTERN_META, detect_candlesticks

    flags = detect_candlesticks(df)
    net = pd.Series(0.0, index=df.index)
    for code, (_, direction, strength) in PATTERN_META.items():
        if code in flags.columns:
            net = net + flags[code].astype(float) * _CANDLE_SIGN[direction] * (strength / 3.0)
    out = pd.DataFrame(index=df.index)
    out["candle_net"] = net.clip(-1.0, 1.0)
    out["candle_net_3"] = (net.rolling(3, min_periods=1).sum() / 3.0).clip(-1.0, 1.0)
    return out


def make_labels(df: pd.DataFrame, horizon: int = 10, threshold: float = 0.0) -> pd.Series:
    """İleriye dönük ikili etiket: ``horizon`` gün sonraki getiri > eşik → 1.

    ``threshold`` oransal (0.02 = %2). Son ``horizon`` bar geleceği bilinmediği
    için NaN olur ve eğitim tarafında düşülür.
    """
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    return (fwd > threshold).astype("float64").where(fwd.notna())


def triple_barrier_outcomes(
    df: pd.DataFrame,
    max_hold: int = 10,
    threshold: float = 0.0,
    benchmark: pd.Series | None = None,
    atr_n: int = TB_ATR_N,
    stop_mult: float = TB_STOP_MULT,
    target_rr: float = TB_TARGET_RR,
    commission_bps: float = TB_COMMISSION_BPS,
    slippage_bps: float = TB_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Her sinyal barı için üçlü-bariyer işlem sonucu simülasyonu.

    Bar ``t`` kapanışında sinyal varsayılır; işlem arena motoruyla aynı fizikle
    simüle edilir:

    - Giriş: ``Open[t+1]·(1+slip)`` + komisyon (T+1 açılış fill).
    - Bariyerler bar ``t`` kapanışından: stop = ``Close−stop_mult·ATR``,
      hedef = ``Close+stop_mult·target_rr·ATR`` (per-bar, motorla aynı formül).
    - Gün içi ``Low≤stop`` → stop; ``High≥target`` → hedef; ikisi birden →
      **muhafazakâr: stop önce**. Giriş günü sonrası ``Open≤stop`` gap'i →
      açılıştan çıkış. Stop/hedef fill'lerinde slippage yok (motorla aynı).
    - ``max_hold`` bar dolarsa süre bariyeri: ertesi açılıştan ``(1−slip)`` ile çıkış.
    - Çıkışta da komisyon düşülür.

    Dönen DataFrame (df ile aynı index): ``label`` (1 = maliyet-sonrası
    [benchmark-relatif] getiri > threshold), ``net_ret``, ``rel_ret``,
    ``exit_reason`` (stop|hedef|süre), ``hold_days``. Geleceği henüz
    bilinmeyen/işlem kurulamayan barlarda hepsi NaN.
    """
    n = len(df) if df is not None else 0
    out = pd.DataFrame(
        {"label": np.nan, "net_ret": np.nan, "rel_ret": np.nan,
         "exit_reason": pd.Series([None] * n, dtype=object),
         "hold_days": np.nan},
        index=df.index if df is not None else None,
    )
    if df is None or n < atr_n + 3:
        return out

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    a = atr(df, atr_n).to_numpy(float)

    bench = None
    if benchmark is not None and len(benchmark) > 0:
        bench = benchmark.reindex(df.index).ffill().to_numpy(float)

    comm = commission_bps / 10_000.0
    slip = slippage_bps / 10_000.0

    labels = np.full(n, np.nan)
    net_rets = np.full(n, np.nan)
    rel_rets = np.full(n, np.nan)
    reasons = np.full(n, None, dtype=object)
    holds = np.full(n, np.nan)

    for t in range(n - 1):
        if not (np.isfinite(a[t]) and a[t] > 0 and np.isfinite(c[t]) and c[t] > 0):
            continue
        stop = c[t] - stop_mult * a[t]
        target = c[t] + stop_mult * target_rr * a[t]
        if stop <= 0 or stop >= c[t]:      # motor _entry_allowed ile aynı ret
            continue
        entry_open = o[t + 1]
        if not (np.isfinite(entry_open) and entry_open > 0):
            continue
        entry_cost = entry_open * (1 + slip) * (1 + comm)

        exit_px = np.nan
        exit_j = -1
        reason = None
        last_hold_bar = t + max_hold
        for j in range(t + 1, min(last_hold_bar, n - 1) + 1):
            if not (np.isfinite(h[j]) and np.isfinite(l[j])):
                continue                    # eksik bar = işlemsiz gün (motorla aynı)
            gap_open = j > t + 1 and np.isfinite(o[j]) and o[j] <= stop
            if l[j] <= stop:                # ikisi birden görülse de stop önce
                exit_px = o[j] if gap_open else stop
                exit_j, reason = j, "stop"
                break
            if h[j] >= target:
                exit_px = (o[j] if (j > t + 1 and np.isfinite(o[j]) and o[j] >= target)
                           else target)
                exit_j, reason = j, "hedef"
                break
        if reason is None:                  # süre bariyeri → ertesi açılış (T+1 satış)
            j2 = last_hold_bar + 1
            if j2 >= n or not (np.isfinite(o[j2]) and o[j2] > 0):
                continue                    # gelecek bilinmiyor → etiket NaN
            exit_px = o[j2] * (1 - slip)
            exit_j, reason = j2, "süre"

        net = exit_px * (1 - comm) / entry_cost - 1.0
        rel = net
        if bench is not None and np.isfinite(bench[t]) and bench[t] > 0 \
                and np.isfinite(bench[exit_j]) and bench[exit_j] > 0:
            rel = net - (bench[exit_j] / bench[t] - 1.0)

        labels[t] = 1.0 if rel > threshold else 0.0
        net_rets[t] = net
        rel_rets[t] = rel
        reasons[t] = reason
        holds[t] = exit_j - t

    out["label"] = labels
    out["net_ret"] = net_rets
    out["rel_ret"] = rel_rets
    out["exit_reason"] = reasons
    out["hold_days"] = holds
    return out


def make_labels_triple_barrier(
    df: pd.DataFrame,
    max_hold: int = 10,
    threshold: float = 0.0,
    benchmark: pd.Series | None = None,
) -> pd.Series:
    """Üçlü-bariyer ikili etiket serisi (1 = işlem maliyet-sonrası kazanırdı)."""
    return triple_barrier_outcomes(df, max_hold, threshold, benchmark)["label"]


def build_training_set(
    df: pd.DataFrame, horizon: int = 10, threshold: float = 0.0,
    labeling: str = "triple_barrier", benchmark: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Hizalı (X, y) döndürür; NaN özellik/etiket satırları düşülmüş.

    ``labeling``: ``"triple_barrier"`` (varsayılan; ``horizon`` = süre bariyeri,
    ``benchmark`` verilirse XU100-relatif) ya da ``"fixed"`` (eski sabit-ufuk).
    """
    X = build_feature_frame(df)
    if labeling == "fixed":
        y = make_labels(df, horizon, threshold)
    else:
        y = make_labels_triple_barrier(df, horizon, threshold, benchmark)
    data = X.join(y.rename("_label")).dropna()
    return data[FEATURE_COLUMNS], data["_label"]

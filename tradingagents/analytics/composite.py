"""Kompozit teknik skor — tüm deterministik sinyallerin ağırlıklı birleşimi.

Tek bir hisse için trend, momentum, formasyon, mum, destek/direnç, sezonsallık
ve dip-al sinyalini [-100, +100] tek skora indirger; rejim modülü bu skorun
GÜVENİNİ ayarlar (oynak piyasa / TL stresi → güven düşer, böylece "neredeyse
%100 doğruluk" hedefinin gerçekçi karşılığı kurulur: emin olmadığında sus).

Ağırlıklar (toplam 100):
    trend 25 · momentum 20 · formasyon 20 · dip-al stratejisi 10 ·
    mum formasyonu 10 · destek/direnç 10 · sezonsallık 5

İki tüketici:
  - Streamlit "Teknik Analiz" sayfası → ``compute_composite`` (sayısal panel)
  - Market Analyst (LLM) → ``build_technical_brief`` (markdown brif; ajan bu
    hazır hesapları okuyup yorumlar, sayı uydurmaz)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.analytics.candlesticks import candle_score, recent_candle_hits
from tradingagents.analytics.indicators import add_core_indicators
from tradingagents.analytics.patterns import detect_patterns, pattern_score
from tradingagents.analytics.regime import RegimeResult, compute_regime
from tradingagents.analytics.seasonality import compute_seasonality, month_edge
from tradingagents.analytics.support_resistance import (
    compute_levels, nearest_levels, sr_score,
)
from tradingagents.dataflows.symbol_utils import is_bist_ticker
from tradingagents.strategy.dip_signal import compute_signals

# Ağırlıklar (toplam 100). money_flow eklendi (video: "para giriş-çıkışı hacimden
# farklıdır; yüksek hacimle düşüş = dağıtım"). Trend/momentum bir miktar bu
# bileşene yer açtı.
WEIGHTS = {
    "trend": 22, "momentum": 16, "pattern": 17, "money_flow": 13,
    "dip": 10, "candle": 8, "sr": 9, "seasonality": 5,
}


@dataclass
class CompositeResult:
    ok: bool
    ticker: str
    score: float = 0.0          # [-100, +100], guard + rejim çarpanı uygulanmış
    raw_score: float = 0.0      # guard/rejim uygulanmadan önce
    verdict: str = "NÖTR"       # "GÜÇLÜ AL" | "AL" | "NÖTR" | "SAT" | "GÜÇLÜ SAT"
    confidence: str = "düşük"   # "düşük" | "orta" | "yüksek"
    components: dict = field(default_factory=dict)   # bileşen -> [-1, +1]
    details: dict = field(default_factory=dict)      # bileşen -> Türkçe açıklama
    regime: RegimeResult | None = None
    patterns: list = field(default_factory=list)     # PatternHit listesi
    candles: list = field(default_factory=list)      # CandleHit listesi
    supports: list = field(default_factory=list)     # Level listesi
    resistances: list = field(default_factory=list)
    guards: dict = field(default_factory=dict)       # tuzak adı -> aktif mi (bool)
    warnings: list = field(default_factory=list)     # Türkçe tuzak uyarıları
    last_close: float = 0.0
    error: str = ""


def _fetch_daily(ticker: str, period: str = "5y") -> pd.DataFrame | None:
    """Günlük OHLCV çeker; hata/boş durumda None (asla istisna fırlatmaz)."""
    try:
        raw = yf.Ticker(ticker).history(period=period, interval="1d")
    except Exception:  # noqa: BLE001
        return None
    if raw is None or raw.empty or len(raw) < 60:
        return None
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    out = raw[keep].copy()
    # yeni yfinance/pandas nullable dtype (Float64/pd.NA) döndürebilir; pd.NA
    # "&"/karşılaştırmada NAType.__bool__ hatası verir → düz float64'e zorla.
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out.dropna(how="any")


def _trend_component(ind: pd.DataFrame) -> tuple[float, str]:
    """SMA hiyerarşisi + golden/death cross [-1, +1]."""
    last = ind.iloc[-1]
    close = float(last["Close"])
    score, notes = 0.0, []
    if pd.notna(last["sma200"]):
        above200 = close > float(last["sma200"])
        score += 0.4 if above200 else -0.4
        notes.append("fiyat SMA200 " + ("üzerinde" if above200 else "altında"))
    if pd.notna(last["sma50"]):
        above50 = close > float(last["sma50"])
        score += 0.3 if above50 else -0.3
        notes.append("SMA50 " + ("üzerinde" if above50 else "altında"))
    if pd.notna(last["sma50"]) and pd.notna(last["sma200"]):
        golden = float(last["sma50"]) > float(last["sma200"])
        score += 0.3 if golden else -0.3
        notes.append("golden cross" if golden else "death cross")
    return max(-1.0, min(1.0, score)), ", ".join(notes) or "SMA verisi yetersiz"


def _momentum_component(ind: pd.DataFrame) -> tuple[float, str]:
    """RSI + MACD histogram + stokastik [-1, +1]."""
    last = ind.iloc[-1]
    score, notes = 0.0, []
    if pd.notna(last["rsi14"]):
        r = float(last["rsi14"])
        if r < 30:
            score += 0.4; notes.append(f"RSI {r:.0f} aşırı satım")
        elif r > 70:
            score -= 0.4; notes.append(f"RSI {r:.0f} aşırı alım")
        else:
            score += (r - 50) / 50 * 0.3
            notes.append(f"RSI {r:.0f}")
    if pd.notna(last["macd_hist"]):
        h = float(last["macd_hist"])
        prev = float(ind["macd_hist"].iloc[-2]) if pd.notna(ind["macd_hist"].iloc[-2]) else h
        score += 0.3 if h > 0 else -0.3
        if abs(h) > abs(prev):
            score += 0.1 if h > 0 else -0.1
        notes.append("MACD hist " + ("pozitif" if h > 0 else "negatif")
                     + (" ve güçleniyor" if abs(h) > abs(prev) else ""))
    if pd.notna(last["stoch_k"]) and pd.notna(last["stoch_d"]):
        k, d = float(last["stoch_k"]), float(last["stoch_d"])
        if k < 20 and k > d:
            score += 0.2; notes.append("stokastik dipten dönüyor")
        elif k > 80 and k < d:
            score -= 0.2; notes.append("stokastik tepeden dönüyor")
    return max(-1.0, min(1.0, score)), ", ".join(notes) or "momentum verisi yetersiz"


def _money_flow_component(ind: pd.DataFrame) -> tuple[float, str]:
    """Para giriş-çıkışı [-1, +1] — OBV eğimi + hacim-fiyat yönü (dağıtım tespiti).

    Video: hacmin artması tek başına anlamlı değildir; YÖNÜ önemlidir. Yüksek
    hacimle düşüş = para çıkışı (mal boşaltma/dağıtım), yüksek hacimle yükseliş
    = para girişi (toplama).
    """
    close, vol, obv = ind["Close"], ind["Volume"], ind["obv"]
    score, notes = 0.0, []
    if len(obv) > 11 and pd.notna(obv.iloc[-1]) and pd.notna(obv.iloc[-11]):
        obv_slope = float(obv.iloc[-1] - obv.iloc[-11])
        if obv_slope > 0:
            score += 0.4; notes.append("OBV yükselişte (para girişi)")
        elif obv_slope < 0:
            score -= 0.4; notes.append("OBV düşüşte (para çıkışı)")
    # Son 10 barda yön-ağırlıklı hacim: yukarı/aşağı hacim dengesizliği
    ret = close.pct_change().tail(10)
    v = vol.tail(10)
    up_vol = float(v[ret > 0].sum())
    down_vol = float(v[ret < 0].sum())
    if up_vol > down_vol * 1.3:
        score += 0.4; notes.append("yüksek hacimli alış baskısı (toplama)")
    elif down_vol > up_vol * 1.3:
        score -= 0.4; notes.append("yüksek hacimli satış baskısı (DAĞITIM)")
    # Hacim teyidi: hacim ortalamanın üzerinde mi (hareketin gerçekliği)
    if len(vol) >= 20 and pd.notna(vol.iloc[-1]):
        vol_ma = float(vol.tail(20).mean())
        if vol_ma and float(vol.iloc[-1]) > 1.5 * vol_ma:
            notes.append("son bar hacmi ortalamanın belirgin üstünde")
    return max(-1.0, min(1.0, score)), ", ".join(notes) or "belirgin para akışı yok"


def _dip_component(df: pd.DataFrame) -> tuple[float, str]:
    """Mevcut dip-al stratejisinin (video.md) son-bar durumu [-1, +1]."""
    try:
        sig = compute_signals(df)
    except Exception:  # noqa: BLE001
        return 0.0, "dip-al hesaplanamadı"
    recent = sig.tail(8)
    if bool(recent["buy"].any()):
        return 1.0, "dip-al stratejisi son barlarda AL üretti"
    if bool(recent["sell"].any()):
        return -1.0, "dip-al stratejisi son barlarda SAT üretti"
    return 0.0, "dip-al sinyali yok"


def _detect_guards(df: pd.DataFrame, ind: pd.DataFrame, candle_val: float,
                   dip_val: float) -> tuple[dict, list[str]]:
    """Video tuzaklarını tespit eder: aşırı coşku/ATH ve düşen bıçak.

    Dönüş: (guards bool sözlüğü, Türkçe uyarı listesi). Bu guard'lar kompozit
    skorun POZİTİF (alış) tarafını kısıtlar — yön üretmez, hatalı alımı eler.
    """
    guards: dict[str, bool] = {"asiri_cosku": False, "dusen_bicak": False}
    warnings: list[str] = []
    last = ind.iloc[-1]
    close = float(last["Close"])

    # Aşırı coşku / tarihi zirve tuzağı: fiyat son 252 barın tepesine yakın +
    # RSI aşırı alımda → "tahtacının mal kitlediği" bölge (video: Ford örneği).
    win = df["High"].tail(252)
    if len(win) >= 60 and pd.notna(last["rsi14"]):
        ath = float(win.max())
        near_ath = close >= 0.97 * ath
        if near_ath and float(last["rsi14"]) > 72:
            guards["asiri_cosku"] = True
            warnings.append("⚠️ Aşırı coşku/tarihi zirve: fiyat zirveye yakın ve RSI "
                            "aşırı alımda — alım riskli (dağıtım/kitleme bölgesi olabilir).")

    # Düşen bıçak tuzağı: sert düşüş trendi (fiyat SMA200'ün çok altında, ADX
    # yüksek, eğim aşağı) + RSI dipte AMA dönüş teyidi yok → "ucuz" diye alma
    # (video: 90→9 TL düşen hisse 4 TL'ye de gidebilir).
    if pd.notna(last.get("sma200")) and pd.notna(last["rsi14"]) and pd.notna(last["adx14"]):
        far_below = close < 0.85 * float(last["sma200"])
        strong_down = float(last["adx14"]) >= 25
        sma50 = ind["sma50"]
        slope_down = bool(len(sma50) > 11 and pd.notna(sma50.iloc[-1])
                          and pd.notna(sma50.iloc[-11]) and sma50.iloc[-1] < sma50.iloc[-11])
        oversold = float(last["rsi14"]) < 38
        # Dönüş teyidi: dip-al AL sinyali ya da boğa mum yükü
        reversal = dip_val > 0 or candle_val > 0.3
        if far_below and strong_down and slope_down and oversold and not reversal:
            guards["dusen_bicak"] = True
            warnings.append("⚠️ Düşen bıçak: güçlü düşüş trendi sürerken oversold — "
                            "dönüş teyidi (para girişi/formasyon) gelmeden alım yapma.")
    return guards, warnings


def _verdict(score: float) -> str:
    if score >= 50:
        return "GÜÇLÜ AL"
    if score >= 20:
        return "AL"
    if score <= -50:
        return "GÜÇLÜ SAT"
    if score <= -20:
        return "SAT"
    return "NÖTR"


def compute_composite(ticker: str, df: pd.DataFrame | None = None) -> CompositeResult:
    """Hissenin kompozit teknik skorunu hesaplar. Asla istisna fırlatmaz.

    ``df`` verilirse (test/önbellek) ağdan veri çekilmez; verilmezse 5 yıllık
    günlük bar çekilir (sezonsallık için uzun geçmiş gerekli).
    """
    if df is None:
        df = _fetch_daily(ticker)
    if df is None or len(df) < 60:
        return CompositeResult(False, ticker, error="Yeterli fiyat verisi alınamadı (min ~60 bar).")

    ind = add_core_indicators(df)
    close = float(df["Close"].iloc[-1])
    components: dict[str, float] = {}
    details: dict[str, str] = {}

    components["trend"], details["trend"] = _trend_component(ind)
    components["momentum"], details["momentum"] = _momentum_component(ind)

    hits = detect_patterns(df)
    components["pattern"] = pattern_score(hits)
    details["pattern"] = ("; ".join(
        f"{h.name} ({'teyitli' if h.confirmed else 'oluşum'}, {h.direction})"
        for h in hits[:4]) or "aktif formasyon yok")

    candles = recent_candle_hits(df, lookback=5)
    components["candle"] = candle_score(df)
    details["candle"] = ("; ".join(f"{h.name} ({h.date})" for h in candles[:4])
                         or "son 5 barda formasyon yok")

    levels = compute_levels(df)
    supports, resistances = nearest_levels(levels, close)
    components["sr"] = sr_score(levels, close)
    details["sr"] = (f"yakın destek: {', '.join(str(s.price) for s in supports) or '—'} · "
                     f"yakın direnç: {', '.join(str(r.price) for r in resistances) or '—'}")

    components["money_flow"], details["money_flow"] = _money_flow_component(ind)
    components["dip"], details["dip"] = _dip_component(df)

    season = compute_seasonality(df)
    cur_month = int(df.index[-1].month) if hasattr(df.index[-1], "month") else 0
    components["seasonality"], details["seasonality"] = month_edge(season, cur_month)

    raw = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)

    # Video tuzak filtreleri: aşırı coşku alış tarafını yarıya indirir, düşen
    # bıçak pozitif (oversold) skoru sıfırlar — yanlış alımı eler, yön değil güven keser.
    guards, warnings = _detect_guards(df, ind, components["candle"], components["dip"])
    guarded = raw
    if guards["asiri_cosku"] and guarded > 0:
        guarded *= 0.4
    if guards["dusen_bicak"] and guarded > 0:
        guarded = min(guarded, 0.0)

    regime = compute_regime(df, is_bist=is_bist_ticker(ticker))
    mult = regime.confidence_mult if regime.ok else 0.8
    score = round(guarded * mult, 1)

    abs_score = abs(score)
    aligned = sum(1 for v in components.values()
                  if (v > 0.15 and score > 0) or (v < -0.15 and score < 0))
    if abs_score >= 40 and aligned >= 4 and mult >= 0.8 and not any(guards.values()):
        confidence = "yüksek"
    elif abs_score >= 20 and aligned >= 3:
        confidence = "orta"
    else:
        confidence = "düşük"

    return CompositeResult(
        ok=True, ticker=ticker, score=score, raw_score=round(raw, 1),
        verdict=_verdict(score), confidence=confidence,
        components={k: round(v, 2) for k, v in components.items()},
        details=details, regime=regime if regime.ok else None,
        patterns=hits, candles=candles, supports=supports, resistances=resistances,
        guards=guards, warnings=warnings, last_close=round(close, 2),
    )


def build_technical_brief(ticker: str, df: pd.DataFrame | None = None) -> str:
    """LLM'e enjekte edilecek markdown teknik brif üretir.

    Market Analyst bu blok sayesinde formasyon/sezonsallık/rejim gibi LLM'in
    ham fiyat verisinden güvenilir türetemeyeceği hesapları hazır okur. Veri
    yoksa bunu açıkça söyleyen tek satır döner (uydurma istatistik üretilmez).
    """
    res = compute_composite(ticker, df)
    if not res.ok:
        return f"<Deterministik teknik brif üretilemedi: {res.error}>"

    lines = [
        f"DETERMİNİSTİK TEKNİK BRİF — {ticker} (kapanış {res.last_close})",
        f"Kompozit skor: {res.score:+.0f}/100 → {res.verdict} (güven: {res.confidence}; "
        f"ham skor {res.raw_score:+.0f}, rejim çarpanı sonrası {res.score:+.0f})",
        "",
        "Bileşenler ([-1,+1] · ağırlık):",
    ]
    for key, w in WEIGHTS.items():
        lines.append(f"  - {key} ({w}): {res.components[key]:+.2f} — {res.details[key]}")
    if res.regime is not None:
        lines += ["", f"Piyasa rejimi: {res.regime.summary}"]
        if res.regime.try_note:
            lines.append(f"  {res.regime.try_note}")
    if res.patterns:
        lines += ["", "Aktif grafik formasyonları:"]
        for h in res.patterns[:5]:
            lines.append(f"  - {h.name} [{h.direction}] "
                         f"{'TEYİTLİ' if h.confirmed else 'oluşum aşamasında'} "
                         f"({h.start} → {h.end}) — {h.note}")
    sup = ", ".join(f"{s.price} ({s.label})" for s in res.supports) or "—"
    resis = ", ".join(f"{r.price} ({r.label})" for r in res.resistances) or "—"
    lines += ["", f"Destek seviyeleri: {sup}", f"Direnç seviyeleri: {resis}"]
    if res.warnings:
        lines += ["", "TUZAK UYARILARI (video kuralları — alım tarafını kısıtlar):"]
        lines += [f"  - {w}" for w in res.warnings]
    return "\n".join(lines)

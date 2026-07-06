"""\"Dipten Al / Tepeden Sat\" stratejisi — video.md kombinasyonu.

Deterministik teknik sinyal üretimi (LLM yok, maliyet yok). Bir hissenin
OHLCV verisini alıp şu indikatörleri hesaplar ve AL/SAT sinyali çıkarır:

  - **SMI** (Stochastic Momentum Index): %K=10, %D=3, sinyal EMA=3 (default).
  - **VWMA(7)** — SMI'ın hacim ağırlıklı hareketli ortalaması ("HAHO").
  - **Bollinger orta bandı** — 20 periyot SMA (+ üst/alt band).

AL sinyali (video kriterleri):
  1) SMI mavi çizgisi turuncu sinyal çizgisini **yukarı keser** (boğa kesişimi),
  2) bu kesişim **0 seviyesinin altında** olur,
  3) SMI kendi **VWMA(7)'sinin üzerine** çıkar (en kritik tetik),
  4) onay: kapanış **Bollinger orta bandının üzerinde**.

SAT sinyali (video: AL kadar hassas değil): SMI sinyali aşağı keser **ve**
fiyat Bollinger üst bandını zorlarken / hacim düşerken.

Zaman dilimleri: günlük (1d) ve 4 saatlik en kaliteli; 2 saatlik daha gürültülü
(video). Saatlik altı dilimler 1h veriden yeniden örneklenir.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Likit BIST hisseleri (BIST 30 + popüler) — otomatik tarama için varsayılan evren.
# Genişletmek için satır ekle; hepsi Yahoo ``.IS`` formatında.
BIST_POPULAR = [
    "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ISCTR.IS", "YKBNK.IS", "VAKBN.IS",
    "HALKB.IS", "SAHOL.IS", "KCHOL.IS", "TUPRS.IS", "EREGL.IS", "KRDMD.IS",
    "BIMAS.IS", "MGROS.IS", "SISE.IS", "ASELS.IS", "TCELL.IS", "TTKOM.IS",
    "FROTO.IS", "TOASO.IS", "ARCLK.IS", "PGSUS.IS", "PETKM.IS", "SASA.IS",
    "HEKTS.IS", "EKGYO.IS", "ENKAI.IS", "GUBRF.IS", "OYAKC.IS", "TAVHL.IS",
    "ALARK.IS", "VESTL.IS", "TKFEN.IS", "KONTR.IS", "SMRTG.IS", "ASTOR.IS",
    "CCOLA.IS", "ULKER.IS", "ISDMR.IS", "DOHOL.IS",
]

# BIST 30 endeksi (yaklaşık güncel bileşenler) — saat başı sürekli taranan
# çekirdek evren. Endeks bileşenleri dönemsel değişir; düzenlemek için satır
# ekle/çıkar ya da çalışma anında BIST30_TICKERS ortam değişkeniyle ez
# (virgülle ayrılmış). Hepsi Yahoo ``.IS`` formatında.
BIST30 = [
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "EKGYO.IS",
    "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS",
    "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KRDMD.IS", "MGROS.IS",  # KOZAL: Yahoo'da yok (2026-07)
    "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS",
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS",
]

# Kullanıcı/zaman-dilimi seçenekleri -> (yfinance interval, fetch period, resample kuralı)
INTERVALS = {
    "Günlük (1g)": ("1d", "2y", None),
    "4 Saatlik": ("1h", "720d", "4h"),
    "2 Saatlik": ("1h", "720d", "2h"),
    "1 Saatlik": ("1h", "720d", None),
}


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def compute_smi(df: pd.DataFrame, k: int = 10, d: int = 3, ema_len: int = 3):
    """SMI (Stochastic Momentum Index) ve sinyal çizgisini döndürür.

    Klasik çift-EMA yumuşatmalı SMI (TradingView ile uyumlu): %K uzunluğu k,
    %D yumuşatma d (iki kez), sinyal çizgisi = EMA(SMI, ema_len).
    """
    hh = df["High"].rolling(k).max()
    ll = df["Low"].rolling(k).min()
    center = (hh + ll) / 2.0
    rel = df["Close"] - center
    diff = (hh - ll)
    smooth_rel = _ema(_ema(rel, d), d)
    smooth_diff = _ema(_ema(diff, d), d)
    smi = 100.0 * smooth_rel / (smooth_diff / 2.0).replace(0, np.nan)
    smi = smi.astype("float64")
    signal = _ema(smi, ema_len)
    return smi, signal


def _vwma(series: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    """Hacim ağırlıklı hareketli ortalama (verilen seri üzerinde)."""
    num = (series * volume).rolling(n).sum()
    den = volume.rolling(n).sum().replace(0, np.nan)
    return (num / den).astype("float64")


def compute_signals(
    df: pd.DataFrame,
    arm_bars: int = 8,
    min_gap: float = 2.0,
    entry_ceil: float = 40.0,
    score_len: int = 14,
    min_sell: int = 2,
    need_rising: bool = True,
    bb_len: int = 20,
) -> pd.DataFrame:
    """OHLCV DataFrame'ine indikatör + teyitli buy/sell bayraklarını ekler.

    PRO mantık (Pine ``dip_al_sinyal.pine`` ile birebir): AL, SMI VWMA'ya sadece
    değdiğinde değil; (1) 0-altı boğa kesişiminden sonra ``arm_bars`` penceresinde,
    (2) SMI VWMA'yı ``min_gap`` farkıyla yukarı keserken, (3) SMI yükseliyorken,
    (4) SMI ``entry_ceil`` altındayken, (5) kapanış BB orta üstündeyken oluşur ve
    her dip döngüsünde yalnızca BİR kez verilir (gürültü/tekrar elenir).
    """
    out = df.copy()
    # yfinance/yeni pandas bazen nullable dtype (Float64/pd.NA) döndürür; bu
    # pd.NA değerleri "&"/karşılaştırmada NAType.__bool__ hatası verir. Tüm
    # OHLCV'yi düz float64'e (NaN, pd.NA değil) zorla.
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    vol = out["Volume"].astype("float64")
    smi, signal = compute_smi(out)
    out["smi"] = smi
    out["smi_signal"] = signal
    out["smi_vwma"] = _vwma(smi, vol, 7)
    out["bb_mid"] = out["Close"].rolling(bb_len).mean()
    bb_std = out["Close"].rolling(bb_len).std()
    out["bb_upper"] = out["bb_mid"] + 2 * bb_std
    out["bb_lower"] = out["bb_mid"] - 2 * bb_std
    # Fiyat skoru (video: "fiyat skoru 50 üstü") = kapanışın son N bardaki stokastiği.
    low_n = out["Low"].rolling(score_len).min()
    high_n = out["High"].rolling(score_len).max()
    out["price_score"] = 100.0 * (out["Close"] - low_n) / (high_n - low_n).replace(0, np.nan)

    # ── AL: dip döngüsünde tek, teyitli ──────────────────────────────────────
    new_dip = ((smi > signal) & (smi.shift() <= signal.shift()) & (smi < 0)).fillna(False)
    out["cross_up_below0"] = new_dip

    pos = pd.Series(np.arange(len(out)), index=out.index)
    last_dip_pos = pos.where(new_dip).ffill()
    dip_bars = pos - last_dip_pos                      # ilk dipten önce NaN
    in_window = (dip_bars <= arm_bars) & dip_bars.notna()

    vwma_break = (smi > out["smi_vwma"]).fillna(False) & (smi.shift() <= out["smi_vwma"].shift()).fillna(False)
    gap_ok = (smi - out["smi_vwma"]) >= min_gap
    rising = (smi > smi.shift()) if need_rising else pd.Series(True, index=out.index)
    not_high = smi < entry_ceil
    price_ok = out["Close"] > out["bb_mid"]

    cand = (in_window & vwma_break & gap_ok & rising & not_high & price_ok).fillna(False)
    cycle = new_dip.cumsum()
    cand_in_cycle = cand & (cycle > 0)
    first_in_cycle = cand_in_cycle.groupby(cycle).cumsum() == 1
    out["buy"] = (cand_in_cycle & first_in_cycle).fillna(False)

    # ── SAT: yükseliş döngüsünde tek, ≥min_sell teyit ────────────────────────
    bear_cross = (smi < signal).fillna(False) & (smi.shift() >= signal.shift()).fillna(False)
    near_upper = out["Close"] >= out["bb_upper"] * 0.99
    vol_falling = vol < vol.rolling(5).mean()
    score_high = out["price_score"] > 50
    conf = (near_upper.fillna(False).astype(int)
            + vol_falling.fillna(False).astype(int)
            + score_high.fillna(False).astype(int))
    cand_sell = (bear_cross & (smi > 0) & (conf >= min_sell)).fillna(False)
    sell_group = (smi < 0).fillna(False).cumsum()       # smi<0 olunca döngü sıfırlanır
    first_sell = cand_sell.groupby(sell_group).cumsum() == 1
    out["sell"] = (cand_sell & first_sell).fillna(False)
    return out


@dataclass
class StrategyResult:
    ok: bool
    ticker: str
    interval_label: str
    df: Optional[pd.DataFrame] = None
    status: str = ""            # "AL BÖLGESİ" | "SAT UYARISI" | "NÖTR"
    conditions: dict = field(default_factory=dict)
    signals: list = field(default_factory=list)   # [{date, type, price}]
    error: str = ""


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna(how="any")


def analyze(ticker: str, interval_label: str = "Günlük (1g)", arm_bars: int = 8) -> StrategyResult:
    """Hisseyi çek, sinyalleri hesapla, güncel durumu ve son sinyalleri döndür.

    Asla istisna fırlatmaz — hata durumunda ``ok=False`` ve ``error`` döner.
    """
    interval, period, rule = INTERVALS.get(interval_label, INTERVALS["Günlük (1g)"])
    try:
        raw = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception as exc:  # noqa: BLE001
        return StrategyResult(False, ticker, interval_label, error=f"Veri alınamadı: {exc}")

    if raw is None or raw.empty or len(raw) < 30:
        return StrategyResult(False, ticker, interval_label,
                              error="Yeterli fiyat verisi yok (en az ~30 bar gerekli).")

    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    raw = raw[keep]
    if rule:
        raw = _resample(raw, rule)
        if len(raw) < 30:
            return StrategyResult(False, ticker, interval_label,
                                  error="Yeniden örnekleme sonrası yeterli bar yok.")

    df = compute_signals(raw, arm_bars=arm_bars)
    last = df.iloc[-1]

    # Güncel durum koşulları (AL kriterleri, en son bar üzerinde). pd.notna
    # kısa-devre kontrolleri, az veri/boşluk olan hisselerde NA karşılaştırması
    # yüzünden "boolean value of NA is ambiguous" hatasını önler.
    recent = df.tail(arm_bars)
    c1_cross = bool(recent["cross_up_below0"].any())
    c3_vwma = bool(pd.notna(last["smi"]) and pd.notna(last["smi_vwma"])
                   and last["smi"] > last["smi_vwma"])
    c4_bb = bool(pd.notna(last["Close"]) and pd.notna(last["bb_mid"])
                 and last["Close"] > last["bb_mid"])
    conditions = {
        "1) SMI 0-altı boğa kesişimi (son barlar)": c1_cross,
        "2) SMI kendi VWMA(7) üzerinde": c3_vwma,
        "3) Kapanış Bollinger orta bandı üzerinde": c4_bb,
    }

    recent_buy = bool(df["buy"].tail(arm_bars).any())
    recent_sell = bool(df["sell"].tail(arm_bars).any())
    if recent_buy or (c1_cross and c3_vwma and c4_bb):
        status = "AL BÖLGESİ"
    elif recent_sell:
        status = "SAT UYARISI"
    else:
        status = "NÖTR"

    sig_rows = df[df["buy"] | df["sell"]].tail(15)
    signals = [
        {
            "date": idx.strftime("%Y-%m-%d %H:%M") if hasattr(idx, "strftime") else str(idx),
            "type": "AL" if bool(r["buy"]) else "SAT",
            "price": round(float(r["Close"]), 2),
        }
        for idx, r in sig_rows.iterrows()
    ]

    return StrategyResult(
        ok=True, ticker=ticker, interval_label=interval_label, df=df,
        status=status, conditions=conditions, signals=signals,
    )


# Tarama sıralaması: AL bölgesi en üstte, sonra nötr (kriter yoğunluğuna göre),
# sonra sat uyarısı, en sonda veri alınamayanlar.
_SCAN_ORDER = {"AL BÖLGESİ": 0, "NÖTR": 1, "SAT UYARISI": 2, "—": 3}


def scan(
    tickers: Optional[list[str]] = None,
    interval_label: str = "Günlük (1g)",
    max_workers: int = 8,
) -> list[dict]:
    """Bir hisse evrenini tarayıp her birinin güncel strateji durumunu döndürür.

    Paralel çeker (ağ-yoğun). Sonuç AL bölgesindekiler en üstte olacak şekilde
    sıralı bir liste: her öğe ``{ticker, status, met, close, smi, ok}``. ``met``
    sağlanan AL kriteri sayısıdır (0-3). Asla istisna fırlatmaz.
    """
    universe = tickers if tickers is not None else BIST_POPULAR

    def _one(tk: str) -> dict:
        try:
            res = analyze(tk, interval_label)
            if not res.ok or res.df is None:
                return {"ticker": tk, "status": "—", "met": 0,
                        "close": None, "smi": None, "ok": False}
            last = res.df.iloc[-1]
            return {
                "ticker": tk,
                "status": res.status,
                "met": sum(1 for v in res.conditions.values() if v),
                "close": round(float(last["Close"]), 2) if pd.notna(last["Close"]) else None,
                "smi": round(float(last["smi"]), 1) if pd.notna(last["smi"]) else None,
                "ok": True,
            }
        except Exception:  # noqa: BLE001 — tek hisse hiçbir zaman taramayı çökertmesin
            return {"ticker": tk, "status": "—", "met": 0,
                    "close": None, "smi": None, "ok": False}

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_one, universe):
            results.append(res)

    results.sort(key=lambda r: (_SCAN_ORDER.get(r["status"], 9), -r["met"]))
    return results

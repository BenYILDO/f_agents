"""Seri davranışı — trend mi, ortalamaya dönücü mü? (Variance Ratio + Hurst).

Hangi sinyal ailesinin (momentum vs mean-reversion) kullanılacağını matematiksel
olarak belirler:

  - :func:`variance_ratio` — Lo & MacKinlay (1988) varyans oranı testi. VR>1
    (z anlamlı) → trendli/süreklilik (momentum); VR<1 → ortalamaya dönüş (reversal);
    VR≈1 → rastgele yürüyüş.
  - :func:`hurst_exponent` — Hurst üsteli. H>0.5 süreklilik, H<0.5 dönüş, ~0.5 rastgele.
  - :func:`classify_behavior` — ikisini birleştirip 'trend'/'reversal'/'rastgele' der
    ve hangi sinyal modunun uygun olduğunu söyler.

Kaynaklar: Lo & MacKinlay (1988); Mandelbrot/Hurst (R/S analizi).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clean(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a)]


@dataclass
class VarianceRatioResult:
    ok: bool
    q: int = 2
    vr: float = 1.0
    z: float = 0.0          # homoskedastik z istatistiği
    reason: str = ""


def variance_ratio(returns, q: int = 5) -> VarianceRatioResult:
    """Lo-MacKinlay varyans oranı (örtüşen tahminci, sapma düzeltmeli)."""
    r = _clean(returns)
    T = r.size
    if T < q * 4 or q < 2:
        return VarianceRatioResult(False, q, reason="Yetersiz veri.")
    mu = r.mean()
    var1 = r.var(ddof=1)
    if var1 <= 0:
        return VarianceRatioResult(False, q, reason="Sıfır varyans.")
    # q-periyot örtüşen getiriler
    csum = np.cumsum(r)
    rq = csum[q - 1:] - np.concatenate(([0.0], csum[:-q]))  # uzunluk T-q+1
    m = q * (T - q + 1) * (1 - q / T)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    z_denom = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    z = (vr - 1) / z_denom if z_denom > 0 else 0.0
    return VarianceRatioResult(True, q, round(float(vr), 4), round(float(z), 3))


def hurst_exponent(level_series, max_lag: int = 50) -> float | None:
    """Yapı-fonksiyonu yöntemiyle Hurst üsteli (seviye/kümülatif seri üzerinde)."""
    s = _clean(level_series)
    n = s.size
    if n < 120:
        return None
    lags = range(2, min(max_lag, n // 2))
    xs, ys = [], []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        sd = diff.std()
        if sd > 0:
            xs.append(np.log(lag))
            ys.append(np.log(sd))
    if len(xs) < 4:
        return None
    slope = np.polyfit(xs, ys, 1)[0]
    return round(float(slope), 4)


@dataclass
class BehaviorResult:
    ok: bool
    behavior: str = "rastgele"      # 'trend' | 'reversal' | 'rastgele'
    mode: str = "belirsiz"          # önerilen sinyal modu: 'momentum'|'mean-reversion'|'nötr'
    vr: float = 1.0
    vr_z: float = 0.0
    hurst: float | None = None
    note: str = ""
    reason: str = ""


def classify_behavior(returns, q: int = 5) -> BehaviorResult:
    """Varyans oranı + Hurst → davranış sınıfı ve önerilen sinyal modu."""
    r = _clean(returns)
    if r.size < 120:
        return BehaviorResult(False, reason="Yetersiz veri (≥120 getiri).")
    vrr = variance_ratio(r, q)
    h = hurst_exponent(np.cumsum(r))  # kümülatif (fiyat-benzeri) seri üzerinde

    trend_votes = 0
    rev_votes = 0
    if vrr.ok:
        if vrr.z > 1.5:
            trend_votes += 1
        elif vrr.z < -1.5:
            rev_votes += 1
    if h is not None:
        if h > 0.55:
            trend_votes += 1
        elif h < 0.45:
            rev_votes += 1

    if trend_votes > rev_votes:
        behavior, mode = "trend", "momentum"
    elif rev_votes > trend_votes:
        behavior, mode = "reversal", "mean-reversion"
    else:
        behavior, mode = "rastgele", "nötr"

    note = (f"VR(q={q})={vrr.vr} (z={vrr.z})"
            + (f" · Hurst={h}" if h is not None else "")
            + f" → {behavior}")
    return BehaviorResult(True, behavior=behavior, mode=mode,
                          vr=vrr.vr, vr_z=vrr.z, hurst=h, note=note)

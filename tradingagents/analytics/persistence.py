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

import math
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
    z_star: float = 0.0     # heteroskedastisiteye dirençli z* (Lo-MacKinlay 1988)
    reason: str = ""


def variance_ratio(returns, q: int = 5) -> VarianceRatioResult:
    """Lo-MacKinlay varyans oranı (örtüşen tahminci, sapma düzeltmeli).

    Homoskedastik ``z`` yanında, değişen-varyansa (volatilite kümelenmesi)
    dirençli ``z_star`` da hesaplanır — finansal getiriler heteroskedastiktir,
    bu yüzden karar için ``z_star`` tercih edilir (Lo-MacKinlay 1988, Teorem 2).
    """
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

    # Heteroskedastisiteye dirençli z* — Lo-MacKinlay (1988) eş. 4.3
    e = r - mu
    denom = float(np.sum(e ** 2)) ** 2
    theta = 0.0
    if denom > 0:
        for j in range(1, q):
            num = float(np.sum((e[j:] ** 2) * (e[:-j] ** 2)))
            delta_j = num / denom
            theta += (2.0 * (q - j) / q) ** 2 * delta_j
    z_star = (math.sqrt(T) * (vr - 1) / math.sqrt(theta)) if theta > 0 else 0.0
    return VarianceRatioResult(True, q, round(float(vr), 4), round(float(z), 3),
                               round(float(z_star), 3))


def _expected_rs(n: int) -> float:
    """Rastgele yürüyüş altında beklenen R/S (Anis-Lloyd 1976 / Peters düzeltmesi).

    Kısa pencerelerde R/S yukarı saparak yatay piyasayı trendliymiş gibi gösterir;
    bu teorik beklenti çıkarılarak (regresyonda) o sapma giderilir.
    """
    if n < 2:
        return float("nan")
    front = (n - 0.5) / n
    i = np.arange(1, n)
    back = float(np.sum(np.sqrt((n - i) / i)))
    if n <= 340:
        middle = math.gamma((n - 1) / 2.0) / (math.sqrt(math.pi) * math.gamma(n / 2.0))
    else:  # büyük n'de Gamma taşar → asimptotik biçim
        middle = 1.0 / math.sqrt(n * math.pi / 2.0)
    return front * middle * back


def _empirical_rs(chunk: np.ndarray) -> float | None:
    """Tek bir parça için yeniden-ölçeklenmiş aralık (R/S)."""
    mean = chunk.mean()
    dev = np.cumsum(chunk - mean)
    rng = float(dev.max() - dev.min())
    s = float(chunk.std(ddof=0))
    return rng / s if s > 0 else None


def hurst_exponent(level_series, max_lag: int = 50) -> float | None:
    """Anis-Lloyd düzeltmeli Hurst üsteli (R/S analizi).

    ``level_series`` seviye/kümülatif (fiyat-benzeri) seridir; içsel olarak
    getiri farklarına dönüştürülüp parça-parça R/S hesaplanır ve teorik beklentiyle
    sapması düzeltilir: H = 0.5 + eğim[ log(n) , log(R/S_emp) − log(R/S_beklenen) ].
    """
    s = _clean(level_series)
    if s.size < 120:
        return None
    incr = np.diff(s)
    N = incr.size
    if N < 64:
        return None
    # Pencere boyutları (geometrik) — Anis-Lloyd beklentisi log-log regresyona girer.
    sizes = sorted({int(round(x)) for x in np.unique(
        np.floor(np.geomspace(8, min(N // 2, max_lag * 4), num=12)))})
    xs, ys = [], []
    for n in sizes:
        if n < 8 or n > N:
            continue
        k = N // n
        rs_vals = [v for v in (_empirical_rs(incr[i * n:(i + 1) * n]) for i in range(k))
                   if v is not None]
        if not rs_vals:
            continue
        ers = _expected_rs(n)
        if not (ers and ers > 0):
            continue
        xs.append(math.log(n))
        ys.append(math.log(float(np.mean(rs_vals))) - math.log(ers))
    if len(xs) < 4:
        return None
    slope = float(np.polyfit(xs, ys, 1)[0])
    return round(0.5 + slope, 4)


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
        # Heteroskedastisiteye dirençli z* ile oy ver (volatilite kümelenmesine sağlam).
        if vrr.z_star > 1.5:
            trend_votes += 1
        elif vrr.z_star < -1.5:
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

    note = (f"VR(q={q})={vrr.vr} (z*={vrr.z_star})"
            + (f" · Hurst={h}" if h is not None else "")
            + f" → {behavior}")
    return BehaviorResult(True, behavior=behavior, mode=mode,
                          vr=vrr.vr, vr_z=vrr.z, hurst=h, note=note)

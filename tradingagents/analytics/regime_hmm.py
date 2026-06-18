"""Rejim tespiti — boğa/ayı (HMM) + volatilite rejimi + GARCH-lite vol.

Trend rejiminde momentum, yatay/ayı rejiminde mean-reversion çalışır; sistem
hangi rejimde olduğunu bilmeli ki eşikleri/stratejisini ona göre seçsin.

  - :func:`fit_gaussian_hmm` — 2 durumlu Gauss HMM, EM (Baum-Welch) ile, saf
    numpy. Düşük-ortalama durum = ayı, yüksek-ortalama = boğa.
  - :func:`ewma_vol` — RiskMetrics tarzı EWMA koşullu volatilite (GARCH-lite).
  - :func:`detect_regime` — endeks getirisinden trend + vol rejimini birleştirir.

Kaynaklar: Hamilton (1989) Markov-switching; RiskMetrics (EWMA varyans).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_VAR_FLOOR = 1e-10
_TRADING_DAYS = 252


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def _gauss(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    var = max(var, _VAR_FLOOR)
    return np.exp(-0.5 * (x - mu) ** 2 / var) / math.sqrt(2 * math.pi * var)


@dataclass
class HMMFit:
    ok: bool
    means: tuple = (0.0, 0.0)        # (durum0, durum1) ortalama getiri
    variances: tuple = (0.0, 0.0)
    trans: tuple = ()                # 2x2 geçiş matrisi (düz liste)
    bull_state: int = 1
    posterior_bull: float = 0.5      # son gözlemde boğa olasılığı
    gamma_bull: list = None          # her gözlemde boğa olasılığı
    reason: str = ""


def fit_gaussian_hmm(returns, n_iter: int = 40, seed: int = 0) -> HMMFit:
    """2 durumlu Gauss HMM'i EM ile eğitir (ölçekli forward-backward)."""
    x = _clean(returns)
    n = x.size
    if n < 60 or x.std() == 0:
        return HMMFit(False, reason="Yetersiz/sabit veri.")

    # Başlangıç: medyana göre iki küme
    med = np.median(x)
    lo, hi = x[x <= med], x[x > med]
    mu = np.array([lo.mean() if lo.size else x.min(), hi.mean() if hi.size else x.max()])
    var = np.array([max(lo.var(), _VAR_FLOOR), max(hi.var(), _VAR_FLOOR)])
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])

    gamma = np.full((n, 2), 0.5)
    for _ in range(n_iter):
        B = np.column_stack([_gauss(x, mu[0], var[0]), _gauss(x, mu[1], var[1])])
        B = np.clip(B, 1e-300, None)
        # Forward (ölçekli)
        alpha = np.zeros((n, 2))
        c = np.zeros(n)
        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum() or 1.0
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum() or 1.0
            alpha[t] /= c[t]
        # Backward
        beta = np.zeros((n, 2))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / (c[t + 1] or 1.0)
        # Posteriors
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        # xi toplamı
        xi_sum = np.zeros((2, 2))
        for t in range(n - 1):
            denom = (alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]).sum()
            if denom > 0:
                xi_sum += (alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]) / denom
        # M-step
        pi = gamma[0] + 1e-12
        pi /= pi.sum()
        A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)
        gsum = gamma.sum(axis=0) + 1e-300
        mu = (gamma * x[:, None]).sum(axis=0) / gsum
        var = (gamma * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / gsum
        var = np.maximum(var, _VAR_FLOOR)

    bull = int(np.argmax(mu))
    return HMMFit(
        ok=True,
        means=(round(float(mu[0]), 6), round(float(mu[1]), 6)),
        variances=(round(float(var[0]), 8), round(float(var[1]), 8)),
        trans=tuple(np.round(A, 4).ravel().tolist()),
        bull_state=bull,
        posterior_bull=round(float(gamma[-1, bull]), 4),
        gamma_bull=np.round(gamma[:, bull], 4).tolist(),
    )


def ewma_vol(returns, lam: float = 0.94, annualize: bool = True) -> float:
    """RiskMetrics EWMA koşullu volatilitesi (GARCH-lite) — son tahmin."""
    r = _clean(returns)
    if r.size < 5:
        return 0.0
    var = r[0] ** 2
    for x in r[1:]:
        var = lam * var + (1 - lam) * x ** 2
    vol = math.sqrt(max(var, 0.0))
    return float(vol * math.sqrt(_TRADING_DAYS)) if annualize else float(vol)


@dataclass
class RegimeState:
    ok: bool
    trend: str = "belirsiz"          # 'boğa' | 'ayı' | 'belirsiz'
    posterior_bull: float = 0.5
    vol_regime: str = "normal"       # 'düşük' | 'normal' | 'yüksek'
    ann_vol: float = 0.0
    note: str = ""
    reason: str = ""


def volatility_regime(returns, window: int = 20) -> tuple[str, float]:
    """Güncel EWMA vol'ü, kendi tarihsel dağılımının terlerine göre sınıflar."""
    r = _clean(returns)
    if r.size < 60:
        return "normal", 0.0
    rolling = (np.asarray([
        ewma_vol(r[max(0, i - window): i + 1], annualize=True)
        for i in range(window, r.size)
    ]))
    cur = ewma_vol(r, annualize=True)
    if rolling.size < 10:
        return "normal", round(cur, 4)
    lo, hi = np.quantile(rolling, [0.33, 0.67])
    label = "düşük" if cur <= lo else "yüksek" if cur >= hi else "normal"
    return label, round(cur, 4)


def detect_regime(returns) -> RegimeState:
    """Endeks getirisinden trend (HMM) + volatilite rejimini birleştirir."""
    r = _clean(returns)
    if r.size < 60:
        return RegimeState(False, reason="Yetersiz veri (≥60 getiri).")
    fit = fit_gaussian_hmm(r)
    if not fit.ok:
        return RegimeState(False, reason=fit.reason)
    pb = fit.posterior_bull
    trend = "boğa" if pb >= 0.6 else "ayı" if pb <= 0.4 else "belirsiz"
    vol_label, ann = volatility_regime(r)
    note = f"Boğa olasılığı %{pb*100:.0f} · yıllık vol ~%{ann*100:.0f} ({vol_label})"
    return RegimeState(True, trend=trend, posterior_bull=pb,
                       vol_regime=vol_label, ann_vol=ann, note=note)

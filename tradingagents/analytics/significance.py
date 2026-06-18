"""İstatistiksel anlamlılık — Sharpe çıkarımı, Deflated Sharpe, bootstrap.

Bir stratejinin/sinyalin "şans eseri mi yoksa gerçek edge mi" olduğunu ölçer.
Veri-madenciliği (çok sayıda deneme arasından en iyisini seçme) yanlılığını
düzeltir. Tamamı saf numpy + standart kütüphane (scipy gerekmez).

Kaynaklar:
  - Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014).
  - Bailey & López de Prado, "The Probabilistic Sharpe Ratio" (2012).
  - Sullivan, Timmermann & White (1999) — bootstrap ile data-snooping.
"""

from __future__ import annotations

import math

import numpy as np

_EULER = 0.5772156649015329  # Euler-Mascheroni sabiti


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Ters normal CDF (Acklam yaklaşımı) — scipy'siz."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def sharpe_ratio(returns, periods: int = 252) -> float:
    """Yıllıklandırılmış Sharpe (risksiz oran=0)."""
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0


def _moments(returns) -> tuple[int, float, float, float]:
    """(n, gözlem-başına Sharpe, çarpıklık, basıklık)."""
    r = _clean(returns)
    n = r.size
    sd = r.std(ddof=1) if n > 1 else 0.0
    if n < 3 or sd == 0:
        return n, 0.0, 0.0, 3.0
    sr = r.mean() / sd
    z = (r - r.mean()) / r.std(ddof=0)
    return n, float(sr), float((z**3).mean()), float((z**4).mean())


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """PSR — gözlemlenen Sharpe'ın ``sr_benchmark``'ı (gözlem-başına) aşma olasılığı.

    Çarpıklık ve basıklığı hesaba katar (finansal getiriler normal değildir).
    """
    n, sr, skew, kurt = _moments(returns)
    if n < 3:
        return 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr**2))
    return _norm_cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """N bağımsız deneme arasından beklenen MAKSİMUM Sharpe (gözlem-başına).

    Seçim yanlılığının "çıtası": rastgele N strateji denersen, en iyisinin
    Sharpe'ı bu kadar yüksek çıkar — gerçek edge bunu aşmalı.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    return math.sqrt(sr_variance) * (
        (1 - _EULER) * _norm_ppf(1 - 1.0 / n_trials)
        + _EULER * _norm_ppf(1 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(returns, n_trials: int, sr_variance: float) -> float:
    """DSR — seçim yanlılığına göre düzeltilmiş PSR. >0.95 güçlü kanıt sayılır."""
    sr_star = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_star)


def block_bootstrap_pvalue(returns, n_boot: int = 1000, block: int = 5,
                           seed: int = 42) -> float:
    """H0: gerçek Sharpe ≤ 0 için tek-yönlü blok-bootstrap p-değeri.

    Otokorelasyonu korumak için dairesel blok bootstrap; getiriler H0 altında
    sıfır-ortalamaya kaydırılır. Küçük p (örn. <0.05) → edge şans değil.
    """
    r = _clean(returns)
    n = r.size
    if n < 20:
        return 1.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 1.0
    obs = r.mean() / sd
    r0 = r - r.mean()  # H0: ortalama 0
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    count = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        s = r0[idx]
        ssd = s.std(ddof=1)
        bsr = s.mean() / ssd if ssd > 0 else 0.0
        if bsr >= obs:
            count += 1
    return (count + 1) / (n_boot + 1)

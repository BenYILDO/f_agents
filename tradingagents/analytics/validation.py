"""Backtest doğruluğu — sızıntısız bölme, walk-forward, PBO ve risk metrikleri.

Aşırı-uyumu (overfitting) ve sızıntıyı (look-ahead) engelleyen değerlendirme
araçları:

  - :func:`purged_kfold_indices` — purge + embargo'lu K-kat (López de Prado);
    test penceresine bitişik eğitim gözlemleri atılır.
  - :func:`walk_forward_indices` — kayan/genişleyen pencere ileri-doğrulama.
  - :func:`pbo_cscv` — Backtest Overfitting Olasılığı (CSCV): seçilen "en iyi"
    konfigürasyonun örnek-dışı medyanın altına düşme olasılığı.
  - Risk metrikleri: :func:`max_drawdown`, :func:`sortino_ratio`, :func:`cagr`,
    :func:`turnover`.

Kaynaklar: Bailey, Borwein, López de Prado & Zhu — PBO; López de Prado,
*Advances in Financial Machine Learning* (purged CV).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

_TRADING_DAYS = 252


# ── Risk metrikleri ──────────────────────────────────────────────────────────
def max_drawdown(equity) -> float:
    """Maksimum tepe-dip düşüşü (negatif yüzde)."""
    e = np.asarray(equity, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 2:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float((e / peak - 1).min() * 100)


def sortino_ratio(returns, periods: int = _TRADING_DAYS, target: float = 0.0) -> float:
    """Sortino — yalnız aşağı-yönlü oynaklığı cezalandıran risk-ayarlı getiri."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    downside = r[r < target]
    dd = math.sqrt((downside ** 2).mean()) if downside.size else 0.0
    if dd == 0:
        return 0.0
    return float((r.mean() - target) / dd * math.sqrt(periods))


def cagr(equity, periods: int = _TRADING_DAYS) -> float:
    """Bileşik yıllık büyüme oranı (%)."""
    e = np.asarray(equity, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 2 or e[0] <= 0:
        return 0.0
    years = e.size / periods
    return float((e[-1] / e[0]) ** (1 / years) - 1) * 100 if years > 0 else 0.0


def turnover(weights) -> float:
    """Ortalama mutlak ağırlık değişimi (işlem maliyeti vekili)."""
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    if w.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean())


# ── Sızıntısız bölme ─────────────────────────────────────────────────────────
def purged_kfold_indices(n: int, k: int = 5, embargo: int = 0):
    """Purge + embargo'lu K-kat: test penceresine bitişik eğitim gözlemleri atılır."""
    if n < k or k < 2:
        return []
    folds = np.array_split(np.arange(n), k)
    out = []
    for fold in folds:
        a, b = fold[0], fold[-1]
        lo, hi = max(0, a - embargo), min(n - 1, b + embargo)
        test = np.arange(a, b + 1)
        train = np.array([i for i in range(n) if i < lo or i > hi])
        out.append((train, test))
    return out


def walk_forward_indices(n: int, train: int, test: int, step: int | None = None):
    """Kayan pencere ileri-doğrulama: geçmişte eğit → hemen sonrasında test et."""
    step = step or test
    out = []
    start = 0
    while start + train + test <= n:
        tr = np.arange(start, start + train)
        te = np.arange(start + train, start + train + test)
        out.append((tr, te))
        start += step
    return out


# ── Backtest aşırı-uyum olasılığı (PBO / CSCV) ───────────────────────────────
def _sharpe_cols(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=1)
    sd[sd == 0] = np.nan
    return mu / sd


def pbo_cscv(perf_matrix, n_blocks: int = 10) -> dict:
    """Backtest Overfitting Olasılığı — combinatorially-symmetric cross-validation.

    ``perf_matrix``: (T gözlem × N konfigürasyon) getiri matrisi. Her IS/OOS
    bölünmesinde IS'te en iyi konfig seçilir; OOS sıralaması bakılır. PBO = en iyi
    IS konfigin OOS medyanın altına düşme oranı (yüksek = aşırı-uyum riski).
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < n_blocks * 2:
        return {"ok": False, "reason": "Yetersiz veri (T≥2·blok, N≥2 gerekli)."}
    if n_blocks % 2 == 1:
        n_blocks -= 1
    blocks = np.array_split(np.arange(M.shape[0]), n_blocks)
    half = n_blocks // 2
    logits = []
    for is_sel in combinations(range(n_blocks), half):
        is_rows = np.concatenate([blocks[i] for i in is_sel])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in is_sel])
        is_sr = _sharpe_cols(M[is_rows])
        oos_sr = _sharpe_cols(M[oos_rows])
        if np.all(np.isnan(is_sr)):
            continue
        n_star = int(np.nanargmax(is_sr))
        # OOS göreli sıralaması (0..1); rastgele ~0.5
        valid = np.isfinite(oos_sr)
        rank = (oos_sr[valid] < oos_sr[n_star]).sum() / max(1, valid.sum() - 1)
        w = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
    if not logits:
        return {"ok": False, "reason": "Bölünme üretilemedi."}
    logits = np.array(logits)
    pbo = float((logits < 0).mean())
    return {
        "ok": True,
        "pbo": round(pbo, 3),               # 0 iyi, →1 aşırı-uyum
        "n_splits": len(logits),
        "median_logit": round(float(np.median(logits)), 3),
    }

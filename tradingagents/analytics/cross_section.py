"""Kesitsel sıralama — BIST evrenini tek bir z-skorlanmış sinyalle sırala.

Hisseleri birbirine göre puanlar: her metrik (momentum, teknik skor, rasyo skoru…)
winsorize edilip (uç değerler kırpılır) z-skora çevrilir, ağırlıklı toplanır ve
evren bu birleşik skora göre sıralanır. Eksik metrik nötr (0) sayılır → cezalanmaz.
İsteğe bağlı sektör nötralizasyonu (grup ortalamasını çıkar) ile sektör yanlılığı
giderilir.

Kaynak: Jegadeesh & Titman (1993) kesitsel momentum; standart faktör z-skorlama.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def demean_by_group(s: pd.Series, groups: pd.Series) -> pd.Series:
    """Sektör nötralizasyonu — her grubun ortalamasını çıkarır."""
    s = pd.to_numeric(s, errors="coerce")
    return s - s.groupby(groups).transform("mean")


def cross_sectional_score(
    metrics: pd.DataFrame,
    weights: dict[str, float],
    winsor: float = 0.05,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Metrik tablosunu birleşik z-skora göre sıralar (yüksek = daha iyi).

    ``metrics``: index=ticker, kolonlar metrik adları. ``weights``: metrik→ağırlık
    (işaretli; + yüksek-iyi, − düşük-iyi). Eksik metrik 0 (nötr) sayılır.
    """
    cols = [c for c in weights if c in metrics.columns]
    if not cols or metrics.empty:
        out = metrics.copy()
        out["composite_z"] = 0.0
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    z = pd.DataFrame(index=metrics.index)
    for c in cols:
        s = winsorize(metrics[c], winsor, 1 - winsor)
        if groups is not None:
            s = demean_by_group(s, groups)
        zc = zscore(s).fillna(0.0)
        z[c] = zc * np.sign(weights[c])

    wsum = sum(abs(weights[c]) for c in cols) or 1.0
    composite = sum(z[c] * abs(weights[c]) for c in cols) / wsum

    out = metrics.copy()
    out["composite_z"] = composite.round(3)
    out = out.sort_values("composite_z", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def top_n(ranked: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """En yüksek birleşik skorlu N hisse."""
    return ranked.head(n)

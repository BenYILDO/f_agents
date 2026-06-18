"""Kalibre olasılık — modelin '%70' dediğinde gerçekten ~%70 çıkıp çıkmadığı.

Mevcut ML özelliklerinden (``ml.features``) ufuk-h için yukarı olasılığı üretir,
ama bunu **zaman-serisi bölmesiyle (sızıntısız)** elde edilen örnek-dışı (OOS)
tahminler üzerinde **isotonic regresyon** ile kalibre eder ve **Brier skoru** ile
güvenilirliğini ölçer (0 mükemmel, 0.25 = yazı-tura). Böylece olasılık "ham model
çıktısı" değil, geçmişte tutarlılığı kanıtlanmış bir sayı olur.

Ağır (5-kat GBDT) — talep üzerine / gecelik iş içindir, saatlik cron'da değil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingagents.ml.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    build_training_set,
)


@dataclass
class ProbabilityResult:
    ok: bool
    ticker: str = ""
    horizon: int = 10
    p_up: float | None = None        # kalibre yukarı olasılığı
    p_up_raw: float | None = None    # kalibrasyon öncesi ham olasılık
    brier: float | None = None       # güvenilirlik (düşük iyi)
    auc: float | None = None
    n_samples: int = 0
    reason: str = ""


def _impute(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return (frame.replace([np.inf, -np.inf], np.nan)
            .fillna(medians).fillna(0.0))


def calibrated_probability(ticker: str, df: pd.DataFrame | None,
                           horizon: int = 10, threshold: float = 0.0) -> ProbabilityResult:
    """Kalibre yukarı olasılığı + Brier/AUC. Asla istisna fırlatmaz."""
    if df is None or len(df) < 320:
        return ProbabilityResult(False, ticker, horizon,
                                 reason="Yeterli geçmiş yok (≥320 bar gerekli).")
    X, y = build_training_set(df, horizon, threshold)
    if len(X) < 250 or y.nunique() < 2:
        return ProbabilityResult(False, ticker, horizon, n_samples=len(X),
                                 reason="Yetersiz ya da tek-sınıflı veri.")
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit

        from tradingagents.ml.model import _new_classifier
    except Exception as exc:  # noqa: BLE001
        return ProbabilityResult(False, ticker, horizon, reason=f"sklearn yok: {exc}")

    medians = X.median()
    Xi = _impute(X, medians)
    tscv = TimeSeriesSplit(n_splits=5)
    oos_p: list[float] = []
    oos_y: list[float] = []
    for tr, te in tscv.split(Xi):
        if y.iloc[tr].nunique() < 2:
            continue
        m = _new_classifier()
        m.fit(Xi.iloc[tr], y.iloc[tr])
        oos_p.extend(m.predict_proba(Xi.iloc[te])[:, 1].tolist())
        oos_y.extend(y.iloc[te].tolist())
    if len(oos_p) < 30 or len(set(oos_y)) < 2:
        return ProbabilityResult(False, ticker, horizon, n_samples=len(X),
                                 reason="Örnek-dışı backtest yetersiz.")

    oos_p_arr = np.asarray(oos_p)
    oos_y_arr = np.asarray(oos_y)
    brier = float(brier_score_loss(oos_y_arr, oos_p_arr))
    try:
        auc = float(roc_auc_score(oos_y_arr, oos_p_arr))
    except Exception:  # noqa: BLE001
        auc = None

    iso = IsotonicRegression(out_of_bounds="clip").fit(oos_p_arr, oos_y_arr)
    final = _new_classifier()
    final.fit(Xi, y)
    latest = _impute(build_feature_frame(df).iloc[[-1]][FEATURE_COLUMNS], medians)
    p_raw = float(final.predict_proba(latest)[:, 1][0])
    p_cal = float(np.clip(iso.predict([p_raw])[0], 0.0, 1.0))

    return ProbabilityResult(
        ok=True, ticker=ticker, horizon=horizon,
        p_up=round(p_cal, 4), p_up_raw=round(p_raw, 4),
        brier=round(brier, 4), auc=round(auc, 4) if auc is not None else None,
        n_samples=len(X), reason="ok",
    )

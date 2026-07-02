"""Sinyal modeli — gradient boosting sınıflandırıcı + zaman-serisi backtest.

Hissenin kendi geçmişinden, ileriye dönük yön (yukarı/aşağı) olasılığını tahmin
eden bir model eğitir. Backtest, zaman-serisi bölmesiyle (geçmişte eğit →
gelecekte test, ileri kayan pencere) yapılır; karıştırmalı CV finansal seride
sızıntı yaratacağı için kullanılmaz.

Çıktı, modelin ham doğruluğunu DEĞİL, "her zaman çoğunluk sınıfı" temeline
göre kazandığı EK isabeti de raporlar (skill) — yükselen bir hissede %60
doğruluk marifet değildir, taban zaten %60'sa. Asla istisna fırlatmaz; az
veri/dejenere durumda ``ok=False`` ve gerekçe döner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.ml.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    build_training_set,
)


@dataclass
class SignalModelResult:
    ok: bool
    ticker: str
    horizon: int = 10
    threshold: float = 0.0
    labeling: str = "triple_barrier"  # "triple_barrier" (S1) | "fixed" (eski sabit-ufuk)
    n_samples: int = 0
    accuracy: float = 0.0          # backtest (out-of-sample) doğruluk
    baseline: float = 0.0          # çoğunluk sınıfı doğruluğu (taban)
    skill: float = 0.0             # accuracy - baseline (pozitif = değer katıyor)
    precision_up: float = 0.0      # "yukarı" dediğinde haklı çıkma oranı
    roc_auc: float | None = None
    prob_up: float | None = None   # son bar için güncel yukarı olasılığı
    signal: str = "VERİ YOK"       # "AL" | "TUT" | "SAT" | "VERİ YOK"
    feature_importance: dict = field(default_factory=dict)
    model: object | None = None    # eğitilmiş sklearn modeli (tekrar tahmin için)
    reason: str = ""


def _new_classifier():
    """Sızıntıya dirençli, küçük veride makul bir GBDT döndürür."""
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05,
        subsample=0.85, random_state=42,
    )


def _signal_from_prob(p: float, lo: float = 0.45, hi: float = 0.58) -> str:
    if p >= hi:
        return "AL"
    if p <= lo:
        return "SAT"
    return "TUT"


def train_signal_model(
    df: pd.DataFrame,
    ticker: str = "",
    horizon: int = 10,
    threshold: float = 0.0,
    n_splits: int = 4,
    labeling: str = "triple_barrier",
    benchmark: pd.Series | None = None,
) -> SignalModelResult:
    """Sinyal modelini eğitir, backtest eder ve güncel olasılığı üretir.

    ``df`` günlük OHLCV (uzun geçmiş iyidir; <~300 bar reddedilir). scikit-learn
    kuruluysa çalışır; değilse ``ok=False`` ve kurulum notu döner.

    Etiket varsayılanı **üçlü-bariyer** (S1): model "horizon gün sonra yukarı mı?"
    değil, "bu barda açılan ATR stop/hedefli işlem maliyet-sonrası kazanır mı?"
    sorusunu öğrenir — motorun gerçek işlem kurallarıyla hizalı. ``horizon`` bu
    modda süre bariyeridir; ``benchmark`` (XU100 kapanışı) verilirse getiri
    endekse relatif ölçülür. ``labeling="fixed"`` eski davranışı verir.
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
    except Exception as exc:  # noqa: BLE001
        return SignalModelResult(False, ticker, horizon, threshold,
                                 reason=f"scikit-learn yüklü değil: {exc}")

    if df is None or len(df) < 300:
        return SignalModelResult(False, ticker, horizon, threshold,
                                 reason="ML için en az ~300 günlük bar gerekli.")

    X, y = build_training_set(df, horizon, threshold, labeling, benchmark)
    if len(X) < 150 or y.nunique() < 2:
        return SignalModelResult(False, ticker, horizon, threshold, labeling,
                                 n_samples=len(X),
                                 reason="Eğitim için yeterli/dengeli örnek yok.")

    Xv, yv = X.to_numpy(), y.to_numpy().astype(int)

    # Zaman-serisi backtest: ileri kayan pencere, out-of-sample tahminler
    splits = min(n_splits, max(2, len(X) // 60))
    tscv = TimeSeriesSplit(n_splits=splits)
    y_true, y_pred, y_prob = [], [], []
    for tr, te in tscv.split(Xv):
        if len(np.unique(yv[tr])) < 2:
            continue
        clf = _new_classifier()
        clf.fit(Xv[tr], yv[tr])
        y_true.extend(yv[te].tolist())
        y_pred.extend(clf.predict(Xv[te]).tolist())
        y_prob.extend(clf.predict_proba(Xv[te])[:, 1].tolist())

    if not y_true:
        return SignalModelResult(False, ticker, horizon, threshold, n_samples=len(X),
                                 reason="Backtest bölmeleri tek sınıfa düştü.")

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    baseline = float(max(y_true.mean(), 1 - y_true.mean()))
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    except Exception:  # noqa: BLE001
        auc = None
    prec = float(precision_score(y_true, y_pred, zero_division=0))

    # Son model tüm veriyle yeniden eğitilir (güncel tahmin için en çok bilgi)
    final = _new_classifier()
    final.fit(Xv, yv)
    latest = build_feature_frame(df).iloc[[-1]][FEATURE_COLUMNS]
    prob_up = None
    signal = "VERİ YOK"
    if not latest.isna().any(axis=None):
        prob_up = float(final.predict_proba(latest.to_numpy())[:, 1][0])
        signal = _signal_from_prob(prob_up)

    importance = dict(sorted(
        zip(FEATURE_COLUMNS, (float(i) for i in final.feature_importances_)),
        key=lambda kv: kv[1], reverse=True,
    ))

    return SignalModelResult(
        ok=True, ticker=ticker, horizon=horizon, threshold=threshold,
        labeling=labeling,
        n_samples=len(X), accuracy=round(acc, 3), baseline=round(baseline, 3),
        skill=round(acc - baseline, 3), precision_up=round(prec, 3),
        roc_auc=round(auc, 3) if auc is not None else None,
        prob_up=round(prob_up, 3) if prob_up is not None else None,
        signal=signal, feature_importance=importance, model=final,
    )

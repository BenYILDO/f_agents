"""Kalibre olasılık — modelin '%70' dediğinde gerçekten ~%70 çıkıp çıkmadığı.

Mevcut ML özelliklerinden (``ml.features``) ufuk-h için yukarı olasılığı üretir,
ama bunu **zaman-serisi bölmesiyle (sızıntısız)** elde edilen örnek-dışı (OOS)
tahminler üzerinde **sigmoid kalibrasyon** ile kalibre eder.

Kalibrasyon yaklaşımı (küçük per-ticker örneği için sigmoid):
  - 5-kat TimeSeriesSplit ile OOS ham olasılıklar üretilir (raw brier + AUC).
  - OOS dizisi kronolojik olarak ikiye bölünür: erken %70 kalibrasyon penceresi,
    geç %30 dokunulmamış test penceresi.
  - Sigmoid (logistic) kalibratör erken pencerede fit edilir.
  - Kalibre Brier ve Brier Skill Score (BSS) yalnız test penceresinde ölçülür.
  - ``recent_skill``: test penceresinin son diliminde ölçülen BSS (rejimin
    son durumunda model hâlâ işe yarıyor mu).
  - Isotonic yöntemi küçük örneklerde overfit eder; varsayılan sigmoid'dir.

Kalite kararı **ayrı saf fonksiyon** :func:`validate_model_evidence` ile verilir
(çıplak boolean yok; ``quality_passed`` + ``rejection_reasons`` döner). Kalite
geçmezse ``p_up=None`` döner — kalitesiz model otomatik işleme katkı vermez.

Ağır (5-kat GBDT) — talep üzerine / gecelik iş içindir, saatlik cron'da değil.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tradingagents.ml.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    build_training_set,
)

# Kalibrasyon/kalite metodolojisinin sürümü — eski snapshot'larla kıyas kesintisini
# işaretlemek için (F0.2 sigmoid + BSS metodolojisi v1).
MODEL_VERSION = "calib-v1-sigmoid-bss"

# Kalite kapısı eşikleri — ileride config'den okunacak
_MIN_AUC = 0.52
_MIN_N_TEST = 20
_RECENT_WINDOW = 40  # test penceresinin son N örneğinde "recent skill" ölçülür
# BSS > 0 zaten yeterli; negatif BSS naive'den kötü demek


@dataclass
class ProbabilityResult:
    """Bir hissenin model kanıtı (ModelEvidence sözleşmesi).

    Hem kalibre olasılığı (``p_up``) hem de onu kullanıp kullanmama kararını
    veren kalite kanıtını taşır. ``quality_passed=False`` ise ``p_up`` daima
    ``None``'dır; reddin nedeni ``rejection_reasons``'ta saklanır (veri kaybolmaz).
    """

    ok: bool
    ticker: str = ""
    horizon: int = 10
    p_up: float | None = None          # kalibre yukarı olasılığı (sigmoid)
    p_up_raw: float | None = None      # kalibrasyon öncesi ham olasılık
    brier_raw: float | None = None     # ham OOS Brier (ham model)
    brier_calibrated: float | None = None  # kalibre Brier — dokunulmamış test penceresinde
    brier_skill_score: float | None = None # BSS = 1 - brier_cal / brier_naive (>0 iyi)
    recent_skill: float | None = None  # test penceresinin son diliminde BSS
    auc: float | None = None
    n_samples: int = 0
    n_train: int = 0                   # son modelin eğitildiği örnek sayısı
    n_calibration: int = 0             # kalibrasyon penceresindeki OOS örnek sayısı
    n_test: int = 0                    # dokunulmamış test penceresindeki örnek sayısı
    quality_passed: bool = False       # arena ML kapısı için — varsayılanı asla True değil
    rejection_reasons: list = field(default_factory=list)
    model_version: str = MODEL_VERSION
    prediction_asof: str = ""          # tahminin üretildiği an (ISO)
    trained_until: str = ""            # son fit'te kullanılan son bar tarihi
    reason: str = ""                   # kısa özet (geriye uyumluluk)

    # geriye uyumluluk — eski kod brier alanını kullanıyorsa
    @property
    def brier(self) -> float | None:
        return self.brier_raw


def _impute(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return (frame.replace([np.inf, -np.inf], np.nan)
            .fillna(medians).fillna(0.0))


def _brier_skill_score(brier_model: float, y: np.ndarray) -> float:
    """BSS = 1 - brier_model / brier_naive; naive = sınıf oranı varyansı."""
    base_rate = float(np.mean(y))
    brier_naive = base_rate * (1.0 - base_rate)
    if brier_naive <= 0:
        return 0.0
    return float(1.0 - brier_model / brier_naive)


def validate_model_evidence(
    *,
    auc: float | None,
    n_test: int,
    brier_skill_score: float | None,
    min_auc: float = _MIN_AUC,
    min_n_test: int = _MIN_N_TEST,
) -> tuple[bool, list[str]]:
    """Model kanıtının kaliteli olup olmadığını belirleyen **saf** fonksiyon.

    Çıplak boolean yerine ``(quality_passed, rejection_reasons)`` döner; böylece
    "model neden kullanılmadı?" (bayat mı, AUC zayıf mı, BSS negatif mi, ölçülemedi
    mi) sorusu tek yerden cevaplanır. ``quality_passed`` varsayılanı asla True olmaz:
    bir şart bile sağlanamazsa False döner.
    """
    rejection_reasons: list[str] = []
    if auc is not None and auc < min_auc:
        rejection_reasons.append(f"auc_dusuk ({auc:.3f}<{min_auc})")
    if n_test < min_n_test:
        rejection_reasons.append(f"test_penceresi_kucuk (n={n_test}<{min_n_test})")
    if brier_skill_score is None:
        rejection_reasons.append("bss_olculemedi")
    elif brier_skill_score <= 0:
        rejection_reasons.append(f"bss_negatif ({brier_skill_score:.3f})")
    return len(rejection_reasons) == 0, rejection_reasons


def calibrated_probability(ticker: str, df: pd.DataFrame | None,
                           horizon: int = 10, threshold: float = 0.0) -> ProbabilityResult:
    """Kalibre yukarı olasılığı + kalite metrikleri. Asla istisna fırlatmaz."""
    now_iso = datetime.now(timezone.utc).isoformat()
    if df is None or len(df) < 320:
        return ProbabilityResult(False, ticker, horizon,
                                 reason="Yeterli geçmiş yok (≥320 bar gerekli).",
                                 rejection_reasons=["yetersiz_veri"],
                                 prediction_asof=now_iso)
    X, y = build_training_set(df, horizon, threshold)
    if len(X) < 250 or y.nunique() < 2:
        return ProbabilityResult(False, ticker, horizon, n_samples=len(X),
                                 reason="Yetersiz ya da tek-sınıflı veri.",
                                 rejection_reasons=["tek_sinifli_veri"],
                                 prediction_asof=now_iso)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit

        from tradingagents.ml.model import _new_classifier
    except Exception as exc:  # noqa: BLE001
        return ProbabilityResult(False, ticker, horizon,
                                 reason=f"sklearn yok: {exc}",
                                 rejection_reasons=["sklearn_eksik"],
                                 prediction_asof=now_iso)

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
                                 reason="Örnek-dışı backtest yetersiz.",
                                 rejection_reasons=["oos_yetersiz"],
                                 prediction_asof=now_iso)

    oos_p_arr = np.asarray(oos_p)
    oos_y_arr = np.asarray(oos_y)

    # Ham model metrikleri (OOS üzerinden)
    brier_raw = float(brier_score_loss(oos_y_arr, oos_p_arr))
    try:
        auc = float(roc_auc_score(oos_y_arr, oos_p_arr))
    except Exception:  # noqa: BLE001
        auc = None

    # Kronolojik bölme: erken %70 kalibrasyon, geç %30 dokunulmamış test
    split_idx = max(int(len(oos_p_arr) * 0.70), 10)
    oos_p_cal = oos_p_arr[:split_idx]
    oos_y_cal = oos_y_arr[:split_idx]
    oos_p_test = oos_p_arr[split_idx:]
    oos_y_test = oos_y_arr[split_idx:]

    n_calibration = len(oos_p_cal)
    n_test = len(oos_p_test)

    # Sigmoid kalibratörü erken pencerede fit et (küçük örnekte isotonic overfit eder)
    brier_calibrated: float | None = None
    bss: float | None = None
    recent_skill: float | None = None
    iso = None

    if n_test >= _MIN_N_TEST and len(set(oos_y_cal)) >= 2 and len(set(oos_y_test)) >= 2:
        try:
            cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
            cal.fit(oos_p_cal.reshape(-1, 1), oos_y_cal)
            p_test_cal = np.clip(cal.predict_proba(oos_p_test.reshape(-1, 1))[:, 1], 0.0, 1.0)
            brier_calibrated = float(brier_score_loss(oos_y_test, p_test_cal))
            bss = _brier_skill_score(brier_calibrated, oos_y_test)
            iso = cal  # sigmoid kalibratörü

            # recent_skill: test penceresinin son diliminde BSS (rejim güncel mi?)
            r = min(_RECENT_WINDOW, n_test)
            y_recent = oos_y_test[-r:]
            if len(set(y_recent.tolist())) >= 2:
                brier_recent = float(brier_score_loss(y_recent, p_test_cal[-r:]))
                recent_skill = _brier_skill_score(brier_recent, y_recent)
        except Exception:  # noqa: BLE001
            brier_calibrated = None
            bss = None
            recent_skill = None

    # Kalite kapısı — saf doğrulayıcıya devredilir (tek yer, çıplak boolean yok)
    quality_passed, rejection_reasons = validate_model_evidence(
        auc=auc, n_test=n_test, brier_skill_score=bss,
    )

    # Son model tüm veriye fit edilir; kalibre olasılık üretilir
    final = _new_classifier()
    final.fit(Xi, y)
    latest = _impute(build_feature_frame(df).iloc[[-1]][FEATURE_COLUMNS], medians)
    p_raw = float(final.predict_proba(latest)[:, 1][0])

    # Sigmoid kalibratör varsa uygula (tüm OOS üzerinde yeniden fit)
    p_cal = p_raw
    if iso is not None:
        try:
            cal_full = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
            cal_full.fit(oos_p_arr.reshape(-1, 1), oos_y_arr)
            p_cal = float(np.clip(cal_full.predict_proba([[p_raw]])[0][1], 0.0, 1.0))
        except Exception:  # noqa: BLE001
            p_cal = p_raw

    reason = "ok" if quality_passed else f"kalite_kapisi: {', '.join(rejection_reasons)}"

    # trained_until: son fit'te kullanılan en son bar tarihi (sürüm izlenebilirliği)
    try:
        trained_until = pd.Timestamp(df.index[-1]).date().isoformat()
    except Exception:  # noqa: BLE001
        trained_until = ""

    return ProbabilityResult(
        ok=True, ticker=ticker, horizon=horizon,
        p_up=round(p_cal, 4) if quality_passed else None,
        p_up_raw=round(p_raw, 4),
        brier_raw=round(brier_raw, 4),
        brier_calibrated=round(brier_calibrated, 4) if brier_calibrated is not None else None,
        brier_skill_score=round(bss, 4) if bss is not None else None,
        recent_skill=round(recent_skill, 4) if recent_skill is not None else None,
        auc=round(auc, 4) if auc is not None else None,
        n_samples=len(X),
        n_train=len(Xi),
        n_calibration=n_calibration,
        n_test=n_test,
        quality_passed=quality_passed,
        rejection_reasons=rejection_reasons,
        model_version=MODEL_VERSION,
        prediction_asof=now_iso,
        trained_until=trained_until,
        reason=reason,
    )

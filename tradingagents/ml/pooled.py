"""Havuzlanmış (pooled) panel model — S2: tüm evren tek modelde.

Per-ticker modelin temel sorunu veri açlığıdır: hisse başına ~2.500 bar, 16
özellikli bir GBDT'ye yetmez (skill ≈ 0 gözlemi). Bu modül tüm evreni (BIST30/
BIST100) **tek panelde** birleştirir: örnek sayısı ~N_hisse × N_bar'a çıkar ve
model hisseler-arası ortak örüntüleri öğrenebilir (plan §Faz 3 / S2).

Tasarım:
  - Etiket: S1 üçlü-bariyer (``ml.features.triple_barrier_outcomes``) — işlem-hizalı,
    maliyet-sonrası, XU100-relatif.
  - Özellikler: mevcut ölçek-bağımsız ``FEATURE_COLUMNS`` + **kesitsel** kolonlar
    (aynı gün evren içi z-skorlar + endekse-relatif momentum). Kesitsel z yalnız
    aynı-gün bilgisi kullanır → nedensel, sızıntı yok.
  - Doğrulama: **purged walk-forward** — kronolojik ileri-kayan bölmeler; eğitim
    sonu ile test başı arasına ``horizon+1`` günlük embargo konur ki eğitim
    etiketlerinin ileri-bakan pencereleri test dönemine taşmasın (López de Prado,
    Advances in Financial ML, §7 purged K-fold).
  - Kalibrasyon: OOS tahminlerin erken %70'inde Platt (sigmoid), geç %30'unda
    dokunulmamış test — ``analytics.probability`` (F0.2) ile aynı disiplin;
    kalite kararı aynı ``validate_model_evidence`` kapısından geçer.
  - Kalıcılık: eğitilmiş model + kalibratör + medyanlar tek artefakt olarak
    pickle+zlib+base64 ile Supabase'e yazılır (``storage.pooled_models``); gün
    içinde yalnız tahmin yapılır, yeniden eğitim gecelik iştedir.

Ağır iş (tek model ama büyük panel) — gecelik cron içindir, saatlik değil.
"""

from __future__ import annotations

import base64
import pickle
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tradingagents.ml.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    make_labels_triple_barrier,
)

# Metodoloji sürümü — etiket/özellik/CV disiplini değişince artırılır (kıyas kesintisi).
POOLED_MODEL_VERSION = "pooled-v1-tb-purged"

# Kesitsel (aynı-gün, evren-içi) ek kolonlar — panel kurulurken hesaplanır.
CS_COLUMNS = ["cs_ret20_z", "cs_vol_z", "cs_rsi_z", "cs_dist_high_z", "rel_ret_20"]
PANEL_FEATURES = FEATURE_COLUMNS + CS_COLUMNS

_CAL_FRAC = 0.70          # OOS'un erken %70'i kalibrasyon, geç %30'u test
_MIN_PANEL_ROWS = 2_000   # havuz bundan küçükse pooled modelin anlamı yok


@dataclass
class PooledModelResult:
    """Havuz modelinin eğitim çıktısı + kalite kanıtı (ModelEvidence disiplini)."""

    ok: bool
    model_version: str = POOLED_MODEL_VERSION
    horizon: int = 10
    threshold: float = 0.0
    universe: list = field(default_factory=list)
    n_tickers: int = 0
    n_samples: int = 0
    n_test: int = 0
    auc: float | None = None
    brier_raw: float | None = None
    brier_calibrated: float | None = None
    brier_skill_score: float | None = None
    quality_passed: bool = False
    rejection_reasons: list = field(default_factory=list)
    feature_importance: dict = field(default_factory=dict)
    trained_at: str = ""
    trained_until: str = ""           # paneldeki son bar tarihi
    model: object | None = None       # eğitilmiş GBDT
    calibrator: object | None = None  # Platt (sigmoid) kalibratörü
    medians: object | None = None     # imputasyon medyanları (pd.Series)
    reason: str = ""


def _zscore_row(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def _add_cross_sectional(panel: pd.DataFrame,
                         benchmark: pd.Series | None) -> pd.DataFrame:
    """Aynı-gün evren-içi z-skorlar + endekse-relatif momentum ekler (nedensel)."""
    g = panel.groupby(level="date")
    panel["cs_ret20_z"] = g["ret_20"].transform(_zscore_row)
    panel["cs_vol_z"] = g["atr_pct"].transform(_zscore_row)
    panel["cs_rsi_z"] = g["rsi14"].transform(_zscore_row)
    panel["cs_dist_high_z"] = g["dist_from_high252"].transform(_zscore_row)

    rel = panel["ret_20"].copy()
    if benchmark is not None and len(benchmark) > 0:
        b20 = benchmark.pct_change(20)
        dates = panel.index.get_level_values("date")
        rel = panel["ret_20"].to_numpy() - b20.reindex(dates).to_numpy()
    panel["rel_ret_20"] = rel
    return panel


def build_panel(
    data: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    horizon: int = 10,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Ticker→OHLCV sözlüğünü tek eğitim paneline döker.

    Döner: ``X`` (MultiIndex [date, ticker], kolonlar ``PANEL_FEATURES``) ve
    hizalı ``y`` (üçlü-bariyer etiketi). NaN özellik/etiket satırları düşülmüş,
    tarih sırasına göre sıralı (walk-forward bölme buna güvenir).
    """
    frames: list[pd.DataFrame] = []
    for tk, df in data.items():
        if df is None or len(df) < 260:
            continue
        try:
            X = build_feature_frame(df)
            y = make_labels_triple_barrier(df, horizon, threshold, benchmark)
        except Exception:  # noqa: BLE001 — tek hisse paneli kırmasın
            continue
        f = X.copy()
        f["_label"] = y
        f["ticker"] = tk
        f = f.rename_axis("date").reset_index()
        frames.append(f)

    if not frames:
        return (pd.DataFrame(columns=PANEL_FEATURES), pd.Series(dtype=float))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.set_index(["date", "ticker"]).sort_index()
    panel = _add_cross_sectional(panel, benchmark)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
    return panel[PANEL_FEATURES], panel["_label"]


def purged_walk_forward(dates: np.ndarray, n_splits: int = 5,
                        embargo: int = 11):
    """Panel satır tarihlerinde ileri-kayan, embargolu (purged) bölmeler üretir.

    ``dates``: satır başına tarih (sıralı panel). Benzersiz tarihler ``n_splits+1``
    eşit dilime bölünür; k. bölmede eğitim = ilk k dilim, test = eğitim sonundan
    ``embargo`` tarih sonrası başlayan bir dilim. Embargo, eğitim etiketlerinin
    ileri-bakan penceresinin (horizon+1 gün) test dönemiyle örtüşmesini keser.
    Verim: (train_idx, test_idx) numpy dizileri.
    """
    uniq = np.unique(dates)
    n = len(uniq)
    fold = n // (n_splits + 1)
    if fold < 2:
        return
    for k in range(1, n_splits + 1):
        train_end = fold * k                      # uniq[:train_end] → eğitim
        test_start = train_end + embargo
        test_end = min(test_start + fold, n)
        if test_start >= n or test_end <= test_start:
            continue
        tr = np.where(dates < uniq[train_end])[0]
        te = np.where((dates >= uniq[test_start]) & (dates <= uniq[test_end - 1]))[0]
        if len(tr) and len(te):
            yield tr, te


def train_pooled_model(
    data: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    horizon: int = 10,
    threshold: float = 0.0,
    n_splits: int = 5,
) -> PooledModelResult:
    """Havuz modelini eğitir, purged walk-forward ile ölçer, kalibre eder.

    Asla istisna fırlatmaz; kalite kapısını geçemezse ``quality_passed=False``
    döner (tahminler yine üretilebilir ama otomatik işlem etkileyemez).
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
    except Exception as exc:  # noqa: BLE001
        return PooledModelResult(False, horizon=horizon, threshold=threshold,
                                 reason=f"sklearn yok: {exc}",
                                 rejection_reasons=["sklearn_eksik"])

    from tradingagents.analytics.probability import (
        _brier_skill_score,
        validate_model_evidence,
    )
    from tradingagents.ml.model import _new_classifier

    X, y = build_panel(data, benchmark, horizon, threshold)
    universe = sorted({tk for _, tk in X.index}) if len(X) else []
    if len(X) < _MIN_PANEL_ROWS or y.nunique() < 2:
        return PooledModelResult(False, horizon=horizon, threshold=threshold,
                                 universe=universe, n_tickers=len(universe),
                                 n_samples=len(X),
                                 reason=f"Panel küçük ({len(X)}<{_MIN_PANEL_ROWS}).",
                                 rejection_reasons=["panel_kucuk"])

    medians = X.median()
    Xi = X.fillna(medians).to_numpy(float)
    yv = y.to_numpy(float)
    dates = X.index.get_level_values("date").to_numpy()

    # ── Purged walk-forward OOS tahminler ────────────────────────────────
    oos_p: list[float] = []
    oos_y: list[float] = []
    oos_d: list = []
    for tr, te in purged_walk_forward(dates, n_splits, embargo=horizon + 1):
        if len(np.unique(yv[tr])) < 2:
            continue
        clf = _new_classifier()
        clf.fit(Xi[tr], yv[tr])
        oos_p.extend(clf.predict_proba(Xi[te])[:, 1].tolist())
        oos_y.extend(yv[te].tolist())
        oos_d.extend(dates[te].tolist())

    if len(oos_p) < 200 or len(set(oos_y)) < 2:
        return PooledModelResult(False, horizon=horizon, threshold=threshold,
                                 universe=universe, n_tickers=len(universe),
                                 n_samples=len(X),
                                 reason="Purged OOS backtest yetersiz.",
                                 rejection_reasons=["oos_yetersiz"])

    order = np.argsort(np.asarray(oos_d, dtype="datetime64[ns]"), kind="stable")
    p_arr = np.asarray(oos_p)[order]
    y_arr = np.asarray(oos_y)[order]

    brier_raw = float(brier_score_loss(y_arr, p_arr))
    try:
        auc = float(roc_auc_score(y_arr, p_arr))
    except Exception:  # noqa: BLE001
        auc = None

    # ── Platt kalibrasyon: erken %70 fit, geç %30 dokunulmamış test ───────
    split = max(int(len(p_arr) * _CAL_FRAC), 10)
    brier_cal: float | None = None
    bss: float | None = None
    calibrator = None
    n_test = len(p_arr) - split
    if n_test >= 50 and len(set(y_arr[:split])) >= 2 and len(set(y_arr[split:])) >= 2:
        try:
            cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
            cal.fit(p_arr[:split].reshape(-1, 1), y_arr[:split])
            p_test = np.clip(cal.predict_proba(p_arr[split:].reshape(-1, 1))[:, 1], 0, 1)
            brier_cal = float(brier_score_loss(y_arr[split:], p_test))
            bss = _brier_skill_score(brier_cal, y_arr[split:])
            # Nihai kalibratör tüm OOS üzerinde yeniden fit edilir
            calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
            calibrator.fit(p_arr.reshape(-1, 1), y_arr)
        except Exception:  # noqa: BLE001
            calibrator = None

    quality_passed, rejection_reasons = validate_model_evidence(
        auc=auc, n_test=n_test, brier_skill_score=bss,
    )

    # ── Nihai model tüm panelde eğitilir ──────────────────────────────────
    final = _new_classifier()
    final.fit(Xi, yv)
    importance = dict(sorted(
        zip(PANEL_FEATURES, (float(i) for i in final.feature_importances_)),
        key=lambda kv: kv[1], reverse=True,
    ))

    return PooledModelResult(
        ok=True, horizon=horizon, threshold=threshold,
        universe=universe, n_tickers=len(universe), n_samples=len(X),
        n_test=n_test,
        auc=round(auc, 4) if auc is not None else None,
        brier_raw=round(brier_raw, 4),
        brier_calibrated=round(brier_cal, 4) if brier_cal is not None else None,
        brier_skill_score=round(bss, 4) if bss is not None else None,
        quality_passed=quality_passed, rejection_reasons=rejection_reasons,
        feature_importance=importance,
        trained_at=now_iso,
        trained_until=str(pd.Timestamp(dates.max()).date()),
        model=final, calibrator=calibrator, medians=medians,
        reason="ok" if quality_passed else
               f"kalite_kapisi: {', '.join(rejection_reasons)}",
    )


def predict_pooled(
    bundle: dict,
    data: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
) -> dict[str, float]:
    """Kayıtlı havuz modeliyle her hissenin güncel kazanma olasılığı.

    ``bundle``: :func:`deserialize_bundle` çıktısı (ya da eğitim sonucunun
    ``model/calibrator/medians`` alanları). Kesitsel z-skorlar, verilen evrenin
    SON barları üzerinden hesaplanır — tahmin de eğitimle aynı bilgiyi görür.
    Hata durumunda o hisse atlanır; asla istisna fırlatmaz.
    """
    model = bundle.get("model")
    calibrator = bundle.get("calibrator")
    medians = bundle.get("medians")
    if model is None or medians is None:
        return {}

    rows: dict[str, pd.Series] = {}
    for tk, df in data.items():
        if df is None or len(df) < 260:
            continue
        try:
            last = build_feature_frame(df).iloc[-1]
        except Exception:  # noqa: BLE001
            continue
        rows[tk] = last
    if not rows:
        return {}

    F = pd.DataFrame(rows).T          # index=ticker, kolonlar=FEATURE_COLUMNS
    # Kesitsel z — son barlar arasında (eğitimdeki aynı-gün z'nin canlı karşılığı)
    F["cs_ret20_z"] = _zscore_row(F["ret_20"])
    F["cs_vol_z"] = _zscore_row(F["atr_pct"])
    F["cs_rsi_z"] = _zscore_row(F["rsi14"])
    F["cs_dist_high_z"] = _zscore_row(F["dist_from_high252"])
    b20 = None
    if benchmark is not None and len(benchmark) > 20:
        b20 = float(benchmark.iloc[-1] / benchmark.iloc[-21] - 1.0)
    F["rel_ret_20"] = F["ret_20"] - b20 if b20 is not None else F["ret_20"]

    F = (F[PANEL_FEATURES].replace([np.inf, -np.inf], np.nan)
         .fillna(medians).fillna(0.0))
    try:
        p_raw = model.predict_proba(F.to_numpy(float))[:, 1]
        if calibrator is not None:
            p_raw = np.clip(
                calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1], 0.0, 1.0)
    except Exception:  # noqa: BLE001
        return {}
    return {tk: round(float(p), 4) for tk, p in zip(F.index, p_raw)}


# ── Artefakt serileştirme (Supabase text kolonu için) ────────────────────────

def serialize_bundle(res: PooledModelResult) -> str:
    """Model + kalibratör + medyanları tek base64 dizesine paketler."""
    payload = {
        "model": res.model, "calibrator": res.calibrator, "medians": res.medians,
        "feature_columns": PANEL_FEATURES, "horizon": res.horizon,
        "threshold": res.threshold, "model_version": res.model_version,
    }
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode("ascii")


def deserialize_bundle(artifact: str) -> dict:
    """:func:`serialize_bundle` çıktısını geri açar (hata → boş sözlük).

    Not: pickle yalnız KENDİ yazdığımız Supabase satırından okunur (service-role
    ile yazılan, tek kullanıcılı tablo) — güvenilmeyen kaynaktan pickle açılmaz.
    """
    try:
        return pickle.loads(zlib.decompress(base64.b64decode(artifact)))
    except Exception:  # noqa: BLE001
        return {}

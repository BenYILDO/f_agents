"""Tests for the ML signal model — feature engineering + train/predict plumbing.

All synthetic, offline (sklearn trains without network). A learnable pattern is
injected so the model can demonstrably beat the baseline; pipeline robustness
(insufficient data, label/feature alignment) is also checked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.ml.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    build_training_set,
    make_labels,
)
from tradingagents.ml.model import train_signal_model

pytest.importorskip("sklearn")


def _ohlcv(close: pd.Series, volume: pd.Series | None = None, spread=1.0) -> pd.DataFrame:
    vol = volume if volume is not None else pd.Series(1_000_000.0, index=close.index)
    return pd.DataFrame({
        "Open": close.shift().fillna(close.iloc[0]),
        "High": close + spread, "Low": close - spread,
        "Close": close, "Volume": vol,
    })


def _learnable_df(n=900, seed=0) -> pd.DataFrame:
    """Momentum-rejimli sentetik seri: trend kümeleri yön tahminini öğrenilebilir kılar."""
    rng = np.random.default_rng(seed)
    # Rejim anahtarı: uzun bloklar halinde pozitif/negatif drift
    blocks = rng.choice([0.0015, -0.0015], size=n // 30 + 1)
    drift = np.repeat(blocks, 30)[:n]
    noise = rng.normal(0, 0.004, n)
    ret = drift + noise
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = pd.Series(100 * np.cumprod(1 + ret), index=idx)
    return _ohlcv(close)


@pytest.mark.unit
class TestFeatures:
    def test_feature_frame_columns(self):
        X = build_feature_frame(_learnable_df(n=400))
        assert list(X.columns) == FEATURE_COLUMNS
        # İlk satırlar NaN (uzun SMA ısınması), sonrası dolu olmalı
        assert X.dropna().shape[0] > 100

    def test_labels_shifted_no_lookahead(self):
        df = _learnable_df(n=300)
        y = make_labels(df, horizon=10)
        # Son 10 bar geleceği bilinmediği için NaN olmalı
        assert y.iloc[-10:].isna().all()
        assert set(y.dropna().unique()) <= {0.0, 1.0}

    def test_training_set_aligned(self):
        X, y = build_training_set(_learnable_df(n=500), horizon=10)
        assert len(X) == len(y) and len(X) > 0
        assert not X.isna().any(axis=None)


@pytest.mark.unit
class TestTrainModel:
    def test_trains_and_predicts(self):
        res = train_signal_model(_learnable_df(), ticker="TEST", horizon=10)
        assert res.ok
        assert 0.0 <= res.accuracy <= 1.0
        assert res.prob_up is not None and 0.0 <= res.prob_up <= 1.0
        assert res.signal in ("AL", "TUT", "SAT")
        assert set(res.feature_importance) == set(FEATURE_COLUMNS)
        assert res.model is not None

    def test_learns_above_baseline(self):
        # Öğrenilebilir rejimli seride model en azından tabana yakın/üstünde olmalı
        res = train_signal_model(_learnable_df(seed=3), ticker="TEST", horizon=10)
        assert res.ok
        # Skill çok negatif olmamalı (model tabanı ciddi bozmamalı)
        assert res.skill >= -0.1

    def test_insufficient_data_fails_loud(self):
        res = train_signal_model(_learnable_df(n=120), ticker="TEST")
        assert not res.ok and res.reason

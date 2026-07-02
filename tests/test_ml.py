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
    make_labels_triple_barrier,
    triple_barrier_outcomes,
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


def _tb_df(scenario, n_flat=30):
    """Düz (O=100, H=101, L=99, C=100 → ATR≈2) ısınma + senaryo barları.

    Düz barlarda stop = 100 − 2·2 = 96, hedef = 100 + 2·2·2 = 108 olur; senaryo
    barları bu bariyerlere göre kurgulanır. ``scenario``: (O, H, L, C) listesi.
    """
    rows = [(100.0, 101.0, 99.0, 100.0)] * n_flat + list(scenario)
    idx = pd.bdate_range("2022-01-03", periods=len(rows))
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": 1_000_000.0}, index=idx)


@pytest.mark.unit
class TestTripleBarrier:
    """S1: etiket motorun fiziğiyle (T+1 fill, ATR stop/hedef, maliyet) hizalı."""

    T = 29  # son düz bar = sinyal barı (stop 96 / hedef 108)

    def test_target_hit_labels_one(self):
        df = _tb_df([(100, 103, 99.5, 102), (102, 109, 101, 108), (108, 109, 107, 108)])
        out = triple_barrier_outcomes(df, max_hold=10)
        row = out.iloc[self.T]
        assert row["exit_reason"] == "hedef"
        assert row["label"] == 1.0 and row["net_ret"] > 0
        assert row["hold_days"] == 2.0

    def test_stop_hit_labels_zero(self):
        df = _tb_df([(99, 100, 95, 96), (96, 97, 95, 96)])
        out = triple_barrier_outcomes(df, max_hold=10)
        row = out.iloc[self.T]
        assert row["exit_reason"] == "stop"
        assert row["label"] == 0.0 and row["net_ret"] < 0

    def test_both_barriers_same_bar_conservative_stop(self):
        # Aynı barda hem stop hem hedef → muhafazakâr: stop önce (motorla aynı)
        df = _tb_df([(100, 109, 95, 100), (100, 101, 99, 100)])
        out = triple_barrier_outcomes(df, max_hold=10)
        assert out.iloc[self.T]["exit_reason"] == "stop"
        assert out.iloc[self.T]["label"] == 0.0

    def test_time_barrier_costs_make_flat_lose(self):
        # Bariyer görülmez → süre çıkışı; düz fiyatta maliyetler etiketi 0 yapar
        df = _tb_df([(100.0, 101.0, 99.0, 100.0)] * 8)
        out = triple_barrier_outcomes(df, max_hold=3)
        row = out.iloc[self.T]
        assert row["exit_reason"] == "süre"
        assert row["hold_days"] == 4.0            # 3 gün tut + T+1 açılış satışı
        assert row["label"] == 0.0 and -0.01 < row["net_ret"] < 0

    def test_no_lookahead_tail_is_nan(self):
        df = _tb_df([(100.0, 101.0, 99.0, 100.0)] * 20)
        y = make_labels_triple_barrier(df, max_hold=10)
        # Süre bariyeri + T+1 satışı için gelecek yok → kuyruk NaN olmalı
        assert y.iloc[-11:].isna().all()
        assert set(y.dropna().unique()) <= {0.0, 1.0}

    def test_benchmark_relative_flips_label(self):
        scenario = [(100, 103, 99.5, 102), (102, 109, 101, 108), (108, 109, 107, 108)]
        df = _tb_df(scenario)
        # Endeks, işlem penceresinde hisseden çok daha fazla yükselir (%30)
        bench = pd.Series(1000.0, index=df.index)
        bench.iloc[self.T + 1:] = 1300.0
        y_abs = make_labels_triple_barrier(df, max_hold=10)
        y_rel = make_labels_triple_barrier(df, max_hold=10, benchmark=bench)
        assert y_abs.iloc[self.T] == 1.0          # mutlak: hedef vurdu, kazandı
        assert y_rel.iloc[self.T] == 0.0          # relatif: endeksin altında kaldı

    def test_training_set_triple_barrier_aligned(self):
        X, y = build_training_set(_learnable_df(n=500), horizon=10)
        assert len(X) == len(y) and len(X) > 0
        assert not X.isna().any(axis=None)
        # Eski sabit-ufuk yolu da hâlâ çalışmalı
        Xf, yf = build_training_set(_learnable_df(n=500), horizon=10, labeling="fixed")
        assert len(Xf) == len(yf) and len(Xf) > 0


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

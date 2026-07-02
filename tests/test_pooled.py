"""Tests for the pooled panel model (S2) — panel build, purged CV, persistence.

All synthetic and offline. Verifies: panel stacking + cross-sectional features,
purged walk-forward embargo (no label-window leakage), end-to-end training with
quality evidence, per-ticker prediction, and artifact round-trip (the Supabase
persistence contract) — without touching the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.ml.pooled import (
    PANEL_FEATURES,
    build_panel,
    deserialize_bundle,
    predict_pooled,
    purged_walk_forward,
    serialize_bundle,
    train_pooled_model,
)

pytest.importorskip("sklearn")


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": close.shift().fillna(close.iloc[0]),
        "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1_000_000.0,
    })


def _universe(n_tickers=6, n=700) -> dict[str, pd.DataFrame]:
    """Rejim-bloklu sentetik evren; hisseler ortak bir piyasa faktörü paylaşır."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-02", periods=n)
    market = np.repeat(rng.choice([0.0012, -0.0012], size=n // 25 + 1), 25)[:n]
    out = {}
    for k in range(n_tickers):
        drift = market + rng.normal(0, 0.0004, n)
        noise = rng.normal(0, 0.006, n)
        close = pd.Series(100 * np.cumprod(1 + drift + noise), index=idx)
        out[f"T{k}.IS"] = _ohlcv(close)
    return out


def _benchmark(data: dict) -> pd.Series:
    closes = pd.concat([df["Close"] for df in data.values()], axis=1)
    return closes.mean(axis=1)


@pytest.mark.unit
class TestPanel:
    def test_build_panel_stacks_and_features(self):
        data = _universe()
        X, y = build_panel(data, _benchmark(data))
        assert list(X.columns) == PANEL_FEATURES
        assert len(X) == len(y) and len(X) > 2000
        assert not X.isna().any(axis=None)
        # Panel, per-ticker'dan büyük olmalı (havuzlamanın amacı)
        assert X.index.get_level_values("ticker").nunique() == len(data)
        # Tarih sıralı (walk-forward buna güvenir)
        dates = X.index.get_level_values("date")
        assert (dates.sort_values() == dates).all()

    def test_cross_sectional_z_is_per_date(self):
        data = _universe()
        X, _ = build_panel(data, _benchmark(data))
        # Tam kesitli bir günde z-skorların ortalaması ~0 olmalı
        by_date = X.groupby(level="date")["cs_ret20_z"]
        full_days = by_date.count() == len(data)
        means = by_date.mean()[full_days]
        assert len(means) > 100
        assert np.allclose(means, 0.0, atol=1e-9)


@pytest.mark.unit
class TestPurgedWalkForward:
    def test_embargo_gap_between_train_and_test(self):
        days = pd.bdate_range("2021-01-01", periods=300)
        # Panel: her tarihte 3 satır (3 hisse)
        dates = np.repeat(days.to_numpy(), 3)
        embargo = 11
        splits = list(purged_walk_forward(dates, n_splits=4, embargo=embargo))
        assert len(splits) >= 3
        for tr, te in splits:
            tr_max = dates[tr].max()
            te_min = dates[te].min()
            # Eğitim sonu ile test başı arasında en az `embargo` benzersiz gün
            gap_days = np.unique(dates[(dates > tr_max) & (dates < te_min)])
            assert len(gap_days) >= embargo - 1
            # Kronoloji: tüm eğitim, tüm testten önce
            assert tr_max < te_min

    def test_too_few_dates_yields_nothing(self):
        dates = np.repeat(pd.bdate_range("2021-01-01", periods=8).to_numpy(), 2)
        assert list(purged_walk_forward(dates, n_splits=5, embargo=11)) == []


@pytest.mark.unit
class TestTrainPooled:
    def test_trains_with_evidence_and_predicts(self):
        data = _universe()
        bench = _benchmark(data)
        res = train_pooled_model(data, bench)
        assert res.ok
        assert res.n_tickers == len(data) and res.n_samples > 2000
        assert res.auc is not None and 0.0 <= res.auc <= 1.0
        assert res.brier_raw is not None
        assert isinstance(res.rejection_reasons, list)
        assert res.trained_until  # sürümleme izi
        assert set(res.feature_importance) == set(PANEL_FEATURES)

        preds = predict_pooled(
            {"model": res.model, "calibrator": res.calibrator,
             "medians": res.medians}, data, bench)
        assert set(preds) == set(data)
        assert all(0.0 <= p <= 1.0 for p in preds.values())

    def test_small_panel_fails_loud(self):
        data = dict(list(_universe(n_tickers=1, n=300).items()))
        res = train_pooled_model(data, None)
        assert not res.ok and res.rejection_reasons


@pytest.mark.unit
class TestArtifactRoundTrip:
    def test_serialize_deserialize_same_predictions(self):
        data = _universe(n_tickers=5, n=700)
        bench = _benchmark(data)
        res = train_pooled_model(data, bench)
        assert res.ok
        artifact = serialize_bundle(res)
        assert isinstance(artifact, str) and len(artifact) > 100

        bundle = deserialize_bundle(artifact)
        assert bundle.get("model_version") == res.model_version
        p1 = predict_pooled({"model": res.model, "calibrator": res.calibrator,
                             "medians": res.medians}, data, bench)
        p2 = predict_pooled(bundle, data, bench)
        assert p1 == p2  # depodan dönen model birebir aynı tahmini vermeli

    def test_corrupt_artifact_returns_empty(self):
        assert deserialize_bundle("bozuk-base64!!") == {}
        assert predict_pooled({}, {}, None) == {}

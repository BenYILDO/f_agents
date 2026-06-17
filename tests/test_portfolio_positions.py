"""Portföy pozisyon/P&L hesapları — saf, ağsız birim testleri."""

from __future__ import annotations

import pytest

from tradingagents.storage.portfolio import Position, compute_positions


@pytest.mark.unit
class TestComputePositions:
    def test_weighted_average_cost_over_multiple_lots(self):
        rows = [
            {"ticker": "THYAO.IS", "quantity": 10, "buy_price": 100},
            {"ticker": "THYAO.IS", "quantity": 10, "buy_price": 120},
        ]
        (pos,) = compute_positions(rows)
        assert pos.ticker == "THYAO.IS"
        assert pos.quantity == 20
        assert pos.avg_cost == 110.0       # (10*100 + 10*120) / 20
        assert pos.invested == 2200.0
        assert pos.lots == 2

    def test_pnl_and_weight_with_prices(self):
        rows = [
            {"ticker": "THYAO.IS", "quantity": 20, "buy_price": 110},
            {"ticker": "GARAN.IS", "quantity": 5, "buy_price": 40},
        ]
        prices = {"THYAO.IS": 130.0, "GARAN.IS": 50.0}
        positions = compute_positions(rows, prices)
        by = {p.ticker: p for p in positions}

        thy = by["THYAO.IS"]
        assert thy.market_value == 2600.0
        assert thy.pnl == 400.0
        assert thy.pnl_pct == pytest.approx(18.18, abs=0.01)

        gar = by["GARAN.IS"]
        assert gar.pnl == 50.0
        assert gar.pnl_pct == pytest.approx(25.0, abs=0.01)

        # Ağırlıklar piyasa değeri üzerinden, toplam %100
        total_w = sum(p.weight_pct for p in positions)
        assert total_w == pytest.approx(100.0, abs=0.05)
        assert thy.weight_pct == pytest.approx(2600 / 2850 * 100, abs=0.05)

    def test_missing_price_leaves_pnl_unset(self):
        rows = [{"ticker": "ASELS.IS", "quantity": 3, "buy_price": 50}]
        (pos,) = compute_positions(rows, prices={})
        assert pos.last_price is None
        assert pos.market_value is None
        assert pos.pnl is None
        assert pos.pnl_pct is None
        assert pos.weight_pct is None

    def test_empty(self):
        assert compute_positions([]) == []

    def test_ticker_case_insensitive_aggregation(self):
        rows = [
            {"ticker": "thyao.is", "quantity": 1, "buy_price": 100},
            {"ticker": "THYAO.IS", "quantity": 1, "buy_price": 200},
        ]
        positions = compute_positions(rows)
        assert len(positions) == 1
        assert positions[0].avg_cost == 150.0

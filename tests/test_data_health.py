"""Tests for the macro data-health contract."""

from __future__ import annotations

import pytest

from tradingagents.dataflows.data_health import (
    SourceHealth, OK, EMPTY, ERROR, any_ok, all_down, render_health,
)


@pytest.mark.unit
class TestSourceHealth:
    def test_icons_and_line(self):
        h = SourceHealth("BloombergHT", OK, count=8)
        assert h.is_ok and h.icon() == "✓" and "BloombergHT" in h.line() and "[8]" in h.line()
        assert SourceHealth("X", EMPTY).icon() == "⚠️"
        assert SourceHealth("Y", ERROR, "403").icon() == "✗" and "403" in SourceHealth("Y", ERROR, "403").line()

    def test_any_ok_and_all_down(self):
        live = [SourceHealth("a", OK, count=1), SourceHealth("b", ERROR)]
        dead = [SourceHealth("a", EMPTY), SourceHealth("b", ERROR)]
        assert any_ok(live) and not all_down(live)
        assert not any_ok(dead) and all_down(dead)
        assert not all_down([])  # no sources != all down

    def test_render_health_multiline(self):
        out = render_health([SourceHealth("a", OK, count=2), SourceHealth("b", EMPTY, "boş")])
        assert out.count("\n") == 1 and "✓ a" in out and "⚠️ b" in out

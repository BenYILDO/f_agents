"""Data-health contract for the Macro analyst's sources.

The trading desk must never silently run on missing data. Every macro source
(Turkish news RSS, TCMB EVDS official series, yfinance price history for the
event study) reports a :class:`SourceHealth` describing whether it returned
real data, came back empty, or errored. The Macro analyst aggregates these,
surfaces them to the UI, and — when everything is down — prepends a loud data
warning to its report instead of passing placeholder text off as analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

# status values, in increasing severity
OK = "ok"
EMPTY = "empty"
ERROR = "error"

_ICON = {OK: "✓", EMPTY: "⚠️", ERROR: "✗"}


@dataclass(frozen=True)
class SourceHealth:
    """Outcome of a single data fetch.

    status: "ok" (real data), "empty" (reachable but no data / fell back to a
    seed), or "error" (fetch failed). ``detail`` is a short human note and
    ``count`` the item/observation count when meaningful.
    """
    name: str
    status: str
    detail: str = ""
    count: int = 0

    @property
    def is_ok(self) -> bool:
        return self.status == OK

    def icon(self) -> str:
        return _ICON.get(self.status, "?")

    def line(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        cnt = f" [{self.count}]" if self.count else ""
        return f"{self.icon()} {self.name}{cnt}{suffix}"


def any_ok(items: list[SourceHealth]) -> bool:
    """True if at least one source returned real data."""
    return any(h.is_ok for h in items)


def all_down(items: list[SourceHealth]) -> bool:
    """True if there are sources and none of them returned real data."""
    return bool(items) and not any_ok(items)


def render_health(items: list[SourceHealth]) -> str:
    """Render a compact multi-line health summary (for prompts and logs)."""
    return "\n".join(h.line() for h in items)

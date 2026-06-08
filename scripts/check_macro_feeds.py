"""Live probe for the Macro analyst's data sources — run where egress is open.

The Macro analyst pulls Turkish macro headlines (RSS) and, optionally, TCMB
policy-rate decisions (EVDS). Both degrade gracefully to placeholders, which
means a misconfigured feed or a blocked network looks the same as "quiet
markets". This script verifies, loudly, that the sources actually return data —
so you can confirm coverage instead of guessing.

Usage:
    python scripts/check_macro_feeds.py            # RSS feeds only
    EVDS_API_KEY=... python scripts/check_macro_feeds.py   # + EVDS check

Note: many news hosts (and Yahoo Finance) are blocked inside the sandboxed
Claude Code execution environment by its network policy, so run this from your
own machine / deployment where outbound HTTP is allowed.
"""

from __future__ import annotations

import os
import sys

from tradingagents.dataflows.turkish_macro import fetch_turkish_macro_news
from tradingagents.dataflows.tcmb_calendar import (
    get_macro_official_numbers,
    resolve_ppk,
)


def check_rss() -> bool:
    print("== RSS macro feeds (with per-source health) ==")
    res = fetch_turkish_macro_news()
    for h in res.sources:
        print(f"  {h.line()}")
    print("\n-- Themed macro block (what the agent receives) --")
    print(res.text)
    return res.ok


def check_evds() -> None:
    print("\n== EVDS official numbers (TCMB) ==")
    text, healths = get_macro_official_numbers()
    for h in healths:
        print(f"  {h.line()}")
    print("  " + text.replace("\n", "\n  "))

    print("\n== PPK decisions (EVDS-authoritative, seed-fallback) ==")
    cuts, health = resolve_ppk(direction="cut")
    hikes, _ = resolve_ppk(direction="hike")
    print(f"  {health.line()}")
    print(f"  Resolved → {len(cuts)} cut(s), {len(hikes)} hike(s). "
          f"Latest cut: {cuts[-1]['date'] if cuts else '—'}")


if __name__ == "__main__":
    live = check_rss()
    check_evds()
    sys.exit(0 if live else 1)

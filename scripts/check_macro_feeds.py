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

from tradingagents.dataflows.turkish_macro import DEFAULT_MACRO_FEEDS, _fetch_feed
from tradingagents.dataflows.turkish_macro import fetch_turkish_macro_news
from tradingagents.dataflows.tcmb_calendar import (
    _fetch_evds_observations,
    _DEFAULT_POLICY_RATE_SERIES,
    get_ppk_decisions,
)


def check_rss() -> bool:
    print("== RSS macro feeds ==")
    any_live = False
    for label, url in DEFAULT_MACRO_FEEDS:
        items = _fetch_feed(label, url, limit=10, timeout=10.0)
        status = f"OK  {len(items):2d} items" if items else "EMPTY/UNREACHABLE"
        sample = f" | e.g. {items[0]['title'][:70]}" if items else ""
        print(f"  [{status:18s}] {label}{sample}")
        any_live = any_live or bool(items)
    print("\n== Themed macro block (what the agent receives) ==")
    print(fetch_turkish_macro_news())
    return any_live


def check_evds() -> None:
    key = os.environ.get("EVDS_API_KEY")
    print("\n== EVDS policy-rate (TCMB) ==")
    if not key:
        print("  EVDS_API_KEY not set — skipping live EVDS check (seed table will be used).")
    else:
        obs = _fetch_evds_observations(key, _DEFAULT_POLICY_RATE_SERIES,
                                       "2023-01-01", "2025-12-31")
        if obs:
            print(f"  OK  {len(obs)} observations for {_DEFAULT_POLICY_RATE_SERIES}; "
                  f"latest {obs[-1][0]} = {obs[-1][1]}%")
        else:
            print(f"  EMPTY — check the series code ({_DEFAULT_POLICY_RATE_SERIES}) "
                  "or key; falling back to seed table.")
    cuts = get_ppk_decisions(direction="cut")
    hikes = get_ppk_decisions(direction="hike")
    print(f"  Resolved decisions → {len(cuts)} cut(s), {len(hikes)} hike(s) "
          f"(seed+EVDS merged). Latest cut: {cuts[-1]['date'] if cuts else '—'}")


if __name__ == "__main__":
    live = check_rss()
    check_evds()
    sys.exit(0 if live else 1)

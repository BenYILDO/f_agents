"""TCMB policy-rate (PPK) decision calendar — seed table + optional EVDS feed.

The Macro analyst's event study needs the dates and directions of past TCMB
Para Politikası Kurulu (PPK) decisions so it can measure how a given BIST name
historically behaved after, e.g., a rate cut. There are two sources here and
they compose:

  1. **EVDS (authoritative, live).** When an ``EVDS_API_KEY`` is configured we
     pull the official policy-rate series from the TCMB EVDS API and derive
     decisions deterministically from its *change points* (a value change from
     the prior observation is a hike/cut on that date). This is the "net",
     always-current source — no hand-maintained dates.
  2. **Curated seed table (fallback).** A small, explicitly-sourced table of
     recent decisions used when no key is set, the network is unavailable, or
     EVDS returns nothing. It is a *seed*, not the source of truth — verify and
     extend against https://www.tcmb.gov.tr (Para Politikası → PPK Kararları).

``get_ppk_decisions`` merges the two (EVDS wins on date collisions), caches the
result per (series, window) so a 40-ticker scan triggers at most one EVDS call,
and never raises — on any failure it degrades to the seed table.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# EVDS REST endpoint and the policy-rate series code. The 1-week repo policy
# rate is the headline PPK rate; the series code is overridable via env in case
# TCMB renames it, and a wrong code simply falls back to the seed table below.
_EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"
_DEFAULT_POLICY_RATE_SERIES = os.environ.get("EVDS_POLICY_RATE_SERIES", "TP.APIFON4")

# Curated SEED of recent PPK decisions (turning points of the 2023–2024 cycle).
# Direction is relative to the previous decision. This is a fallback baseline —
# when an EVDS key is present, live data supersedes it. Verify/extend against
# tcmb.gov.tr. (rate = announced 1-week repo policy rate, %.)
PPK_DECISIONS_SEED: tuple[dict, ...] = (
    {"date": "2023-06-22", "rate": 15.0, "direction": "hike"},
    {"date": "2023-07-20", "rate": 17.5, "direction": "hike"},
    {"date": "2023-08-24", "rate": 25.0, "direction": "hike"},
    {"date": "2023-09-21", "rate": 30.0, "direction": "hike"},
    {"date": "2023-10-26", "rate": 35.0, "direction": "hike"},
    {"date": "2023-11-23", "rate": 40.0, "direction": "hike"},
    {"date": "2023-12-21", "rate": 42.5, "direction": "hike"},
    {"date": "2024-01-25", "rate": 45.0, "direction": "hike"},
    {"date": "2024-03-21", "rate": 50.0, "direction": "hike"},
    {"date": "2024-12-26", "rate": 47.5, "direction": "cut"},
    {"date": "2025-01-23", "rate": 45.0, "direction": "cut"},
)

# In-process cache: decisions are ticker-independent, so a scan reuses one fetch.
# Keyed by (series, start, end); value is the merged decision list.
_CACHE: dict[tuple[str, str, str], list[dict]] = {}


def _parse_evds_date(raw: str) -> Optional[date]:
    """Parse an EVDS ``Tarih`` field, which may be daily, monthly, or yearly."""
    raw = (raw or "").strip()
    for fmt in ("%d-%m-%Y", "%m-%Y", "%Y-%m-%d", "%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _fetch_evds_observations(
    api_key: str, series: str, start: str, end: str, timeout: float = 10.0,
) -> list[tuple[date, float]]:
    """Fetch (date, rate) observations for ``series`` from EVDS. [] on any failure.

    ``start``/``end`` are ISO ``YYYY-MM-DD``; EVDS wants ``DD-MM-YYYY``. The key
    is sent as the ``key`` header per the EVDS API contract.
    """
    def _ddmmyyyy(iso: str) -> str:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d-%m-%Y")

    url = (
        f"{_EVDS_BASE}/series={series}&startDate={_ddmmyyyy(start)}"
        f"&endDate={_ddmmyyyy(end)}&type=json"
    )
    req = Request(url, headers={"key": api_key, "User-Agent": "tradingagents/0.2"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        logger.warning("EVDS fetch failed for %s: %s", series, exc)
        return []

    items = payload.get("items") or []
    value_key = series.replace(".", "_")
    obs: list[tuple[date, float]] = []
    for it in items:
        d = _parse_evds_date(it.get("Tarih", ""))
        raw_val = it.get(value_key)
        if d is None or raw_val in (None, "", "null"):
            continue
        try:
            obs.append((d, float(raw_val)))
        except (TypeError, ValueError):
            continue
    obs.sort(key=lambda t: t[0])
    return obs


def _decisions_from_rate_series(observations: list[tuple[date, float]]) -> list[dict]:
    """Derive PPK decisions from rate change points in an observation series.

    A change vs. the previous non-null observation is a decision on that date,
    with direction ``hike`` (rate up) or ``cut`` (rate down). Unchanged
    observations (holds) are not emitted — the event study conditions on
    cut/hike moves.
    """
    decisions: list[dict] = []
    prev: Optional[float] = None
    for d, rate in observations:
        if prev is not None and rate != prev:
            decisions.append({
                "date": d.strftime("%Y-%m-%d"),
                "rate": rate,
                "direction": "hike" if rate > prev else "cut",
            })
        prev = rate
    return decisions


def get_ppk_decisions(
    direction: Optional[str] = None,
    api_key: Optional[str] = None,
    series: Optional[str] = None,
    start: str = "2022-06-01",
    end: Optional[str] = None,
) -> list[dict]:
    """Return PPK decisions (EVDS-authoritative, seed-fallback), newest last.

    Args:
        direction: keep only ``"cut"`` / ``"hike"`` decisions when set; None = all.
        api_key: EVDS key; falls back to ``EVDS_API_KEY`` env. When absent, only
            the seed table is used.
        series: EVDS policy-rate series code (defaults to the env/module default).
        start/end: ISO date window; ``end`` defaults to today.

    Never raises. Each item: ``{date, rate, direction}``.
    """
    api_key = api_key or os.environ.get("EVDS_API_KEY")
    series = series or _DEFAULT_POLICY_RATE_SERIES
    end = end or date.today().strftime("%Y-%m-%d")
    cache_key = (series, start, end)

    if cache_key in _CACHE:
        merged = _CACHE[cache_key]
    else:
        merged = _merge_decisions(api_key, series, start, end)
        _CACHE[cache_key] = merged

    if direction in ("cut", "hike"):
        return [d for d in merged if d["direction"] == direction]
    return list(merged)


def _merge_decisions(api_key: Optional[str], series: str, start: str, end: str) -> list[dict]:
    """Merge EVDS-derived decisions (authoritative) with the seed table."""
    by_date: dict[str, dict] = {}
    # Seed first so EVDS can overwrite on date collisions.
    for d in PPK_DECISIONS_SEED:
        if start <= d["date"] <= end:
            by_date[d["date"]] = dict(d)

    if api_key:
        obs = _fetch_evds_observations(api_key, series, start, end)
        for d in _decisions_from_rate_series(obs):
            by_date[d["date"]] = d  # EVDS wins
        if obs:
            logger.info("EVDS supplied %d policy-rate observations for %s", len(obs), series)

    return sorted(by_date.values(), key=lambda d: d["date"])


def clear_cache() -> None:
    """Drop the in-process decision cache (used by tests)."""
    _CACHE.clear()

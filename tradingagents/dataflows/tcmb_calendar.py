"""TCMB EVDS backbone — policy-rate decisions + official macro numbers.

The Macro analyst's most reliable leg is official TCMB data, fetched from the
EVDS API when an ``EVDS_API_KEY`` is configured. Two things are derived here:

  1. **PPK decisions** (:func:`resolve_ppk`): dates and directions of past rate
     decisions, derived deterministically from *change points* in the official
     policy-rate series. A curated seed table is the fallback when no key is
     set or EVDS is unreachable — it is a seed, not the source of truth; verify
     against https://www.tcmb.gov.tr.
  2. **Official macro numbers** (:func:`get_macro_official_numbers`): the latest
     policy rate, USD/TRY, and (optionally) CPI — hard numbers the analyst can
     stand on even if the news RSS feeds are blocked.

Both paths report a :class:`SourceHealth` so the desk sees, loudly, whether it
is running on live official data or a fallback. Results are cached per window
so a session triggers at most one EVDS call per series. Nothing here raises.

Series codes are env-overridable (TCMB occasionally renames them); a wrong code
simply surfaces as an unhealthy/empty source in the data-health panel rather
than a silent gap.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tradingagents.dataflows.data_health import SourceHealth, OK, EMPTY, ERROR

logger = logging.getLogger(__name__)

# TCMB migrated EVDS to the evds3 host in 2026; the legacy
# evds2.tcmb.gov.tr/service/evds path now 302-redirects to the evds3 SPA and
# returns HTML, so json.loads fails. The live REST base is now the igmevdsms-dis
# service on evds3 (key still sent in the request header; same DD-MM-YYYY /
# "Tarih" / "TP_APIFON4" response shape the parser below already expects).
_EVDS_BASE = "https://evds3.tcmb.gov.tr/igmevdsms-dis"
# 1-week repo policy rate, USD/TRY selling rate, and CPI. CPI has no default —
# it is opt-in via env so we never guess a code we are unsure of.
_DEFAULT_POLICY_RATE_SERIES = os.environ.get("EVDS_POLICY_RATE_SERIES", "TP.APIFON4")
_USD_TRY_SERIES = os.environ.get("EVDS_USDTRY_SERIES", "TP.DK.USD.S.YTL")
_CPI_SERIES = os.environ.get("EVDS_CPI_SERIES", "")

# Curated SEED of recent PPK decisions (turning points of the 2023–2024 cycle).
# Fallback only — EVDS supersedes when keyed. Verify/extend against tcmb.gov.tr.
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

# Caches (per window). Decisions and official numbers are ticker-independent.
_PPK_CACHE: dict[tuple[str, str, str], tuple[list[dict], SourceHealth]] = {}
_NUM_CACHE: dict[tuple[str, str], tuple[str, list[SourceHealth]]] = {}


def _parse_evds_date(raw: str) -> Optional[date]:
    """Parse an EVDS ``Tarih`` field (daily / monthly / yearly)."""
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
    """Fetch (date, value) observations for ``series`` from EVDS. [] on failure.

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


def fetch_evds_series(
    label: str, series: str, api_key: Optional[str], start: str, end: str,
    timeout: float = 10.0,
) -> tuple[list[tuple[date, float]], SourceHealth]:
    """Fetch one EVDS series and classify its health (ok/empty/error)."""
    if not api_key:
        return [], SourceHealth(label, EMPTY, "EVDS_API_KEY yok", 0)
    obs = _fetch_evds_observations(api_key, series, start, end, timeout)
    if obs:
        return obs, SourceHealth(label, OK, f"son {obs[-1][0]} = {obs[-1][1]}", len(obs))
    return [], SourceHealth(label, ERROR, f"seri boş/erişilemedi ({series})", 0)


# EVDS exposes the realized funding cost (TP.APIFON4), which mostly pins to the
# announced policy rate but wobbles fractionally (e.g. 38.00 ↔ 37.30, or a
# multi-day ramp 38 → 39.35 → 39.98 → 40.0 around a single decision). A naive
# "any change is a decision" rule turns that sub-point noise into hundreds of
# phantom PPK decisions. TCMB acts in ≥250bp steps in the current cycle, and the
# observed funding-cost wobble stays under ~1.35 points, so we only register a
# decision once the rate moves at least this far from the last *accepted* level —
# collapsing both the wobble and the ramp into one decision. Tunable if TCMB
# returns to sub-150bp moves.
_MIN_RATE_CHANGE = 1.5


def _decisions_from_rate_series(observations: list[tuple[date, float]]) -> list[dict]:
    """Derive PPK decisions from rate change points in an observation series.

    A move of at least :data:`_MIN_RATE_CHANGE` points vs. the last *accepted*
    decision level is a decision on that date, direction ``hike`` (up) / ``cut``
    (down). Holds and sub-threshold funding-cost wobble are not emitted.
    """
    decisions: list[dict] = []
    last_accepted: Optional[float] = None
    for d, rate in observations:
        if last_accepted is None:
            last_accepted = rate
            continue
        if abs(rate - last_accepted) >= _MIN_RATE_CHANGE:
            decisions.append({
                "date": d.strftime("%Y-%m-%d"),
                "rate": rate,
                "direction": "hike" if rate > last_accepted else "cut",
            })
            last_accepted = rate
    return decisions


def _merge_decisions(
    api_key: Optional[str], series: str, start: str, end: str,
) -> tuple[list[dict], SourceHealth]:
    """Merge EVDS-derived decisions (authoritative) with the seed table + health."""
    by_date: dict[str, dict] = {
        d["date"]: dict(d) for d in PPK_DECISIONS_SEED if start <= d["date"] <= end
    }
    if not api_key:
        health = SourceHealth("TCMB EVDS faiz kararları", EMPTY,
                              "EVDS_API_KEY yok — seed tablo kullanıldı", len(by_date))
    else:
        obs = _fetch_evds_observations(api_key, series, start, end)
        if obs:
            for d in _decisions_from_rate_series(obs):
                by_date[d["date"]] = d  # EVDS wins on collision
            health = SourceHealth("TCMB EVDS faiz kararları", OK,
                                  f"canlı seri, son {obs[-1][0]} = {obs[-1][1]}%", len(obs))
        else:
            health = SourceHealth("TCMB EVDS faiz kararları", ERROR,
                                  f"EVDS boş/erişilemedi ({series}) — seed tablo", len(by_date))
    decisions = sorted(by_date.values(), key=lambda d: d["date"])
    return decisions, health


def resolve_ppk(
    direction: Optional[str] = None,
    api_key: Optional[str] = None,
    series: Optional[str] = None,
    start: str = "2022-06-01",
    end: Optional[str] = None,
) -> tuple[list[dict], SourceHealth]:
    """Return ``(decisions, health)``; EVDS-authoritative, seed-fallback. Cached."""
    api_key = api_key or os.environ.get("EVDS_API_KEY")
    series = series or _DEFAULT_POLICY_RATE_SERIES
    end = end or date.today().strftime("%Y-%m-%d")
    cache_key = (series, start, end)

    if cache_key not in _PPK_CACHE:
        _PPK_CACHE[cache_key] = _merge_decisions(api_key, series, start, end)
    decisions, health = _PPK_CACHE[cache_key]

    if direction in ("cut", "hike"):
        decisions = [d for d in decisions if d["direction"] == direction]
    return list(decisions), health


def get_ppk_decisions(
    direction: Optional[str] = None,
    api_key: Optional[str] = None,
    series: Optional[str] = None,
    start: str = "2022-06-01",
    end: Optional[str] = None,
) -> list[dict]:
    """Back-compat thin wrapper returning only the decision list (no health)."""
    decisions, _ = resolve_ppk(direction, api_key, series, start, end)
    return decisions


def get_macro_official_numbers(
    api_key: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[str, list[SourceHealth]]:
    """Latest official policy rate / USD-TRY / CPI from EVDS as a fact block.

    Returns ``(text, [SourceHealth])``. CPI is included only when
    ``EVDS_CPI_SERIES`` is configured (no guessed default). Cached per window.
    """
    api_key = api_key or os.environ.get("EVDS_API_KEY")
    end = end or date.today().strftime("%Y-%m-%d")
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")
    cache_key = (start, end)
    if cache_key in _NUM_CACHE:
        return _NUM_CACHE[cache_key]

    series_list = [
        ("Politika Faizi (%)", _DEFAULT_POLICY_RATE_SERIES),
        ("USD/TRY", _USD_TRY_SERIES),
    ]
    if _CPI_SERIES:
        series_list.append(("TÜFE", _CPI_SERIES))

    lines: list[str] = []
    healths: list[SourceHealth] = []
    for label, series in series_list:
        obs, h = fetch_evds_series(label, series, api_key, start, end)
        healths.append(h)
        if obs:
            lines.append(f"  {label}: {obs[-1][1]} (tarih {obs[-1][0]})")

    if lines:
        text = "TCMB RESMİ VERİLER (EVDS, en güncel gözlem):\n" + "\n".join(lines)
    elif not api_key:
        text = "<TCMB EVDS resmi verileri yok: EVDS_API_KEY tanımlı değil>"
    else:
        text = "<TCMB EVDS resmi serilerine ulaşılamadı (seri kodu/anahtar kontrol edin)>"

    _NUM_CACHE[cache_key] = (text, healths)
    return text, healths


def clear_cache() -> None:
    """Drop in-process caches (used by tests)."""
    _PPK_CACHE.clear()
    _NUM_CACHE.clear()

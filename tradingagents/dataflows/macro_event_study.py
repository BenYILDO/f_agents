"""Macro event study — how a ticker historically behaved after similar events.

Given a list of past macro-event dates (e.g. TCMB rate-cut decisions from
:mod:`tcmb_calendar`) this computes the target ticker's forward returns after
each event and aggregates them, so the Macro analyst can tell the rest of the
desk a concrete, data-grounded fact like: "after the last 6 TCMB rate cuts,
THYAO returned a median +3.1% over the next 5 trading days, positive 5/6 times."

Everything here is deterministic (prices in, statistics out) — the LLM is handed
the computed numbers, never asked to estimate them. Optionally each event's raw
return is measured against a benchmark index (BIST 100 by default) to separate
the stock's move from the market's. Degrades gracefully: no price data or no
events yields an ``ok=False`` result with a human-readable reason, never a raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = (1, 5, 10)


@dataclass
class EventStudyResult:
    ok: bool
    ticker: str
    direction_label: str                 # e.g. "faiz indirimi"
    n_events: int = 0
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    # horizon -> {"mean", "median", "hit_rate", "n", "mean_alpha"|None}
    stats: dict = field(default_factory=dict)
    per_event: list = field(default_factory=list)  # [{date, fwd_1, fwd_5, ...}]
    benchmark: Optional[str] = None
    reason: str = ""


def _forward_returns(prices: pd.Series, entry_idx: int, horizons: Sequence[int]) -> dict:
    """Pct return from ``entry_idx`` close to ``entry_idx+h`` close for each h."""
    out: dict[int, Optional[float]] = {}
    base = prices.iloc[entry_idx]
    for h in horizons:
        tgt = entry_idx + h
        if tgt < len(prices) and base:
            out[h] = float((prices.iloc[tgt] - base) / base)
        else:
            out[h] = None
    return out


def _entry_index(index: pd.DatetimeIndex, event_date: pd.Timestamp) -> Optional[int]:
    """First bar on/after the event date (the tradable reaction starts there)."""
    pos = index.searchsorted(event_date, side="left")
    return int(pos) if pos < len(index) else None


def compute_event_study(
    ticker: str,
    event_dates: Sequence[str],
    direction_label: str = "",
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    benchmark: Optional[str] = "XU100.IS",
    max_events: int = 12,
    history: Optional[pd.DataFrame] = None,
    benchmark_history: Optional[pd.DataFrame] = None,
) -> EventStudyResult:
    """Measure ``ticker`` forward returns after each event date and aggregate.

    ``history``/``benchmark_history`` let callers (and tests) inject OHLCV
    instead of hitting the network; when omitted they are fetched via yfinance.
    Only the most recent ``max_events`` events are used (recency-weighted
    relevance). Never raises.
    """
    horizons = tuple(horizons)
    if not event_dates:
        return EventStudyResult(False, ticker, direction_label, horizons=horizons,
                                reason="Karşılaştırılacak geçmiş benzer olay bulunamadı.")

    events = sorted(event_dates)[-max_events:]
    span_start = (pd.Timestamp(min(events)) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    span_end = (pd.Timestamp(max(events)) + pd.Timedelta(days=40)).strftime("%Y-%m-%d")

    try:
        px = history if history is not None else \
            yf.Ticker(ticker).history(start=span_start, end=span_end)
    except Exception as exc:  # noqa: BLE001
        return EventStudyResult(False, ticker, direction_label, horizons=horizons,
                                reason=f"Fiyat verisi alınamadı: {exc}")
    if px is None or px.empty or "Close" not in px or len(px) < max(horizons) + 2:
        return EventStudyResult(False, ticker, direction_label, horizons=horizons,
                                reason="Olay-etüdü için yeterli fiyat verisi yok.")

    px = px.copy()
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    close = px["Close"].astype("float64")

    bench_close = None
    if benchmark:
        try:
            bpx = benchmark_history if benchmark_history is not None else \
                yf.Ticker(benchmark).history(start=span_start, end=span_end)
            if bpx is not None and not bpx.empty and "Close" in bpx:
                if getattr(bpx.index, "tz", None) is not None:
                    bpx.index = bpx.index.tz_localize(None)
                bench_close = bpx["Close"].astype("float64")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Benchmark %s unavailable for event study: %s", benchmark, exc)

    per_event: list[dict] = []
    for ev in events:
        ts = pd.Timestamp(ev)
        ei = _entry_index(close.index, ts)
        if ei is None:
            continue
        fwd = _forward_returns(close, ei, horizons)
        row = {"date": ev, **{f"fwd_{h}": fwd[h] for h in horizons}}
        if bench_close is not None:
            bi = _entry_index(bench_close.index, ts)
            if bi is not None:
                bfwd = _forward_returns(bench_close, bi, horizons)
                for h in horizons:
                    if fwd[h] is not None and bfwd[h] is not None:
                        row[f"alpha_{h}"] = fwd[h] - bfwd[h]
        per_event.append(row)

    stats: dict[int, dict] = {}
    for h in horizons:
        vals = [r[f"fwd_{h}"] for r in per_event if r.get(f"fwd_{h}") is not None]
        if not vals:
            continue
        alphas = [r[f"alpha_{h}"] for r in per_event if r.get(f"alpha_{h}") is not None]
        stats[h] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(median(vals)),
            "hit_rate": float(sum(1 for v in vals if v > 0) / len(vals)),
            "mean_alpha": float(np.mean(alphas)) if alphas else None,
        }

    if not stats:
        return EventStudyResult(False, ticker, direction_label, horizons=horizons,
                                n_events=len(per_event),
                                reason="Olay tarihleri fiyat verisi penceresine düşmedi.")

    return EventStudyResult(
        ok=True, ticker=ticker, direction_label=direction_label, n_events=len(per_event),
        horizons=horizons, stats=stats, per_event=per_event,
        benchmark=benchmark if bench_close is not None else None,
    )


def format_event_study(result: EventStudyResult) -> str:
    """Render an :class:`EventStudyResult` as a Turkish prompt block."""
    base = result.ticker.upper().removesuffix(".IS")
    if not result.ok:
        return (f"<Geçmiş olay-etüdü ({base}, {result.direction_label or 'benzer olaylar'}): "
                f"{result.reason}>")

    lines = [
        f"GEÇMİŞ OLAY-ETÜDÜ — {base}, benzer makro olay: "
        f"\"{result.direction_label or 'TCMB kararı'}\" (n={result.n_events} olay):"
    ]
    for h in result.horizons:
        s = result.stats.get(h)
        if not s:
            continue
        line = (f"  +{h} işlem günü → ort %{s['mean']*100:+.1f} · medyan %{s['median']*100:+.1f} · "
                f"pozitif oran %{s['hit_rate']*100:.0f} (n={s['n']})")
        if s.get("mean_alpha") is not None:
            line += f" · endekse göre ort. alfa %{s['mean_alpha']*100:+.1f}"
        lines.append(line)
    if result.benchmark:
        lines.append(f"  (alfa karşılaştırması: {result.benchmark})")
    lines.append("  Not: geçmiş davranış geleceğin garantisi değildir; örneklem küçük olabilir.")
    return "\n".join(lines)

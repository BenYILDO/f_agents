"""Analiz orchestrator — bir hisseyi snapshot'a indirger (LLM yok, ücretsiz).

Tek kaynak: hem Streamlit'teki "ekleyince anında analiz" / "şimdi analiz et"
butonu hem de saat başı zamanlayıcı (GitHub Actions) bu modülü çağırır. Üç
deterministik motoru birleştirir:

  - :func:`combined_signal`  — temel (rasyo) + teknik (kompozit) → birleşik karar
  - :func:`dip_analyze`      — SMI/VWMA/Bollinger güncel AL/SAT durumu
  - :mod:`trust`             — çoklu-yöntem mutabakatı + veri tazeliği

Çıktı :class:`AnalysisOutcome`; :func:`to_snapshot_row` ile ``analysis_snapshots``
satırına çevrilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from tradingagents.analysis import trust
from tradingagents.analytics.combined import combined_signal
from tradingagents.analytics.composite import _fetch_daily
from tradingagents.strategy.dip_signal import analyze as dip_analyze

_STATUS_FROM_DIP = {"AL BÖLGESİ": "AL", "SAT UYARISI": "SAT"}
_RATIO_DIR = {"AL": trust.UP, "SAT": trust.DOWN, "TUT": trust.FLAT}


def _composite_dir(score: float | None, ok: bool) -> int:
    if not ok or score is None:
        return trust.FLAT
    if score >= 20:
        return trust.UP
    if score <= -20:
        return trust.DOWN
    return trust.FLAT


def _dip_dir(status: str) -> int:
    return {"AL BÖLGESİ": trust.UP, "SAT UYARISI": trust.DOWN}.get(status, trust.FLAT)


@dataclass
class AnalysisOutcome:
    """Bir hissenin tek noktadan deterministik analiz çıktısı."""

    ticker: str
    ok: bool = False
    status: str = "NÖTR"          # AL | SAT | NÖTR (Dip-Al güncel durumu)
    decision: str = "VERİ YOK"    # GÜÇLÜ AL | AL | TUT | SAT | KAÇIN (birleşik)
    combined_score: Optional[float] = None
    tech_score: Optional[float] = None
    ratio_score: Optional[float] = None
    ratio_verdict: Optional[str] = None
    confidence: str = ""
    agreement_level: str = "nötr"
    agreement: str = ""
    close: Optional[float] = None
    smi: Optional[float] = None
    signals: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)
    rationale: list = field(default_factory=list)
    error: str = ""


def _freshness(df: Optional[pd.DataFrame]) -> dict:
    """Fiyat verisinin tazeliği — son bar yaşı (gün). Bayatsa yüksek sesle işaretle."""
    if df is None or df.empty:
        return {"price": "error"}
    try:
        last_date = pd.Timestamp(df.index[-1]).date()
        age = (datetime.now(timezone.utc).date() - last_date).days
        return {
            "last_bar": last_date.isoformat(),
            "price": "ok" if age <= 4 else f"stale ({age}g)",
        }
    except Exception:  # noqa: BLE001
        return {"price": "ok"}


def analyze_ticker(
    ticker: str,
    interval_label: str = "Günlük (1g)",
    *,
    df: Optional[pd.DataFrame] = None,
) -> AnalysisOutcome:
    """Bir hisseyi deterministik motorlarla analiz eder. Asla istisna fırlatmaz."""
    ticker = ticker.strip().upper()
    if df is None:
        df = _fetch_daily(ticker)

    comb = combined_signal(ticker, df=df)
    dip = dip_analyze(ticker, interval_label)

    health = _freshness(df)
    health["ratio"] = "ok" if (comb.fundamental and comb.fundamental.ok) else "missing"

    if not comb.ok and not dip.ok:
        return AnalysisOutcome(
            ticker=ticker,
            ok=False,
            error=comb.error or dip.error or "Veri yok",
            health=health,
        )

    tech_ok = comb.technical is not None
    tech_score = comb.technical.score if tech_ok else None
    confidence = comb.technical.confidence if tech_ok else ""
    ratio_ok = comb.fundamental is not None
    ratio_score = comb.fundamental.score if ratio_ok else None
    ratio_verdict = comb.fundamental.verdict if ratio_ok else None

    dip_status_label = dip.status if dip.ok else "—"
    votes = {
        "teknik": _composite_dir(tech_score, tech_ok),
        "rasyo": _RATIO_DIR.get(ratio_verdict, trust.FLAT),
        "dip-strateji": _dip_dir(dip_status_label),
    }
    agr_level, agr_summary = trust.agreement(votes)

    close = comb.last_close or None
    smi = None
    signals: dict = {}
    if dip.ok and dip.df is not None and not dip.df.empty:
        last = dip.df.iloc[-1]
        if pd.notna(last.get("smi")):
            smi = round(float(last["smi"]), 2)
        if pd.notna(last.get("Close")):
            close = float(last["Close"])
        signals = {
            "dip_status": dip.status,
            "dip_conditions": {k: bool(v) for k, v in (dip.conditions or {}).items()},
            "recent": (dip.signals or [])[-3:],
        }

    return AnalysisOutcome(
        ticker=ticker,
        ok=True,
        status=_STATUS_FROM_DIP.get(dip_status_label, "NÖTR"),
        decision=comb.decision,
        combined_score=comb.combined_score,
        tech_score=tech_score,
        ratio_score=ratio_score,
        ratio_verdict=ratio_verdict,
        confidence=confidence,
        agreement_level=agr_level,
        agreement=agr_summary,
        close=round(close, 4) if close else None,
        smi=smi,
        signals=signals,
        health=health,
        rationale=list(comb.rationale),
    )


def to_snapshot_row(
    outcome: AnalysisOutcome, scope: str = "portfolio", source: str = "cron"
) -> dict:
    """:class:`AnalysisOutcome` → ``analysis_snapshots`` satırı (Supabase)."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ticker": outcome.ticker,
        "scope": scope,
        "source": source,
        "status": outcome.status,
        "decision": outcome.decision,
        "combined_score": outcome.combined_score,
        "tech_score": outcome.tech_score,
        "ratio_score": outcome.ratio_score,
        "ratio_verdict": outcome.ratio_verdict,
        "confidence": outcome.confidence,
        "agreement": f"{outcome.agreement_level}: {outcome.agreement}",
        "close": outcome.close,
        "smi": outcome.smi,
        "signals": outcome.signals or {},
        "health": outcome.health or {},
    }


def analyze_universe(
    tickers: list[str], interval_label: str = "Günlük (1g)"
) -> list[AnalysisOutcome]:
    """Bir hisse listesini sırayla analiz eder (cron/tarama için)."""
    seen: set[str] = set()
    outcomes: list[AnalysisOutcome] = []
    for raw in tickers:
        ticker = raw.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        outcomes.append(analyze_ticker(ticker, interval_label))
    return outcomes

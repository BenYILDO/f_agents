"""Analiz geçmişi (``analysis_snapshots``) — saatlik snapshot yaz/oku.

Saat başı zamanlayıcı ve "şimdi analiz et" butonu buraya yazar; Portföyüm /
Otomatik Analiz ekranları ve (ileride) geçmiş karşılaştırma ekranı buradan okur.
Güven katmanı (sinyal karnesi, istikrar) bu geçmiş üzerine kurulur.
"""

from __future__ import annotations

from tradingagents.storage.supabase_client import SupabaseREST

_TABLE = "analysis_snapshots"
_LATEST_VIEW = "latest_snapshots"


def write_snapshots(rows: list[dict]) -> list[dict]:
    """Birden çok snapshot satırını tek seferde yazar."""
    if not rows:
        return []
    return SupabaseREST().insert(_TABLE, rows)


def write_snapshot(row: dict) -> dict:
    """Tek bir snapshot satırı yazar."""
    return write_snapshots([row])[0]


def latest(scope: str | None = None) -> list[dict]:
    """Her (ticker, scope) için en güncel snapshot (``latest_snapshots`` view'i)."""
    params: dict[str, str] = {"order": "ticker.asc"}
    if scope:
        params["scope"] = f"eq.{scope}"
    return SupabaseREST().select(_LATEST_VIEW, params)


def latest_for(tickers: list[str], scope: str = "portfolio") -> dict[str, dict]:
    """Verilen sembollerin en güncel snapshot'ını sembol→satır olarak döndürür."""
    want = {t.strip().upper() for t in tickers}
    return {r["ticker"]: r for r in latest(scope) if r["ticker"] in want}


def history(ticker: str, scope: str | None = None, limit: int = 500) -> list[dict]:
    """Bir sembolün snapshot geçmişi (en yeni → en eski)."""
    params: dict[str, str] = {
        "ticker": f"eq.{ticker.strip().upper()}",
        "order": "ts.desc",
        "limit": str(limit),
    }
    if scope:
        params["scope"] = f"eq.{scope}"
    return SupabaseREST().select(_TABLE, params)

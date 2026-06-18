"""Gecelik model önbelleği (Faz H) — kalibre olasılık + DSR'yi sakla/oku.

Gecelik ağır iş (``scripts/run_nightly_models.py``) her ticker için kalibre
yukarı-olasılığını hesaplayıp buraya yazar; saatlik iş bunu okuyup güven skoruna
katar. Böylece saatlik cron ağır ML çalıştırmaz.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tradingagents.storage.supabase_client import SupabaseREST

_TABLE = "model_cache"


def upsert_models(rows: list[dict]) -> list[dict]:
    """Ticker başına tek satır olacak şekilde model önbelleğini günceller."""
    if not rows:
        return []
    for r in rows:
        r.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    return SupabaseREST().upsert(_TABLE, rows, on_conflict="ticker")


def read_models() -> dict[str, dict]:
    """Tüm model önbelleğini ticker→satır olarak döndürür (hata → boş)."""
    try:
        rows = SupabaseREST().select(_TABLE, {"order": "ticker.asc"})
    except Exception:  # noqa: BLE001 — önbellek yoksa güven onsuz çalışır
        return {}
    return {r["ticker"]: r for r in rows}

"""İnce Supabase (PostgREST) REST istemcisi — yalnız ``requests`` bağımlılığı.

Neden ``supabase-py`` değil? Tek kullanıcılı kişisel bir uygulamada sorgular
basittir (select/insert/update/delete + birkaç filtre). PostgREST'e doğrudan
``requests`` ile gitmek hem yeni ağır bir bağımlılık (gotrue/postgrest sürüm
çakışmaları) getirmez hem de Streamlit Community Cloud / GitHub Actions
ortamlarında öngörülebilir kalır.

Kimlik bilgileri sırayla aranır: önce ortam değişkeni (``os.environ`` — GitHub
Actions secrets böyle gelir), sonra Streamlit ``st.secrets`` (Community Cloud
böyle enjekte eder). Tek kullanıcılı kurulumda ``SUPABASE_SERVICE_KEY``
(RLS'i bypass eder) sunucu tarafı secret olarak kullanılır; tarayıcıya hiçbir
zaman sızmaz (bkz. ``docs/SUPABASE_SETUP.md``).
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import requests

_DEFAULT_TIMEOUT = 15


class SupabaseError(RuntimeError):
    """Supabase yapılandırması eksik ya da REST çağrısı başarısız."""


def _secret(key: str) -> str | None:
    """Bir secret'ı önce ortamdan, sonra Streamlit secrets'tan okur."""
    val = os.environ.get(key)
    if val:
        return val
    try:  # Streamlit yoksa / secrets dosyası yoksa sessizce geç
        import streamlit as st  # type: ignore

        try:
            sval = st.secrets.get(key)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — secrets okunamıyorsa env'e güven
            sval = None
        if sval:
            return str(sval)
    except Exception:  # noqa: BLE001 — streamlit import edilemiyorsa (cron) sorun yok
        pass
    return None


def get_credentials() -> tuple[str, str]:
    """(url, key) döndürür; eksikse yol gösteren bir :class:`SupabaseError`."""
    url = _secret("SUPABASE_URL")
    key = (
        _secret("SUPABASE_SERVICE_KEY")
        or _secret("SUPABASE_KEY")
        or _secret("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        raise SupabaseError(
            "Supabase yapılandırılmamış. SUPABASE_URL ve SUPABASE_SERVICE_KEY "
            "(ya da SUPABASE_KEY) değerlerini ortam değişkeni veya Streamlit "
            "secrets olarak ayarla. Kurulum rehberi: docs/SUPABASE_SETUP.md"
        )
    return url.rstrip("/"), key


def is_configured() -> bool:
    """Supabase kimlik bilgileri var mı? (UI'da nazik uyarı için.)"""
    try:
        get_credentials()
        return True
    except SupabaseError:
        return False


class SupabaseREST:
    """PostgREST tabloları için minimal CRUD sarmalayıcı.

    Kimlik bilgileri verilmezse :func:`get_credentials` ile ortam/secrets'tan
    çözülür. Tüm metotlar 4xx/5xx durumunda :class:`SupabaseError` fırlatır.
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        if url and key:
            self.url, self.key = url.rstrip("/"), key
        else:
            self.url, self.key = get_credentials()
        self.timeout = timeout
        self._base = f"{self.url}/rest/v1"

    # ── HTTP yardımcıları ────────────────────────────────────────────────
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if extra:
            h.update(extra)
        return h

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code >= 400:
            raise SupabaseError(f"Supabase {resp.status_code}: {resp.text[:400]}")

    # ── CRUD ─────────────────────────────────────────────────────────────
    def select(self, table: str, params: dict[str, Any] | None = None) -> list[dict]:
        """GET /<table> — PostgREST filtre/sıralama parametreleriyle."""
        q = dict(params or {})
        q.setdefault("select", "*")
        resp = requests.get(
            f"{self._base}/{table}", headers=self._headers(), params=q, timeout=self.timeout
        )
        self._raise_for_status(resp)
        return resp.json()

    def insert(self, table: str, rows: dict | Iterable[dict]) -> list[dict]:
        """POST /<table> — tek satır ya da satır listesi; eklenen satırları döndürür."""
        payload = [rows] if isinstance(rows, dict) else list(rows)
        resp = requests.post(
            f"{self._base}/{table}",
            headers=self._headers({"Prefer": "return=representation"}),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return resp.json()

    def update(self, table: str, values: dict, match: dict[str, Any]) -> list[dict]:
        """PATCH /<table> — ``match`` (eq filtreleri) ile eşleşen satırları günceller."""
        params = {col: f"eq.{val}" for col, val in match.items()}
        resp = requests.patch(
            f"{self._base}/{table}",
            headers=self._headers({"Prefer": "return=representation"}),
            params=params,
            json=values,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return resp.json()

    def delete(self, table: str, match: dict[str, Any]) -> None:
        """DELETE /<table> — ``match`` (eq filtreleri) ile eşleşen satırları siler."""
        params = {col: f"eq.{val}" for col, val in match.items()}
        resp = requests.delete(
            f"{self._base}/{table}", headers=self._headers(), params=params, timeout=self.timeout
        )
        self._raise_for_status(resp)

    def rpc(self, fn: str, params: dict[str, Any] | None = None) -> Any:
        """POST /rpc/<fn> — bir Postgres fonksiyonunu çağırır (örn. retention)."""
        resp = requests.post(
            f"{self._base}/rpc/{fn}",
            headers=self._headers(),
            json=params or {},
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return resp.json() if resp.content else None

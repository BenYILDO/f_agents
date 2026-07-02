"""Havuz modeli deposu (S2) — eğitilmiş artefakt + karne Supabase'de.

Kullanıcının kök talebi: "modeller Supabase'de dursun ki her gün geçmiş veriyle
baştan eğitilmesin." Gecelik iş modeli bir kez eğitir, artefaktı (pickle+zlib+
base64) ve kalite karnesini buraya yazar; gün içi işler yalnız en son geçerli
modeli indirip **tahmin** yapar. Her eğitim yeni satırdır (insert, upsert değil):
karne tarihçesi birikir → modelin zamanla iyileşip iyileşmediği izlenebilir.
"""

from __future__ import annotations

from tradingagents.storage.supabase_client import SupabaseREST

_TABLE = "pooled_models"


def save_model(row: dict) -> list[dict]:
    """Yeni eğitim kaydı ekler (artefakt + karne). Satırı döndürür."""
    return SupabaseREST().insert(_TABLE, row)


def load_latest(model_version: str | None = None,
                require_quality: bool = False) -> dict | None:
    """En son eğitilmiş modeli döndürür (hata/boş → None).

    ``model_version`` verilirse yalnız o metodoloji sürümü aranır (kıyas
    kesintilerinde eski sürüm yanlışlıkla yüklenmez). ``require_quality=True``
    ise kalite kapısını geçmiş son model aranır — otomatik işlem etkileyecek
    tüketiciler bunu kullanmalıdır.
    """
    params: dict = {"order": "trained_at.desc", "limit": 1}
    if model_version:
        params["model_version"] = f"eq.{model_version}"
    if require_quality:
        params["quality_passed"] = "is.true"
    try:
        rows = SupabaseREST().select(_TABLE, params)
    except Exception:  # noqa: BLE001 — depo yoksa çağıran onsuz devam eder
        return None
    return rows[0] if rows else None


def list_report_cards(limit: int = 30) -> list[dict]:
    """Karne tarihçesi (artefakt hariç — büyük kolonu indirme)."""
    cols = ("id,model_version,trained_at,trained_until,horizon,universe_size,"
            "n_samples,n_test,auc,brier_calibrated,brier_skill_score,"
            "quality_passed,rejection_reasons")
    try:
        return SupabaseREST().select(
            _TABLE, {"select": cols, "order": "trained_at.desc", "limit": limit})
    except Exception:  # noqa: BLE001
        return []

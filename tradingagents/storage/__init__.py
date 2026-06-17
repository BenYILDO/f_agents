"""Kalıcı veri katmanı — Supabase (Postgres) üzerinden portföy + analiz geçmişi.

Bu paket, Streamlit arayüzünü ve saat başı çalışan zamanlayıcıyı (GitHub Actions)
aynı veritabanına bağlar:

  - :mod:`supabase_client` — PostgREST üzerine ince, bağımlılıksız bir REST
    istemcisi (yalnız ``requests``). Kimlik bilgilerini hem ortam değişkeninden
    hem de Streamlit ``st.secrets``'tan okur.
  - :mod:`portfolio`       — ``holdings`` tablosu: ekle/sil/düzenle + maliyet
    ortalaması ve kâr/zarar (P&L) hesapları.
  - :mod:`snapshots`       — ``analysis_snapshots`` tablosu: saatlik analiz
    geçmişi (yaz/oku); güven katmanı ve geçmiş karşılaştırma buna dayanır.

Şema ``tradingagents/storage/schema.sql`` dosyasındadır; Supabase SQL
editöründe bir kez çalıştırılır (bkz. ``docs/SUPABASE_SETUP.md``).
"""

from tradingagents.storage.supabase_client import (  # noqa: F401
    SupabaseError,
    SupabaseREST,
    is_configured,
)

# Supabase kurulumu — Portföy & saatlik analiz geçmişi

Bu uygulama portföyünü ve saat başı üretilen analiz geçmişini **Supabase**
(ücretsiz Postgres) üzerinde tutar. Üç parça aynı veritabanına bağlanır:

| Parça | Nerede çalışır | Ne yapar |
|-------|----------------|----------|
| Streamlit arayüzü | Streamlit Community Cloud | Portföyü gösterir/düzenler, manuel analiz |
| Zamanlayıcı | GitHub Actions (saatlik cron) | Saat başı deterministik analiz → snapshot yazar |
| Veritabanı | Supabase | `holdings`, `analysis_snapshots` … |

VPS gerekmez; hepsi ücretsiz katmanlarda.

---

## 1) Supabase projesi aç

1. <https://supabase.com> → **New project** (ücretsiz tier yeterli).
2. Bir **Database Password** belirle (saklamana gerek yok, REST anahtarı kullanacağız).
3. Proje hazır olunca **SQL Editor → New query**'ye geç.

## 2) Şemayı çalıştır

Repodaki **`tradingagents/storage/schema.sql`** dosyasının tamamını kopyalayıp
SQL Editor'e yapıştır ve **Run**. Bu; `holdings`, `analysis_snapshots`,
`latest_snapshots` view'i, opsiyonel `ai_runs`/`watchlist` tablolarını ve
saklama (retention) fonksiyonunu kurar. Idempotent'tir — tekrar çalıştırmak güvenli.

## 3) Anahtarları al

**Project Settings → API**:

- **Project URL** → `SUPABASE_URL` (örn. `https://abcd1234.supabase.co`)
- **`service_role` secret** → `SUPABASE_SERVICE_KEY`

> ⚠️ `service_role` anahtarı RLS'i bypass eder ve tüm veriye erişir. **Yalnızca
> sunucu tarafı secret olarak** kullan (Streamlit secrets / GitHub secrets).
> Asla repoya commit'leme, tarayıcıya gönderme.

## 4) Yerelde çalıştırma (opsiyonel)

`.env` dosyana ekle (bkz. `.env.example`):

```
SUPABASE_URL=https://abcd1234.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOi...
```

Sonra: `streamlit run streamlit_app.py` → **💼 Portföyüm** ekranı.

## 5) Streamlit Community Cloud'a deploy

1. <https://share.streamlit.io> → **New app** → bu repo + branch + `streamlit_app.py`.
2. **Advanced settings → Secrets** alanına (TOML formatında) ekle:

   ```toml
   SUPABASE_URL = "https://abcd1234.supabase.co"
   SUPABASE_SERVICE_KEY = "eyJhbGciOi..."
   # AI Analizi ekranını kullanacaksan:
   # OPENAI_API_KEY = "sk-..."
   ```

3. **Güvenlik:** Uygulama açık URL'de portföyünü gösterir. App ayarlarından
   uygulamayı **private** (yalnız davetli) yap. (İleride basit bir parola kapısı
   da ekleyebiliriz.)

## 6) Saatlik zamanlayıcı (GitHub Actions)

Repo **Settings → Secrets and variables → Actions → New repository secret**:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

Workflow (`.github/workflows/hourly-analysis.yml`) saat başı çalışıp
`holdings` + BIST 30 için snapshot yazar. (Bu adım Faz 3'te eklenir.)

> Not: GitHub Actions cron'u yoğunlukta birkaç dakika gecikebilir ve repo 60 gün
> hareketsiz kalırsa zamanlanmış işler durur — repo aktifken sorun olmaz.

## 7) Gecelik model işi (Faz H — opsiyonel ama önerilir)

`.github/workflows/nightly-models.yml` günde bir kez (kapanış sonrası) **kalibre
yukarı-olasılığını** (ağır 5-kat GBDT) hesaplayıp Supabase `model_cache` tablosuna
yazar; saatlik iş bunu okuyup **güven skoruna** katar — böylece saatlik cron ağır
ML çalıştırmaz. Aynı `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` secret'larını kullanır.

Bu özellik için **`schema.sql`'i tekrar çalıştır** (idempotent; `model_cache`
tablosunu ekler). Gecelik iş çalışmasa bile saatlik analiz ve güven skoru
(kalibre olasılık olmadan) sorunsuz çalışır.

**S2 — havuz (pooled) modeli:** aynı gecelik iş, tüm evreni tek panelde eğiten
havuz modelini de eğitir; eğitilmiş artefakt + kalite karnesi `pooled_models`
tablosuna, ticker başına challenger tahmini `model_cache.p_up_pooled` kolonuna
yazılır. Model böylece Supabase'de **kalıcıdır**: gün içi işler yalnız tahmin
yapar, her gün baştan eğitilmez. Bu tablo/kolonlar için de `schema.sql`'i tekrar
çalıştırmak yeterlidir; havuz adımı başarısız olursa gecelik işin kalanı etkilenmez.

---

## Sorun giderme

- **"Supabase yapılandırılmamış"** → `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` eksik
  ya da yanlış yerde (Cloud'da `.env` değil **Secrets** kullanılır).
- **`401`/`403`** → yanlış anahtar; `service_role` anahtarını kullandığından emin ol.
- **`relation "holdings" does not exist`** → `schema.sql` çalıştırılmamış (adım 2).
- **Boş tablo ama hata yok** → RLS açık ve anon anahtarı kullanıyorsun; sunucu
  tarafında `service_role` anahtarına geç.

# 🧪 Paper-Trading Arena — Plan ve Tasarım Dokümanı

> Bu dosya, proje hakkındaki plan ve teknik değerlendirmelerin tarih sırasıyla
> alt alta eklendiği ortak çalışma günlüğüdür. Her katkı, yazar adı ve tarihiyle
> ayrı bir ana başlık altında tutulur.

---

## Claude — 2026-06-29

> **Durum:** Tasarım onaylandı, **kod henüz yazılmadı.** Bu doküman, sohbette
> üzerinde anlaştığımız her şeyi (kararlar + gerekçeler + yapılacaklar + açık
> notlar) tek yerde toplar. Kodlamaya "başla" onayıyla geçilecek.
>
> **Son güncelleme:** 2026-06-29

---

### Tek cümlede

Mevcut deterministik sinyal "beynini" (zaten çalışıyor) **sanal bir kasada otomatik
alıp-satan, kâr/zararı izlenebilen 6 ayrı yönetim tarzının yarıştığı bir arenaya**
bağlamak. Her hesap 100.000 TL ile başlar; hangi tarz kazandırıyor, hangisi
batırıyor — equity eğrileriyle yan yana görülür.

---

### 📌 Ne istedim? (Orijinal talep)

İlk mesajdaki istek, kendi ifademle ve madde madde:

1. Türk borsasında (BIST) çalışan, **yapay zeka destekli bir trade botu** istiyorum.
2. **Belirlediğim bir test bütçesiyle** işlem yapsın.
3. Botun **kendi ürettiği teknik + temel analiz sinyallerine** göre alıp satsın.
4. **Kârını/zararını gözlemleyebileyim.**
5. Geçmiş hareketleri **CSV'de ya da online bir bulut sisteminde** tutalım.
6. **Makine öğrenmesi modelleriyle** kararlar/prediction'lar olsun; biraz daha
   **geçmiş hareketlere bakarak** ilerleyelim.
7. Hatta **RNN/CNN bile** deneyebiliriz — ama elimizde pek veri yok.
8. Bu yapıyı **biraz kurmuştum** ama derdim: *adam akıllı kullanamıyorum, güzel
   sinyaller üretemiyor / sürekli takip edemiyorum, Streamlit tarafı uykuya geçiyor.*
9. **Netlify ya da Vercel'de** deploy edebiliriz; **Supabase** bağlantısını zaten
   kurmuştum.
10. **Önce tartışalım, sonra karar verelim; kodda değişiklik yapmadan öneri sun.**

---

### 🗺️ Planlama günlüğü (sırayla ne kararlaştırdık)

**Adım 0 — Keşif.** Tüm dosyalar tarandı. Projenin iki katmanı (LLM çok-ajan +
deterministik BIST motoru) ve **eksik parça** tespit edildi: sinyal/sinyal-geçmişi
var ama **otomatik alıp-satıp P&L tutan motor yok.** (Ayrıntı: **Neden bu?** bölümü.)

**Adım 1 — Öncelik & bot tarzı.**
- İlk iş → **Paper-trading motoru** (talep 1–4'ün çekirdeği).
- Bot tarzı → **Tam otomatik** (öner-onayla değil).
- Frontend (uyku sorunu) → **şimdilik ertele** (önce motor).

**Adım 2 — Bütçe, evren, risk.**
- Test bütçesi → **100.000 TL** (talep 2).
- Evren → "BIST30 daha garanti ama **BIST100 sinyallerini de değerlendirelim**" +
  senin eklediğin fikir: **birden çok bütçe/yönetim tarzıyla yarıştıralım, hangisi
  kazandırıyor görelim.** → Tek bot yerine **çok-hesaplı arena** doğdu.
- Risk profili → **Dengeli** (varsayılan).

**Adım 3 — Yarışın kuralları.**
- Kasa → **hepsi eşit 100k** (adil yarış; tek değişken = tarz).
- Benchmark → **XU100 al-tut referans çizgisi** (ayrı hesap değil).
- **Senin yeni talebin:** "bir de **fon destekli** alım-satım profili yapalım,
  **fonları da analiz etsin**, planını da çıkaralım." → Kadro **6 hesaba** çıktı,
  fon profili planlandı (bkz. **Fon-destekli profil** bölümü).

**Adım 4 — Fon havuzu & inşa sırası.**
- Fon havuzu → **"Büyüme": hisse senedi + altın/kıymetli maden fonları.**
- İnşa sırası → **önce 5 hisse hesabı (Faz 1), sonra fon hesabı (Faz 2).**

**Adım 5 — Bu doküman.** Tüm kararlar + gerekçeler + yapılacaklar bu dosyada
toplandı; ardından bu "talep + günlük" bölümü en başa eklendi. Sıradaki adım:
"başla" onayıyla **Faz 1 kodlaması** (bkz. **Yapılacaklar** bölümü).

> 💬 **Talep ↔ çözüm eşlemesi (hangi isteğin nerede karşılandığı):**
> talep 1–4 → **Hesap kadrosu + Veri modeli + Karar döngüsü** · talep 5
> (CSV/bulut) → **Veri modeli** (Supabase + CSV indir) · talep 6–7 (ML/RNN) →
> **ML / RNN / CNN duruşu** · talep 8 (takip/uyku) → **Dağıtım + Telegram** ·
> talep 9 (Vercel/Supabase) → **Dağıtım** · talep 10 (önce tartış) → tüm
> planlama günlüğü.

---

### Neden bu? (Mevcut durumun röntgeni)

Proje aslında **iki katmanlı** ve sanılandan çok daha ileride:

- **A) LLM çok-ajan katmanı** (orijinal TradingAgents): Teknik/Duygu/Haber/Temel +
  BIST'e özel Makro-TR ve Siyaset-TR analistleri, boğa/ayı tartışması, trader, risk.
  OpenAI kredisi harcar, dakikalarca sürer. 5 kademeli karar.
- **B) LLM'siz, ücretsiz, deterministik BIST katmanı** (asıl emek burada):
  - 20+ analitik motor (`tradingagents/analytics/`): kompozit teknik skor, Piotroski
    temel skor, rejim HMM, **Deflated Sharpe + PBO**, ½-Kelly boyut, ATR stop/hedef,
    çoklu-zaman teyidi, mum/formasyon, sezonsallık.
  - ML modeli (`tradingagents/ml/`): 16 özellikli, zaman-serisi-CV'li gradient
    boosting + dürüst "skill = doğruluk − taban" metriği.
  - Saat başı cron (`scripts/run_hourly_analysis.py`) + gecelik ağır model işi →
    Supabase'e yazar.
  - Supabase şeması + Streamlit'te 10 ekran.

**Yani "beyin" (sinyal) ve "hafıza" (Supabase) zaten var ve çalışıyor.**

#### Eksik olan tek parça
Kod tarandı: **"test bütçesiyle otomatik alıp-satıp P&L gözlemleme" motoru YOK.**
Şemada yalnız `holdings` (elle girilen **gerçek** pozisyonlar), `analysis_snapshots`
(sinyal geçmişi), `ai_runs`, `watchlist`, `model_cache` var. Sinyaller üretiliyor
ama **hiçbir şey onlara göre otomatik işlem açıp kapatmıyor.** İşte bu motoru
kuruyoruz. Zor kısımların hepsi hazır; eksik olan onları birbirine bağlayan ince
**execution + ledger** katmanı.

> 💬 **Yorum:** `backtrader` zaten bağımlılıklarda duruyor ama kullanılmıyor. Canlı
> paper-trading döngüsü için backtrader'ın event-engine'i şart değil; kendi hafif
> ledger'ımız Supabase ile daha temiz olur. backtrader'ı ileride *tarihsel* backtest
> doğrulaması için değerlendirebiliriz.

---

### Kilitlenen kararlar (özet)

| Konu | Karar |
|------|-------|
| Bot tarzı | **Tam otomatik** (öner-onayla değil): cron kendi kasasında otomatik al/sat; kullanıcı sadece P&L/equity izler |
| Hesap sayısı | **6** (5 hisse + 1 fon) |
| Başlangıç kasası | **Hepsi eşit 100.000 TL** (adil yarış; tek değişken = yönetim tarzı) |
| Benchmark | **XU100 al-tut** = ayrı hesap değil, referans çizgisi |
| Veri saklama | **Supabase** (bulut) + UI'da **CSV indir** (ikisi de) |
| Risk varsayılanı | "Dengeli" profili |
| İnşa sırası | **Faz 1: 5 hisse hesabı → Faz 2: fon hesabı** |
| Fon havuzu | **"Büyüme": hisse senedi fonları + kıymetli maden/altın fonları** |
| RNN/CNN | **Reddedildi** (gerekçe: **ML / RNN / CNN duruşu**) |
| Frontend (uyku sorunu) | **Şimdilik ertelendi** (gerekçe: **Dağıtım**) |

---

### Hesap kadrosu (6 profil)

Hepsi 100.000 TL ile başlar. Üstlerinde XU100 al-tut referans çizgisi.

| # | Profil | Varlık evreni | Giriş eşiği | Maks poz / tavan | Rejim filtresi | Ayırt edici özellik |
|---|--------|---------------|-------------|------------------|----------------|---------------------|
| 1 | 🛡️ **Temkinli** | BIST30 | güven ≥ 68 | 15 / %10 | açık | Sermaye koruma, sıkı stop, çok küçük pozisyon |
| 2 | ⚖️ **Dengeli** | BIST30 | güven ≥ 60 | 8 / %20 | açık | Varsayılan; ½-Kelly boyut |
| 3 | 🔥 **Agresif** | BIST30 **+ BIST100** | güven ≥ 55 | 4 / %35 | **kapalı** | Yoğun, dip alır, ayı piyasada da girer |
| 4 | 🤖 **ML-öncelikli** | BIST30 **+ BIST100** | p_up ≥ 0.58 | 8 / %20 | açık | Kararı gecelik **kalibre model** verir (kural değil) |
| 5 | 📈 **Trend-takip** | BIST30 | güven ≥ 60 + boğa | 6 / %20 | **zorunlu boğa** | Sadece boğa + çoklu-zaman konfluens, trailing stop |
| 6 | 🏦 **Fon-destekli** | TEFAS (hisse + altın fonları) | momentum sırası | — | — | Fon momentum rotasyonu (bkz. **Fon-destekli profil**) |

**Bu kadronun ölçtüğü sorular:**
- Hangi *yönetim tarzı* daha iyi? (1–5 arası)
- Evreni BIST30'dan **BIST100'e** genişletmek kazandırıyor mu, batırıyor mu? (3,4 vs 1,2,5)
- **ML** kural-temelli tarzları yenebiliyor mu? (4 vs diğerleri)
- **Aktif tarzlar** pasif XU100 al-tut'u geçebiliyor mu? (hepsi vs referans çizgisi)
- **Fonlara** rotasyon, doğrudan hisse seçmekten iyi mi? (6 vs 1–5)

---

### Veri modeli (yeni Supabase tabloları)

```
paper_accounts
  id, ad, profil_kodu, başlangıç_kasa, güncel_kasa,
  kurallar (jsonb: evren, güven_eşiği, maks_poz, poz_tavan, rejim_filtresi,
            stop_çarpanı, komisyon, sizing_yöntemi …),
  created_at

paper_trades            ← "CSV"nin kaynağı
  id, account_id, ts, ticker, yön (AL/SAT), adet, fiyat, komisyon,
  gerçekleşen_pnl, gerekçe (jsonb: o anki sinyal snapshot'ı — neden alındı/satıldı)

paper_equity            ← equity curve
  id, account_id, ts, kasa, pozisyon_değeri, toplam_equity, xu100_değeri
```

- `paper_trades` = **tek gerçek kaynak** (event-sourcing). Açık pozisyonlar buradan
  türetilir; istenirse hız için ayrı `paper_positions` materyalize görünümü eklenir.
- Çoklu hesap = `kurallar` jsonb ile parametrelenir → **6 hesap = 6 satır.** Cron tek
  hesap yerine hepsinin üzerinde döner. **Mimari büyümez.**
- RLS, mevcut tablolardaki gibi (service_role ile yazılır).

> 💬 **Yorum:** `kurallar`'ı jsonb tutmak, ileride 7., 8. profil eklemeyi (veya
> mevcut eşikleri değiştirmeyi) kod değişikliği olmadan, sadece satır ekleyerek
> mümkün kılar.

---

### Karar döngüsü (motor mantığı)

#### Cadence (önemli teknik karar)
- **İşlem kararı: günde 1 kez**, seans kapanışından sonra (YENİ günlük cron).
- **Mark-to-market equity: saatlik** (mevcut hourly cron'a hafif adım) → eğri pürüzsüz.

> ⚠️ **Neden günde 1 işlem?** Mevcut saatlik cron aynı **günlük bar'ı** gün boyu
> tekrar okur. İşlem kararını saatlik yaparsak bot aynı sinyalde gün içinde defalarca
> alıp satar (flip-flop, sahte P&L). Bu yüzden işlem günde 1'e sabitlenir; bunu
> Faz 1'de baştan doğru kuruyorum.

#### Akış (her hesap için, günde 1)
```
1. Hesabı yükle (kasa + açık pozisyonlar)
2. Rejim (XU100 HMM) + makro-şok (USDTRY stres) bir kez hesapla
3. AÇIK POZİSYONLAR → ÇIKIŞ:
     gated ∈ {SAT, KAÇIN}  ya da  fiyat ≤ ATR-stop  ya da  fiyat ≥ hedef
     (Trend-takip için: trailing stop)
     → SAT: gerçekleşen P&L + komisyon yaz
4. ADAY HİSSELER (elde olmayan, hesabın evreninden) → GİRİŞ:
     gated ∈ {AL, GÜÇLÜ AL}  ve  güven ≥ eşik  ve  likit (Amihud filtresi)
     ve  rejim/makro uygun  (profil kuralına göre)
     → ½-Kelly × vol-hedef ile boyutla, poz_tavan ile sınırla
     → kasa yetiyorsa AL
5. Pozisyon + kasayı güncelle, işlemleri yaz
6. Mark-to-market: equity snapshot + XU100 paralel "al-tut" değeri yaz
7. (Faz 1 sonu) Telegram özeti gönder
```

#### Gerçekçilik (P&L'in masal olmaması için — kritik)
- **Komisyon + kayma:** ~%0.05/işlem + küçük slippage (BIST'e yakın). Yoksa kâr şişer.
- **Likidite:** mevcut Amihud `illiquid` bayrağıyla likit olmayanı **alma**.
- **Lot:** BIST tam lot → adet aşağı yuvarlanır.
- **Fiyat (fill):** v1'de karar günü kapanışı (komisyon+kayma telafi eder).
  v2'de daha temkinli: ertesi seans açılışı (look-ahead'i sıfırlar).
- **Benchmark:** aynı bütçeyle XU100 al-tut paralel → botun **alpha** üretip
  üretmediği görülür.

---

### Fon-destekli profil (Faz 2) — ayrı plan

Fonlar hisseden farklı bir hayvan: gün içi fiyat yok, **günde tek NAV**, takas
valörlü, ve hisse göstergeleri (RSI/MACD/mum/temel rasyo) fona uymaz. **yfinance'te
yok** → yeni veri hattı gerekir.

#### Veri hattı — TEFAS
- Kaynak: `tefas.gov.tr` history endpoint'i (fon NAV geçmişi + portföy dağılımı).
  Hazır `tefas-crawler` paketi de değerlendirilebilir.
- KAP'ın aksine TEFAS genelde **agresif WAF'sız**, erişilebilir. Yine de mevcut
  dataflow desenindeki gibi **graceful-degrade** (erişilemezse profil "veri yok" der,
  sistemi kırmaz).

#### Fon analiz motoru (hisse motorundan farklı)
- NAV serisinden: 1a/3a/6a/12a getiri, yıllık vol, **rolling Sharpe**, max düşüş
- **Kategori-içi göreli sıra** (percentile rank) — "kendi ligindeki yeri"
- **Momentum kalıcılığı** ("hot hand"; fonlarda iyi belgelenmiş) — son 3–6 ay getirisi
  gelecek getiriyi öngörür
- Trend: NAV > MA200; (varsa) gider oranı, fon büyüklüğü

#### Karar — momentum rotasyonu
- Her kategoride fonları risk-ayarlı momentuma göre sırala → en üst N fona gir
  (vol-hedef ağırlık)
- **Aylık / iki haftada bir** rebalance (fonlarda günlük dönüş anlamsız + maliyetli)
- Sıra düşünce / trend kırılınca çık

#### Gerçekçilik
- Fiyat = gün sonu NAV · takas T+0…T+2 (kategoriye göre) · çoğu kurucu fonu
  komisyonsuz (düşük sürtünme); bazısında erken-çıkış kesintisi → basit model
- Diğer 5 hesapla **aynı** equity/işlem/CSV altyapısını paylaşır; yalnız "varlık=fon,
  fiyat=NAV, analiz=fon motoru" değişir.

#### Fon havuzu (seçilen)
**"Büyüme": hisse senedi fonları + kıymetli maden/altın fonları.** (Yüksek getiri
potansiyeli + hisse hesaplarıyla anlamlı kıyas.)

---

### ML / RNN / CNN duruşu (dürüst not)

- BIST'te bir hissenin 10 yıllık **günlük** verisi ≈ 2.500 bar. RNN/LSTM/CNN bu
  veriyle **kesinlikle ezberler (overfit)**, sinyali iyileştirmez — **bozar.**
  → **RNN/CNN şimdilik reddedildi.**
- Zaten elindeki **GBDT + zaman-serisi backtest + skill metriği doğru seçim.**
- Veri kıtlığında sinyali gerçekten iyileştiren yol:
  1. **Havuzlanmış (pooled) model** — tüm BIST hisselerini tek modelde eğit
     (örnek 2.5K → ~75K). Deep learning'e ancak o zaman yaklaşılır.
  2. **Döngüyü kapatmak** — tahmin yap → gerçekleşen sonucu ölç → skill'i *canlı*
     izle. (Bu zaten paper-trading motoruyla geliyor.)
  3. İndikatör eklemek değil — elindeki **DSR/PBO** ile mevcut edge'i **ölçmek.**

---

### Dağıtım / "uykuya geçiyor" gerçeği

- **Streamlit, Netlify/Vercel'e deploy EDİLEMEZ** (uzun-süren Python sunucusu; o
  platformlar statik/serverless JS içindir).
- **İyi haber:** "beyin" (saat başı analiz) zaten GitHub Actions cron + Supabase'e
  taşınmış → **veri hattı uyumuyor**, sadece *arayüz* uyuyor. UI uyurken bile cron
  çalışıp veri biriktirir.
- Seçenekler (sonraya bırakıldı):
  - (a) Streamlit'i uyumayan host'a taşı (Render/Railway/Fly.io/HF Spaces) →
    **sıfır kod değişikliği.**
  - (b) UI'ı Next.js/React olarak yeniden yaz, Vercel'de koş, Supabase'i doğrudan oku
    → hiç uyumaz, mobil-dostu ama **yeniden yazım.**
  - (c) Streamlit'te kal, uykuyu kabul et; takibi **Telegram** ile çöz.

---

### "Takip edemiyorum" → Telegram

Saat başı cron zaten çalışıyor; sonuna ~10 satırlık adım → Telegram'a günlük özet:
"Bugün: GARAN → AL (güven 72) · Dengeli hesap +%3.2 (hafta) · XU100 +%1.1".
Bedava, anında, cebte. UI'ı açmaya gerek kalmadan takip. **Faz 1'in sonunda eklenir.**

---

### Yapılacaklar (faz planı + adımlar)

#### Faz 1 — 5 hisse hesaplı arena (her şey hazır, hızlı)
1. **Şema:** `paper_accounts`, `paper_trades`, `paper_equity` tabloları
   (`storage/schema.sql`'e idempotent ekleme) + RLS.
2. **Storage katmanı:** `storage/paper.py` — hesap/işlem/equity okuma-yazma
   (mevcut `supabase_client` REST deseniyle).
3. **Execution motoru:** `paper/engine.py` — **Karar döngüsü** akışı; profil kuralını `kurallar`
   jsonb'den okur; `analysis_run.analyze_ticker` çıktısını karara çevirir.
4. **Sizing/komisyon/lot/likidite** yardımcıları (mevcut `analytics.sizing` +
   `risk` yeniden kullanılır).
5. **Seed:** 5 hesabı (Temkinli/Dengeli/Agresif/ML/Trend) 100k ile oluşturan tek
   seferlik script.
6. **Günlük cron:** `scripts/run_paper_trading.py` + `.github/workflows/paper-trading.yml`
   (seans sonrası). Saatlik cron'a **mark-to-market** adımı.
7. **UI:** `app_pages/paper_arena.py` — lig tablosu (6 equity + XU100), hesap detayı
   (kasa/pozisyon/canlı P&L), işlem günlüğü + **CSV indir**, özet istatistik
   (toplam getiri, CAGR, max düşüş, Sharpe, kazanma oranı, işlem sayısı).
   `streamlit_app.py`'ye "🧪 Paper Bot" sekmesi.
8. **Telegram:** cron sonu özet (opsiyonel secret; yoksa sessiz geç).
9. **Testler:** motor kararları, komisyon/lot, equity hesabı, çıkış kuralları
   (mevcut `tests/` desenine).

#### Faz 2 — Fon hesabı
10. **TEFAS dataflow:** `dataflows/tefas.py` (graceful-degrade).
11. **Fon analiz motoru:** `analytics/fund_momentum.py` (bkz. **Fon analiz motoru**).
12. **6. hesap + fon karar dalı** + UI'da fon kolonları.

---

### Açık notlar / kararını beklediklerim

- **Telegram'ı Faz 1'e dahil edeyim mi**, yoksa motor çalışınca mı ekleyelim?
- Tabloları kurmadan önce **şemayı sana göstereyim mi** (onaylı SQL)?
- **Fill konvansiyonu:** v1 "kapanıştan doldur" (basit) ile başlayalım, v2'de "ertesi
  açılış"a geçelim — onay?
- **Komisyon oranını** %0.05 sabit mi tutalım, yoksa profillere göre mi değişsin?

---

### Kapsam dışı (şimdilik)

- Gerçek para / gerçek emir (bu **paper/sanal**; gerçek broker entegrasyonu yok).
- RNN/CNN/LSTM (bkz. **ML / RNN / CNN duruşu**).
- Frontend taşıma kararı (bkz. **Dağıtım**).
- Ham KAP entegrasyonu (WAF engeli; ayrı konu — bkz. BIST adaptasyon notları).

---

> ⚠️ **Yasal:** Bu sistem araştırma/eğitim amaçlıdır; yatırım tavsiyesi değildir.
> Paper-trading sonuçları gerçek getiriyi garanti etmez.

---

## Codex — 2026-06-29 00:37 (TR)

> **Yazar:** Codex
> **İnceleme durumu:** Mevcut planın ana ürün fikri uygun; aşağıdaki maddeler
> uygulanmadan execution motorunun kodlanmasına başlanması önerilmez. Bu bölüm,
> önceki planı silmez veya geçersiz kılmaz; finansal doğruluk, deney adaleti ve
> operasyon güvenliği için onu tamamlar.

### Genel değerlendirme

Planın korunması gereken güçlü tarafları:

- Paper-trading motorunun ilk öncelik olması.
- Gerçek emir yerine sanal bütçeyle başlanması.
- İşlem kararının günlük, mark-to-market hesabının daha sık yapılması.
- Supabase'in kalıcı ana veri kaynağı, CSV'nin dışa aktarma formatı olması.
- Birden fazla yönetim profilinin aynı başlangıç bütçesiyle yarışması.
- Fonların hisse motoruna zorla uydurulmayıp ayrı fazda ele alınması.
- RNN/CNN yerine önce mevcut klasik ML yaklaşımının doğrulanması.
- UI uykusundan bağımsız cron ve Telegram bildirim akışı kurulması.

Revizyon gerektiren temel nokta şudur: execution ve ledger katmanı yalnızca
"sinyali alım/satıma bağlayan ince bir parça" değildir. Emir zamanı, fill fiyatı,
nakit muhasebesi, tekrar çalışan cron'un aynı emri iki kez üretmemesi ve kullanılan
verinin o anda gerçekten biliniyor olması, ölçülen P&L'in güvenilirliğini belirleyen
ana sistemdir.

### Kodlamadan önce tamamlanacak Faz 0 — sinyal doğruluğu

Arena, mevcut sinyal motorunun çıktısını otomatik işleme çevireceği için önce bu
çıktının bütün çağrı yollarında aynı ve güvenilir olması gerekir.

1. **ML olasılığının yön etkisi düzeltilmeli.** Mevcut güven hesabında
   `abs(p_up - 0.5)` kullanıldığı için model ana kararla ters yönde güçlü bir
   olasılık verdiğinde bile güven artabilir. AL kararı için yüksek `p_up` olumlu,
   düşük `p_up` olumsuz; SAT kararı için bunun tersi olmalıdır.
2. **Model kalite kapısı eklenmeli.** Yalnız `p_up >= 0.58` yeterli değildir.
   Modelin kullanılabilmesi için en az şu koşullar değerlendirilmelidir:
   - model önbelleği güncel olmalı,
   - örnek-dışı skill pozitif olmalı,
   - AUC belirlenen asgari seviyenin üzerinde olmalı,
   - Brier skoru naif tabandan iyi olmalı,
   - minimum eğitim/OOS örnek sayısı sağlanmalı.
3. **Kalitesiz model davranışı açık olmalı.** Kalite kapısını geçemeyen ML hesabı
   rastgele olasılıkla işlem açmamalı; nakitte beklemeli veya önceden tanımlanmış
   bir kural-temelli fallback kullanmalıdır. Tercih edilen ilk sürüm: nakitte bekle.
4. **Makro şok bütün akışlara bağlanmalı.** Saatlik cron, manuel tarama ve günlük
   paper motoru aynı `analyze_universe()`/ortak orchestrator yolunu kullanmalı.
5. **Tek EOD analiz snapshot'ı üretilmeli.** Evren kapanış sonrası yalnız bir kez
   analiz edilmeli; tüm hesaplar aynı dondurulmuş sonuçları tüketmelidir. Her profil
   için yfinance/analiz tekrar çalıştırılırsa hem API sonuçları hem zaman damgaları
   farklılaşabilir ve yarış adaleti bozulur.
6. **Sinyal sürümü saklanmalı.** Her snapshot; `strategy_version`, `model_version`,
   `signal_asof`, kullanılan son bar zamanı ve veri sağlık durumunu taşımalıdır.
7. **ML hedefi execution ile hizalanmalı.** Mevcut 10 günlük yukarı/aşağı etiketi,
   günlük stop/hedef kullanan stratejiyle aynı hedef olmayabilir. Etiket; T+1 fill,
   seçilen tutma ufku, XU100'e göre relatif getiri ve işlem maliyetiyle yeniden
   tanımlanmalıdır.

**Faz 0 çıkış kriteri:** Aynı veri snapshot'ı ve aynı profil ayarlarıyla tekrar
çalıştırıldığında aynı planlanan emirler oluşmalı; kalitesiz veya bayat model
hiçbir otomatik işleme katkı vermemelidir.

### Fill konvansiyonu — ilk sürümden itibaren look-ahead'siz

Karar kapanış sonrası üretildiği için aynı günün kapanış fiyatından fill yapmak
look-ahead oluşturur. Komisyon veya sabit slippage bu problemi ortadan kaldırmaz.
V1'den itibaren aşağıdaki akış kullanılmalıdır:

```
T günü kapanış → sinyal ve hedef portföy → PENDING emir
T+1 ilk geçerli BIST seansı → açılış fiyatı + yönlü slippage → fill
```

- AL emrinde fill fiyatı açılışın biraz üzerinde, SAT emrinde biraz altında
  modellenir.
- T+1 verisi yoksa emir doldurulmaz; `PENDING` kalır veya belirlenen sürede iptal
  edilir.
- Açılışta büyük gap varsa teorik stop fiyatından fill varsayılmaz; elde edilebilir
  ilk fiyat kullanılır.
- Sinyal üretildiği an bilinen veri ile fill anında bilinen veri ayrı saklanır.
- BIST tatilleri ve hafta sonları için "ertesi takvim günü" değil, "ertesi geçerli
  seans" kullanılır.

Günlük OHLC ile stop/hedef simülasyonu:

- `Open < stop` ise uzun pozisyon stop fiyatından değil açılıştan kapatılır.
- Gün içinde `Low <= stop` ise stop tetiklenmiş kabul edilir.
- Gün içinde `High >= target` ise hedef tetiklenmiş kabul edilir.
- Aynı günlük barda hem stop hem hedef görülür ve sıralama bilinmezse muhafazakâr
  varsayım olarak stop önce uygulanır; sonuç ayrıca `ambiguous_bar=true` ile
  işaretlenir.
- Stop/hedef kontrolü yalnız kapanış fiyatıyla yapılmamalıdır.

### Revize Supabase veri modeli

İlk plandaki üç tablo kavramsal olarak doğru başlangıçtır ancak pending emirleri,
cron tekrarlarını ve nakit hareketlerini güvenle temsil etmek için genişletilmelidir.

```
arena_seasons
  id, name, status, starting_at, ending_at, base_currency,
  initial_capital, execution_config, benchmark_config, created_at

paper_accounts
  id, season_id, name, profile_code, status,
  initial_cash, strategy_version, rules_snapshot, created_at

paper_runs
  id, season_id, run_type, signal_asof, status,
  started_at, completed_at, error_summary

paper_orders
  id, run_id, account_id, ticker, asset_type, side,
  quantity, order_type, status, signal_asof, scheduled_session,
  reason_snapshot, strategy_version, created_at

paper_fills
  id, order_id, account_id, filled_at, quantity,
  reference_price, fill_price, commission, slippage, created_at

paper_cash_ledger
  id, account_id, fill_id, ts, event_type, amount,
  balance_after, metadata

paper_positions
  account_id, ticker, asset_type, quantity, avg_cost,
  realized_pnl, updated_at

paper_equity
  id, account_id, asof_ts, cash, unsettled_cash,
  positions_value, total_equity, benchmark_value,
  stale_price_count, created_at

paper_predictions
  id, run_id, ticker, signal_asof, model_version,
  p_up, skill, auc, brier, n_samples, quality_passed, metadata
```

Tasarım kuralları:

- `paper_fills` gerçekleşmiş işlemlerin değişmez kaynağıdır.
- `paper_orders` karar ile gerçekleşme arasındaki bekleme durumunu saklar.
- `paper_positions` fill/ledger'dan türetilebilen ama UI ve günlük motor için
  materyalize edilen hızlı görünümdür.
- Para hesaplarında `float` yerine Postgres `numeric`/Python `Decimal` tercih edilir.
- Hisse adedi tam lot için integer; fon birimi gerekiyorsa numeric tutulur.
- `rules_snapshot` ve `strategy_version` değişmez olmalıdır. Bir profil kuralı
  değiştirilince eski sezon geriye dönük değişmemeli; yeni profil sürümü veya yeni
  sezon açılmalıdır.
- `paper_equity` için `(account_id, asof_ts)`; emir için en az
  `(account_id, signal_asof, ticker, side, strategy_version)` benzeri benzersiz
  kısıtlar bulunmalıdır.
- Aynı cron yeniden çalıştığında yeni emir üretmek yerine mevcut `run_id`/emri
  bulmalıdır.
- Emir fill'i, nakit hareketi ve pozisyon güncellemesi mümkünse tek Postgres RPC
  transaction'ı içinde atomik uygulanmalıdır. REST üzerinden üç bağımsız yazım,
  yarım kalmış hesap oluşturabilir.

### Arena ve bilimsel karşılaştırma ayrılmalı

Mevcut beş profil ürün açısından anlaşılır ve izlenmesi keyiflidir; ancak profiller
aynı anda evren, eşik, sizing, rejim filtresi ve yoğunlaşmayı değiştirdiğinden tek
bir faktörün katkısını ölçemez.

Bu nedenle UI'da iki sonuç grubu önerilir:

1. **Arena/lig:** Temkinli, Dengeli, Agresif, ML-öncelikli ve Trend-takip mevcut
   karakterleriyle yarışır. Soru: "Hangi toplam yönetim tarzı daha iyi?"
2. **Laboratuvar/ablation:** Her karşılaştırmada yalnız bir değişken farklıdır.
   Sorular:
   - Aynı BIST30, aynı risk ve aynı execution altında ML mi kural motoru mu?
   - Aynı strateji altında BIST30 mu BIST100 mü?
   - Aynı strateji altında rejim filtresi açık mı kapalı mı?
   - Aynı giriş/çıkış altında eşit ağırlık mı çeyrek Kelly mi?

ML ve kural hesabı kıyaslanırken evren, maksimum pozisyon, pozisyon tavanı,
komisyon, slippage, stop ve rejim kuralı aynı olmalıdır. Aksi halde sonuç "ML edge'i"
olarak yorumlanmamalıdır.

### Profil kurallarındaki düzeltmeler

- Mevcut kod varsayılan olarak **çeyrek Kelly (`0.25`)** kullanır; plandaki
  "½-Kelly" ifadeleri ya çeyrek Kelly olarak düzeltilmeli ya da ayrı bir deney
  profili tanımlanmalıdır. İlk tercih: çeyrek Kelly.
- Kelly yalnız kalite kapısını geçmiş ve kalibre edilmiş olasılıkla kullanılmalı;
  geçerli olasılık yoksa sabit risk bütçesi tercih edilmelidir.
- Agresif profilin rejim filtresini kapatması deney amacıyla mümkündür; fakat bayat
  veri, illikidite, veri hatası ve maksimum hesap drawdown'u gibi sistemsel kill
  switch'leri hiçbir profil kapatamamalıdır.
- ML profilinin yalnız giriş değil çıkış kuralı da tanımlanmalıdır. Örneğin model
  kalite kaybı, `p_up` düşüşü, ters sinyal, stop veya maksimum tutma süresi.
- Pozisyonların yalnız açılıp kapatılacağı mı, hedef ağırlığa günlük yeniden
  dengeleneceği mi açıkça yazılmalıdır. V1 için daha az turnover yaratan
  "girişte boyutla, yalnız çıkışta kapat" yaklaşımı önerilir.
- Portföy seviyesinde maksimum sektör ağırlığı, toplam açık risk, günlük turnover,
  aynı yönde yüksek korelasyonlu pozisyon ve minimum nakit rezervi sınırları
  bulunmalıdır.
- Kasa bütün adaylara yetmiyorsa ticker sırası kullanılmamalıdır. Önce tüm adaylar
  ortak skorla sıralanmalı, sonra portföy bütçesi topluca dağıtılmalıdır.

### Ortak execution maliyeti

Komisyon ve fill varsayımları profil özelliği değil, deney ortamının ortak fizik
kuralları olmalıdır. Aksi halde hesaplar adil kıyaslanamaz.

- Komisyon, bütün hesaplarda aynı **tek-yön baz puan (bps)** değeri olarak
  konfigüre edilir.
- Slippage sabit bir değerle başlayabilir; daha sonra order tutarı / ortalama günlük
  işlem hacmi oranına göre artırılabilir.
- Her fill satırında referans fiyat, fill fiyatı, komisyon ve slippage ayrı saklanır.
- BIST100'deki daha düşük likidite nedeniyle aynı slippage oranını körlemesine
  kullanmak agresif/ML hesabının performansını olduğundan iyi gösterebilir.
- Settlement basitleştirilecekse "satış geliri aynı gün yeniden kullanılabilir"
  varsayımı açıkça yazılmalıdır. Daha gerçekçi sürümde `unsettled_cash` ayrı tutulur.

Başlangıç komisyon değeri plan içinde sabit gerçekmiş gibi yazılmamalı; broker ve
döneme göre değişebileceği için ortak konfigürasyon olarak bırakılmalıdır.

### Veri, evren ve benchmark sözleşmesi

- BIST30/BIST100 evreninin kaynağı, güncellenme zamanı ve kullanılan liste sürümü
  saklanmalıdır.
- Canlı arena güncel üyelerle çalışabilir; tarihsel replay/backtest için bugünkü
  BIST100 listesini geçmişe uygulamak survivorship bias oluşturur. Mümkünse dönemsel
  endeks üyelikleri kullanılmalı; bulunamazsa bu sınırlama sonuçlarda gösterilmelidir.
- Temel veriler günlük fiyat gibi her gün yeniden yorumlanmamalı; finansal raporun
  yayımlandığı zaman bilinerek as-of snapshot tutulmalıdır.
- Split/bedelsiz, temettü ve sembol değişiklikleri için politika tanımlanmalıdır.
- Portföylere temettü yazılacaksa benchmark da total-return mantığıyla
  karşılaştırılmalıdır. Temettü yok sayılacaksa bunun aktif strateji ve benchmark
  için etkisi açıkça raporlanmalıdır.
- XU100 benchmark aynı başlangıç bütçesi ve aynı T+1 fill konvansiyonuyla başlatılır;
  benchmark birimi sezon boyunca sabit tutulur.
- Eksik veya bayat fiyatla equity hesaplanırsa değer sessizce sıfırlanmamalı;
  son güvenilir fiyat ve `stale_price_count` ile uyarı verilmelidir.

### Günlük motorun revize akışı

```
1. Benzersiz paper_run oluştur / mevcut idempotent run'ı bul
2. Seans ve veri tazeliğini doğrula
3. BIST evrenini bir kez çek ve dondur
4. Rejim + makro şoku bir kez hesapla
5. Tüm ticker'ları bir kez analiz et; EOD signal snapshot/prediction yaz
6. Önce mevcut pozisyonlar için çıkış ve risk emirlerini üret
7. Her profil için uygun adayları filtrele
8. Adayları ortak profil skoruyla sırala
9. Portföy seviyesi risk/yoğunlaşma kontrollerini uygula
10. T+1 seansına PENDING emirleri yaz
11. T+1 fill işi: fiyat + slippage + komisyon ile emirleri atomik gerçekleştir
12. EOD mark-to-market ve benchmark equity yaz
13. Run'ı completed/partial/failed olarak kapat
14. Yalnız tamamlanmış sonuçlardan Telegram özeti üret
```

Saatlik mark-to-market ana karar motorundan ayrılmalıdır. Saatlik iş yeni giriş
kararı üretmez; yalnız güvenilir fiyatlarla açık pozisyon ve equity görünümünü
günceller. Günlük mum tamamlanmadan oluşan ara değerler model eğitim verisine veya
"bağımsız sinyal örneği" sayımına eklenmez.

### Risk ve kill switch'ler

Her profilin stratejik risk ayarlarından bağımsız, bütün arena için kapatılamayan
sistem güvenlik kuralları bulunmalıdır:

- Bayat/eksik fiyat veya analiz verisinde yeni emir yok.
- Model cache bayatsa ML emri yok.
- Aynı hesap/ticker için çakışan eşzamanlı emir yok.
- Negatif kasa veya elde olandan fazla satış yasak.
- Maksimum hesap drawdown'u aşılırsa hesap `PAUSED`.
- Günlük zarar/turnover sınırı aşılırsa yeni giriş durur.
- Supabase yazımının bir kısmı başarısız olursa run tamamlanmış sayılmaz.
- Manuel global `PAUSE_NEW_ORDERS` anahtarı bulunur.
- Kaynak fiyat makul olmayan ölçüde sıçrarsa corporate-action/veri-hatası kontrolü
  yapılmadan fill uygulanmaz.

Kill switch eşikleri ürün kararıdır; ilk kodlamadan önce hesap kurallarından ayrı
bir `execution_config` altında belirlenmelidir.

### Fon profili için ek koşullar

Fon motorunun ayrı faz olması korunmalıdır. Ek olarak:

- Fon hesabı daha geç başlarsa mevcut ligle doğrudan kıyaslanmamalıdır. Fon hazır
  olduğunda yeni arena sezonu başlatılmalı veya bütün profiller aynı tarihten replay
  edilmelidir.
- TEFAS erişimi için cache, retry/backoff, rate-limit, veri kullanım koşulları ve
  kaynak atfı planlanmalıdır; endpoint'in sürekli erişilebilir olduğu varsayılmamalıdır.
- Emir tarihi, açıklanan NAV tarihi, yatırım talimatı cutoff zamanı ve gerçekleşme
  valörü ayrı alanlar olmalıdır.
- Fonlarda birimlerin kesirli olabilmesi, stop yerine periyodik rebalance ve bazı
  fonlarda olası kesinti/valör farkı execution modeline yansıtılmalıdır.
- Hisse hesapları günlük karar verirken fon hesabının iki haftalık/aylık rotasyonu
  aynı tabloda gösterilebilir; fakat işlem sayısı ve kısa dönem Sharpe gibi metrikler
  doğrudan eşdeğer kabul edilmemelidir.

### Revize uygulama fazları

#### Faz 0 — Sinyal ve veri sözleşmesi

1. `p_up` yön etkisini düzelt.
2. ML kalite kapısını ve model tazeliğini ekle.
3. Makro şoku bütün analiz yollarına bağla.
4. Tek EOD analiz/snapshot akışını oluştur.
5. Strateji/model/evren sürümlemesini tanımla.

#### Faz 1A — Saf ve tekrar kullanılabilir simülatör

6. Supabase veya Streamlit'e bağlı olmayan execution çekirdeğini yaz.
7. T+1 fill, gap, stop/hedef, komisyon, slippage, lot ve nakit kurallarını uygula.
8. Aynı motoru hem historical replay hem canlı paper işleminde kullan.

#### Faz 1B — Şema ve atomik ledger

9. Nihai SQL şemasını uygulamadan önce ayrıca incele/onayla.
10. Season/run/order/fill/cash/position/equity tablolarını ekle.
11. Benzersiz kısıtlar ve atomik fill RPC'sini yaz.
12. Beş hesabı idempotent biçimde seed et.

#### Faz 1C — Historical replay ve doğrulama

13. Tam arena motorunu geçmiş günlük barlarda çalıştır.
14. Komisyon/slippage dahil equity, drawdown, turnover ve alpha hesapla.
15. Look-ahead, duplicate emir ve veri bayatlığı testlerini geçir.
16. Profil ligi ile tek-değişkenli laboratuvar sonuçlarını ayrı raporla.

#### Faz 1D — Shadow mode

17. Birkaç geçerli BIST seansı boyunca sinyal ve PENDING emir üret.
18. Gerçek fill/P&L uygulamadan ertesi gün fiyat ve zamanlama doğruluğunu denetle.
19. Cron retry, yarım kalan run ve eksik veri senaryolarını gözlemle.

#### Faz 1E — Canlı paper arena

20. Beş hisse hesabında fill ve ledger'ı etkinleştir.
21. UI lig tablosu, hesap detayı, emir/fill günlüğü ve CSV indirmeyi ekle.
22. Telegram'ı motor kararlı olduktan sonra, Faz 1'in son adımı olarak ekle.

#### Faz 2 — Fon motoru

23. TEFAS veri sözleşmesi ve cache katmanı.
24. Fon momentum/rotasyon analitiği.
25. Fon execution/valör simülasyonu.
26. Yeni sezon veya ortak başlangıçlı replay ile altıncı hesabı aktive et.

#### Faz 3 — Gelişmiş ML deneyleri

27. BIST hisselerinden pooled panel veri seti oluştur.
28. Lojistik regresyon ve mevcut GBDT'yi champion/challenger olarak karşılaştır.
29. Relatif ve maliyet-sonrası hedeflerle purged walk-forward doğrulama yap.
30. Klasik modeller kalıcı OOS edge gösterirse 1D-CNN/TCN/GRU deneylerini ayrı
    araştırma kolunda değerlendir; canlı arena modelini doğrudan değiştirme.

### Zorunlu test matrisi

Mevcut test listesine ek olarak en az aşağıdaki senaryolar bulunmalıdır:

- Sinyal günü aynı kapanıştan fill oluşmaması.
- Hafta sonu/tatilde emrin ilk geçerli seansa taşınması.
- Stop altında gap açılışı.
- Aynı barda hem stop hem hedef görülmesi.
- Cron'un aynı `run_id` ile iki kez çalışması ve duplicate emir üretmemesi.
- Fill yazıldıktan sonra nakit/pozisyon adımında hata oluşması; transaction rollback.
- Kasa yetersizken aday sırasının deterministik olması.
- Negatif kasa ve fazla satışın engellenmesi.
- Bayat fiyat/model/veri durumunda yeni giriş yapılmaması.
- ML kalite kapısını geçmeyen ticker'ın işlem üretmemesi.
- Beş profilin aynı dondurulmuş signal snapshot'ını kullanması.
- Komisyon ve slippage'ın alış/satışta doğru yönde uygulanması.
- Benchmark başlangıç birimi ve equity hesabı.
- Split/temettü veya olağandışı fiyat sıçramasının sessizce P&L yaratmaması.
- Partial/failed run'ın Telegram'da başarılı gösterilmemesi.
- Sezon/profil sürümü değiştiğinde eski sonuçların değişmemesi.

### Açık kararlara Codex önerisi

| Konu | Codex önerisi |
|------|---------------|
| Fill | V1'den itibaren ertesi geçerli seans açılışı; aynı kapanış fill yok |
| Komisyon | Profil bazlı değil, bütün arena için ortak ve konfigüre edilebilir tek-yön bps |
| Slippage | İlk sürümde ortak sabit; sonra order/ADV ve likiditeye bağlı model |
| Kelly | Varsayılan çeyrek Kelly; yalnız kaliteyi geçmiş olasılıkla |
| Telegram | Motor + shadow mode kararlı olduktan sonra Faz 1'in sonunda |
| Şema | Uygulamadan önce SQL ve benzersiz/atomik işlem kuralları ayrıca onaylanmalı |
| ML fallback | Kalitesiz/bayat modelde yeni ML işlemi yok; hesap nakitte bekler |
| Settlement | V1 varsayımı açıkça belgelenmeli; şemada `unsettled_cash` için yer bırakılmalı |
| Fon hesabı | Ortak başlangıçlı yeni sezon veya replay olmadan lig kıyasına eklenmemeli |
| Plan durumu | "Tasarım onaylandı" yerine "Taslak — teknik onay bekliyor" |

### Nihai başarı kriterleri

Faz 1, yalnız ekran açılması veya tabloların dolmasıyla tamamlanmış sayılmaz.
Aşağıdaki koşullar birlikte sağlanmalıdır:

- Hiçbir karar, karar anında bilinmeyen fiyatla doldurulmuyor.
- Aynı run'ın tekrarı hesap bakiyesini veya pozisyonu ikinci kez değiştirmiyor.
- Her P&L rakamı order → fill → cash ledger → position → equity zinciriyle
  geriye doğru açıklanabiliyor.
- Bütün profiller aynı veri ve ortak execution maliyetlerini kullanıyor.
- ML hesabı yalnız örnek-dışı kalite kapısını geçen modelle işlem yapıyor.
- Bayat/eksik veri güvenli biçimde yeni işlemi durduruyor.
- Historical replay ve canlı paper akışı aynı execution çekirdeğini kullanıyor.
- Arena profilleri ile tek-değişkenli bilimsel kıyas birbirinden ayrılıyor.
- XU100 benchmark aynı başlangıç zamanı ve fill sözleşmesiyle hesaplanıyor.
- UI uyusa dahi cron, ledger ve bildirim akışı bağımsız çalışıyor.

Bu kriterler sağlandığında sistem yalnız "sinyal gösteren panel" olmaktan çıkıp,
sonuçları denetlenebilir ve stratejileri karşılaştırılabilir gerçek bir
paper-trading araştırma platformuna dönüşür.

---

## Claude — 2026-06-29 00:42 (TR) · Codex review'una cevap + sentez

> **Ne yaptım:** Codex'in eklediği review'u baştan sona okudum, iki somut kod
> iddiasını dosyaları açıp **kendi gözümle doğruladım**, sonra "nerede haklı,
> nerede ben yanlıştım, nerede ona itirazım var" diye tek tek değerlendirdim.
> Aşağısı sohbet gibi yazıldı; sonunda planın **en son sentez hâli** var.

Abi açık konuşayım: Codex'in review'u iyi. Hem de çok iyi. "Bana laf sokmuş,
egomu koruyayım" havasına girmeden söylüyorum — önerdiği şeylerin neredeyse
tamamı doğru ve bir kısmı **benim planımdaki gerçek hataları** yakaladı. Bir
mühendis arkadaşın senin tasarımını alıp "şurası patlar, şurada kendini
kandırıyorsun" demesi gibi; kıymetli. Ama bir noktada da ölçüyü kaçırıyor; onu da
söyleyeceğim. Madde madde gidelim.

### 1) Önce dürüst itiraf: Codex benim planımda 2 şeyi yakaladı, ikisinde de haklı

**(a) `p_up` yön hatası — bu sadece planımda değil, canlı kodda bir bug.**
Kodu açtım, doğruladım: `tradingagents/analysis/confidence.py` (108–112. satır)
şunu yapıyor:
```python
conviction = abs(p_up - 0.5) * 2
score += conviction * 12
```
`abs()` yüzünden model **kararın tersini** güçlü söylediğinde bile güven *artıyor*.
Somut örnek: karar "AL", ama model `p_up = 0.10` (yani "bu hisse düşecek" diyor güçlü
şekilde). `abs(0.10 − 0.5)·2 = 0.8` → skora **+9.6** ekleniyor. Model "alma" diye
bağırırken sistem güveni yükseltiyor. Bu mantık hatası. Doğrusu **yönlü** olmalı:
AL kararında yüksek `p_up` olumlu / düşük `p_up` cezalı; SAT'ta tersi. Codex tam
isabet. Üstelik bu, **4 numaralı ML-öncelikli hesabın** tam da bel bağladığı sinyal
— düzeltilmeden o hesap gürültüyle işlem açar. Faz 0'a alıyorum, pazarlıksız.

**(b) ½-Kelly değil, çeyrek Kelly.** `tradingagents/analytics/sizing.py`'ye baktım:
`position_size(..., kelly_mult=0.25)`, docstring bile "**çeyrek Kelly (0.25)**" diyor.
Ben dokümanda ısrarla "½-Kelly" yazmışım — düz yanlış. (Komik olan, modül başlığındaki
yorumda "½-Kelly" geçiyor ama işleyen varsayılan 0.25; kodun kendi yorumu bile
tutarsız.) **Errata:** yukarıdaki kendi bölümümde "½-Kelly" geçen her yer "çeyrek
Kelly" olacak. Tarihsel kayıt bozulmasın diye üstü silmedim; istersen düzeltirim.
Yarım Kelly'yi denemek istersek onu **ayrı bir ablation profili** yaparız (zaten
Codex de bunu söylüyor).

Bu ikisi küçük şeyler değil — biri canlı bug, biri de "fair race"i etkileyen bir
parametre. Yakaladığı için Codex'e puan.

### 2) Codex'in haklı olduğu ve doğrudan planıma aldığım noktalar

Bunların hepsinde "evet, doğru, alıyorum" diyorum; her birine bir cümle gerekçe:

- **Fill, v1'den itibaren T+1 açılış olmalı (aynı gün kapanış DEĞİL).** Benim
  "v1'de kapanıştan doldur, komisyon telafi eder" cümlem **yanlış muhakemeydi.**
  Komisyon *maliyeti* modeller; look-ahead ise *bias*'tır — başka bir hata sınıfı.
  Sistemin bütün amacı "sinyalim para kazandırıyor mu" ölçmekse, ilk günden kendini
  kandıran (look-ahead'li) bir fill koymak ölçümü yalancı yapar. Codex haklı; T+1
  açılış zaten çok daha fazla iş değil.
- **Stop/hedef gün-içi OHLC ile kontrol edilmeli.** Sadece kapanışa bakarsan gün
  içinde tetiklenen stopu kaçırırsın; gap açılışta teorik stop fiyatından değil
  elde edilebilir ilk fiyattan dolması, aynı barda hem stop hem hedef görülürse
  muhafazakâr varsayım… hepsi doğru. Benim akışım bunu atlamıştı.
- **Idempotent run + atomik ledger + benzersiz kısıt.** GitHub Actions cron'u
  gerçekten iki kez tetiklenebilir/retry edebilir. `run_id` + unique constraint +
  tek transaction olmazsa çift işlem ve bozuk kasa olur. 3 tablolu modelim bunu
  düşünmemişti. Doğru.
- **Tek dondurulmuş EOD snapshot, tüm hesaplar onu tüketsin.** Bu ince ama kritik:
  5 hesap ayrı ayrı yfinance/analiz çağırırsa veri/timestamp ufak ufak kayar →
  "eşit başlangıç" yarışı **adil olmaktan çıkar.** Senin "hangisi kazandırıyor"
  sorununun bütün anlamı buna bağlı. Codex'in en değerli yakalamalarından.
- **Parayı `Decimal`/`numeric` tut, `float` değil.** Ledger'da float birikimi klasik
  hata. Doğru.
- **Profilden bağımsız kill switch'ler.** "Agresif, rejim filtresi kapalı" olsa bile
  bayat-veri/illikidite/negatif-kasa korumalarını hiçbir profil kapatamamalı. Bunlar
  strateji değil, *emniyet*. Doğru.
- **Kasa yetmezse ticker sırasıyla değil, ortak skorla sırala-sonra-dağıt.**
  Alfabetik alırsan portföyünü hissenin *adı* belirler, sinyal kalitesi değil. Doğru
  ve aynı zamanda determinizm meselesi.
- **Arena (lig) ile laboratuvar (ablation) ayrılmalı.** İşte bu, senin asıl
  derdine birebir hizmet ediyor. Benim 5 profilim aynı anda *evren + eşik + sizing +
  rejim + yoğunlaşma*yı değiştiriyor; Agresif kazanırsa "BIST100 mü, düşük eşik mi,
  yoğunlaşma mı kazandırdı" ayırt **edemezsin.** "ML kuralı yendi mi?", "BIST100
  faydalı mı?" gibi sorular ancak **tek değişkenli** kıyasla cevaplanır. Lig'i
  "ürün/eğlence" görünümü olarak tut, yanına "laboratuvar" görünümü ekle. Çok yerinde.
- **Replay'de survivorship bias.** Bugünkü BIST100 listesini geçmişe uygularsan
  "zaten hayatta kalmışları" test edersin → şişik sonuç. Canlı-ileri arenada sorun
  yok; ama tarihsel replay yapacaksak dönemsel endeks üyeliği gerekir ya da sınır
  açıkça yazılır. Doğru.
- **Shadow mode (önce gölgede koş).** Birkaç seans sinyal + PENDING emir üret, fill
  uygulamadan ertesi gün zamanlama/veri doğruluğunu denetle, sonra canlıya geç. Ucuz
  sigorta, akıllıca.
- **Telegram yalnız tamamlanmış run'dan.** Yarım/başarısız koşuyu "başarılı" gibi
  bildirme. Doğru.
- **TEFAS sertleştirme** (cache/retry/backoff/rate-limit/ToS/atıf). KAP dersiyle
  birebir tutarlı. Doğru.

### 3) Codex'e itirazım: "ne" değil, "ne zaman" — orantı meselesi

Burada ayrışıyoruz. Codex'in dediklerinin hepsi *doğru*, ama hepsini *baştan*
yaparsak planın boyutu şöyle değişiyor: **3 tablo → 9 tablo**, **~12 adım → 30 adım**,
üstüne `arena_seasons`, historical replay, ayrı ablation lab, champion/challenger ML…
Bu, "kusursuz araştırma platformu" için doğru. Ama senin **bana ilk gün anlattığın
derdi** hatırla: *"biraz kurmuştum ama adam akıllı kullanamıyorum, takip edemiyorum."*
Yani senin acın eksik titizlik değil — **çalışan, görebildiğin bir şeyin olmaması.**

Eğer 30 adımlık akademik kaleyi önce inşa edersek, ilk çalışan ekranı aylar sonra
görürsün ve aynı "kuramadım" duygusuna geri döneriz. O yüzden Codex'in maddelerini
ikiye ayırıyorum:

- **Pazarlıksız doğruluk (ŞİMDİ, ilk sürümün şartı):** p_up yön fix · T+1 açılış
  fill · OHLC stop mantığı · idempotent run + atomik ledger + unique kısıt · tek
  dondurulmuş snapshot · `Decimal` para · profilden bağımsız kill switch'ler ·
  sırala-sonra-dağıt · tamamlanmış-run Telegram. Bunlar "sonra eklenecek süs" değil,
  ölçümü güvenilir yapan **çekirdek.** Hepsini alıyorum.
- **Araştırma olgunluğu (SONRA, ama yol haritasında):** tam `arena_seasons`
  alt-sistemi · point-in-time endeks üyeliğiyle tarihsel replay · ayrı ablation
  lab UI'ı · champion/challenger ML · ADV-bağlı slippage modeli. Değerli ama ilk
  ışığı geciktirmemeli.

Bir spesifik nüans: Codex, **ML hedefini execution'a göre yeniden tanımla** diyor
(10 günlük yukarı/aşağı etiketi yerine T+1 fill + tutma ufku + XU100-göreli +
maliyet-sonrası). Bu **doğru** ama Faz 0 *blocker'ı* değil. Faz 0'ın şartı `p_up`'ın
*yönünü* düzeltmek; etiketi yeniden eğitmek bir *iyileştirme* ve ilk ışığı bekletmemeli
→ onu Faz 3'e koyuyorum. Bir de `arena_seasons`: baştan tam sezon-yönetimi yerine
hafif bir `version`/`season` kolonuyla başlarım; ihtiyaç büyüyünce alt-sisteme
çıkarırız.

Özet itirazım tek cümle: **Codex "doğru olanı" söylüyor, ben "doğru olanı zamanında"
istiyorum.** Çelişmiyoruz; ben sadece çekirdeği önce ayağa kaldırıp üstüne titizliği
katman katman eklemek istiyorum ki sen *çalışan* bir şeyi erkenden görebilesin.

### 4) Sentez — planın en son hâli (güncellenen kararlar)

| Konu | Önceki (benim) | **Güncel sentez** |
|------|----------------|-------------------|
| Plan durumu | "Tasarım onaylandı" | **"Taslak — teknik onay bekliyor"** (Codex haklı) |
| Fill | v1 kapanış, v2 T+1 | **v1'den T+1 geçerli seans açılışı + yönlü slippage** |
| Stop/hedef | kapanışta | **gün-içi OHLC (Low≤stop / High≥target) + gap + ambiguous-bar** |
| Kelly | "½-Kelly" (hatalı) | **çeyrek Kelly (kod gerçeği); ½ istersek ayrı ablation profili** |
| Komisyon | %0.05 sabit | **tüm arenada ortak, tek-yön bps, `execution_config`'te konfigüre** |
| p_up | `abs()` (bug) | **yönlü + ML kalite kapısı (skill>0, AUC/Brier eşiği, tazelik, n)** |
| Snapshot | her hesap kendi | **tek dondurulmuş EOD snapshot, tüm hesaplar paylaşır** |
| Ledger | 3 tablo | **order→fill→cash→position→equity, atomik, `Decimal`, idempotent** |
| Para tipi | (belirsiz) | **`numeric`/`Decimal`** |
| Kıyas | sadece lig | **Lig (ürün) + Laboratuvar (tek değişkenli ablation) ayrı** |
| Kill switch | dağınık | **profilden bağımsız, kapatılamayan emniyet katmanı + global PAUSE** |
| Kasa dağıtımı | (belirsiz) | **adayları ortak skorla sırala → portföy bütçesini topluca dağıt** |
| Telegram | cron sonu | **yalnız tamamlanmış run'dan, motor kararlı olunca (faz sonu)** |
| Fon hesabı | 6. hesap | **ortak başlangıçlı yeni sezon/replay olmadan lige sokma** |
| Replay | (yoktu) | **point-in-time endeks üyeliği; yoksa survivorship sınırı raporla** |

### 5) Revize faz sırası (Codex'in fazlamasını sadeleştirip benimsiyorum)

Açıkçası Codex'in faz kurgusu benim "5 hesabı kur gitsin"den **daha iyi** — özellikle
*shadow mode* ve *replay-doğrulama* adımlarını araya sokması olgun bir fikir. Hafifçe
sadeleştirip alıyorum:

- **Faz 0 — Sinyal & veri sözleşmesi:** p_up yön fix · ML kalite kapısı + model
  tazeliği · makro şoku tüm analiz yollarına bağla · tek EOD snapshot · strateji/model/
  evren sürümleme.
- **Faz 1A — Saf execution çekirdeği:** Supabase/Streamlit'siz, test edilebilir motor;
  T+1 fill · gap · OHLC stop/hedef · komisyon · slippage · lot · nakit. *Aynı çekirdek
  hem canlı paper hem historical replay'de kullanılacak.*
- **Faz 1B — Şema & atomik ledger:** SQL'i **uygulamadan önce sana göster/onayla** ·
  tablolar + unique kısıt + atomik fill RPC · 5 hesabı idempotent seed.
- **Faz 1C — Replay & doğrulama:** motoru geçmiş barlarda koştur; look-ahead/duplicate/
  stale testlerini geçir; lig ↔ laboratuvar sonuçlarını ayrı raporla.
- **Faz 1D — Shadow mode:** birkaç seans sinyal + PENDING, fill yok; zamanlama/veri
  doğruluğunu denetle.
- **Faz 1E — Canlı paper arena:** fill+ledger aç · UI (lig + lab + emir/fill günlüğü +
  CSV) · Telegram (en son).
- **Faz 2 — Fon:** TEFAS sözleşmesi + fon momentum + valör simülasyonu + 6. hesabı
  ortak başlangıçla devreye al.
- **Faz 3 — Gelişmiş ML:** pooled panel veri seti · GBDT champion/challenger · purged
  walk-forward (maliyet-sonrası, XU100-göreli hedef) · ancak klasik model kalıcı OOS
  edge gösterirse 1D-CNN/TCN/GRU'yu **ayrı araştırma kolunda** dene (canlı modeli
  doğrudan değiştirme).

### 6) Codex'in "açık kararlar" tablosuna benim cevabım

Hepsine katılıyorum, ikisinde küçük şerh:
- Fill → **kabul** (T+1). · Komisyon → **kabul** (ortak bps). · Slippage → **kabul**
  (önce sabit, sonra ADV). · Kelly → **kabul** (çeyrek). · Şema önce onay → **kabul.**
  · ML fallback nakitte bekle → **kabul.** · Settlement v1 varsayımı belgelensin →
  **kabul** (`unsettled_cash` kolonu dursun). · Fon ayrı sezon → **kabul.** · Plan
  durumu "taslak" → **kabul.**
- **Şerh 1 (Telegram):** Codex "shadow mode kararlı olunca" diyor; ben "tamamlanmış
  run + Faz 1E" diyorum — pratikte aynı kapıya çıkıyor, sadece sıralamada netleştirdim.
- **Şerh 2 (ML etiket yeniden tanımı):** doğru ama Faz 0 değil Faz 3 — gerekçe §3.

### Kapanış

Codex'in katkısı planı "sinyal gösteren panel"den "denetlenebilir araştırma
platformu"na taşıdı; benim katkım onu, senin nihayet **çalışırken görebileceğin** bir
ilk-ışığa zamanında ulaştıracak şekilde sıralamak. İkisi rakip değil, üst üste biniyor:
**Codex titizliği, ben de o titizliğin doğru sırada gelmesini** koyuyorum ortaya.

Bir sonraki adım bana kalırsa **Faz 0** (özellikle `p_up` yön bug'ı — bu zaten canlı
sistemini de etkiliyor). "Başla" dersen oradan açarım; ya da önce Faz 1B şemasını
SQL olarak önüne koyayım, onaylarsın, sonra kod. Sen söyle abi.

---

## Codex — 2026-06-29 00:47 (TR) · Claude sentezine cevap + güncel nihai öneri

> **Ne yaptım:** Claude'un 00:42'de eklediği cevabı ve sentezi, önceki Claude planı
> ve kendi 00:37 incelememle birlikte yeniden okudum. Bu bölümde kimin "kazandığını"
> değil, hangi kararın neden güvenilir olduğunu ve buradan sonra gerçekten hangi
> sırayla ilerlememiz gerektiğini anlatıyorum. Önceki kayıtlar günlük niteliğinde
> korunuyor; aşağıdaki bölüm benim şu anki güncel görüşümdür.

Abi önce kısa cevabı vereyim: **Claude son cevabında büyük ölçüde mantıklı konuşmuş.**
Hataları inkâr etmemiş, koda bakıp doğrulamış ve benim uzun vadeli güvenilirlik
önerilerimi "hemen gerekli" ile "sonra olgunlaşacak" diye ayırmış. Bu ayrım projeyi
gerçekten bitirebilmek açısından değerli. Benim ilk review'um doğru bir hedef mimari
çiziyordu ama ilk çalışan sürümün kapsamını yer yer gereğinden geniş tutuyordu.
Claude'un burada yaptığı kapsam düzeltmesini kabul ediyorum.

Fakat iki konuda hâlâ net bir koşul koyuyorum:

1. **ML hesabı ilk sürümde gerçekten işlem açacaksa**, modelin hedefi ile execution
   ufku arasındaki uyum Faz 3'e bırakılamaz. Ya bu uyum erkenden çözülür ya da
   ML-öncelikli hesap ilk sürümde yalnız `observer/shadow` olarak tahmin üretir,
   para kullanmaz.
2. Tam teşekküllü sezon yönetimi sonraya kalabilir ama bütün kayıtları hangi yarışa
   ait olduğunu söyleyen basit bir **`season_id` ilk günden bulunmalıdır.** Sonradan
   eklemek mümkün olsa da mevcut hesapları, replay'i ve farklı başlangıç tarihlerini
   ayırmak gereksiz yere zorlaşır.

### Claude nerede doğru söylüyor?

Claude'un "doğru olanı doğru zamanda yapalım" itirazı yerinde. Kullanıcının aylarca
yalnız altyapı görüp hâlâ çalışan bir ekran görememesi gerçek bir ürün riskidir.
Burada çözüm doğruluk kurallarını atmak değil, **dikey ve küçük bir çalışan dilim**
oluşturmaktır.

Örneğin önce tek hesap + iki ticker + yerel veriyle şu zinciri çalışırken görebiliriz:

```
EOD snapshot → planlanan emir → sonraki seans açılış fill'i
→ atomik nakit/pozisyon → equity → basit ekran
```

Bu dilim doğru çalıştıktan sonra aynı motoru beş hesaba ve geniş evrene açarız.
Böylece hem erken "ışık" görürüz hem de sonradan atmak zorunda kalacağımız sahte
bir demo üretmeyiz. Claude'un hız kaygısına katılıyorum; yalnız çözümü çekirdeği
gevşetmekte değil, **evreni ve UI kapsamını küçültmekte** görüyorum.

Claude şu maddeleri çekirdek şartlar listesine alırken tamamen doğru davranmış:

- `p_up` yön hatasının arena öncesinde düzeltilmesi.
- Aynı kapanıştan fill'in kaldırılıp sonraki geçerli seans açılışının kullanılması.
- Günlük `Open/High/Low/Close` ile gap ve stop/hedef mantığının tanımlanması.
- Cron tekrarlarında duplicate emir üretmeyen idempotent run tasarımı.
- Nakit, fill ve pozisyon güncellemesinin tek transaction/RPC içinde yapılması.
- Bütün profillerin aynı dondurulmuş EOD snapshot'ını tüketmesi.
- Para alanlarında `numeric`/`Decimal` kullanılması.
- Stratejik profilden bağımsız veri/negatif-kasa/duplicate-emir kill switch'leri.
- Adayların alfabetik sırayla değil, önce sıralanıp sonra topluca dağıtılması.
- Telegram'ın yalnız tamamlanmış run'dan ve motor kararlı olduktan sonra çalışması.
- Arena ligi ile tek-değişkenli bilimsel kıyasın kavramsal olarak ayrılması.

Bu maddeler "akademik süs" değil; ölçtüğümüz kârın gerçekten açıklanabilir olmasını
sağlayan tabandır.

### Benim önceki önerimde neresi fazla genişti?

İlk önerimde order, fill, cash ledger, prediction, season ve ayrı laboratuvar dahil
tam araştırma platformunun hedef şemasını bir kerede koydum. Uzun vadede yanlış
değil; ama paper-only V1'de kısmi fill veya gerçek broker entegrasyonu yoksa bazı
tablolar birleştirilebilir.

Özellikle:

- Ayrı `paper_predictions` tablosu ilk gün şart değil; model bilgisi sürümlü EOD
  snapshot'ın içinde korunabilir.
- Partial fill desteklemeyeceksek `paper_orders` içine `filled_at`, `fill_price`,
  `filled_quantity`, `commission` ve `slippage` alanları koyup ayrı `paper_fills`
  tablosunu sonraya bırakabiliriz.
- Laboratuvar motorunun konfigürasyon desteği baştan düşünülmeli ama ayrı laboratuvar
  UI'ı canlı paper arena ekranını geciktirmemeli.
- Point-in-time BIST100 üyeliği kusursuz historical araştırma için gerekli; ancak
  forward-only canlı arena için blocker değildir. Basit replay güncel evrenle
  yapılırsa sonuçta survivorship uyarısı açıkça gösterilmelidir.
- ADV'ye bağlı gelişmiş slippage, champion/challenger yönetimi ve derin öğrenme
  araştırma olgunluğu fazına bırakılabilir.

Yani Claude'un "dokuz tabloyu ilk gün kurmayalım" itirazını kabul ediyorum. Ama
pending emir, atomik nakit/pozisyon ve run kimliği olmadan üç tabloya geri dönmeyi de
doğru bulmuyorum. Aşağıda ikisinin ortasında güvenli bir MVP veriyorum.

### Hâlâ itiraz ettiğim noktalar

#### ML hedefini Faz 3'e koşulsuz bırakmak doğru değil

Mevcut model "bugünkü kapanıştan yaklaşık 10 gün sonra fiyat yukarıda mı?" sorusuna
yakın bir hedef öğreniyor. Paper motoru ise sinyali kapanışta üretip sonraki seans
açılışında dolduracak; stop, hedef veya maksimum tutma süresiyle daha erken
çıkabilecek. Bunlar aynı ekonomik bahis değildir.

Bu yüzden iki güvenli seçenek var:

- **Seçenek A — önerim:** İlk çalışan arenada dört kural hesabı para kullanır;
  ML hesabı `SHADOW/OBSERVER` olur. Tahminlerini ve gerçekleşen sonuçlarını biriktirir,
  kalite ve hedef uyumu kanıtlanınca sezon başında aktive edilir.
- **Seçenek B:** ML hedefi, T+1/sonraki seans fill ve belirlenen tutma/çıkış
  sözleşmesiyle Faz 0'da yeniden tanımlanır; o zaman ML hesabı ilk sezonda işlem açar.

"Eski modeli şimdilik kullanalım, sonra hedefi düzeltiriz" seçeneğini önermiyorum.
Çünkü bu durumda ML hesabının kaybetmesi veya kazanması model kalitesinden çok yanlış
tanımlanmış hedefi ölçer; "ML kuralları yendi mi?" sorusuna cevap vermez.

#### Basit `season_id` sonraya kalmamalı

Burada tam bir sezon yönetim ekranından bahsetmiyorum. İlk gün yalnız şunlar yeter:

- `arena_seasons`: `id`, ad, başlangıç zamanı, başlangıç bütçesi, durum,
  ortak execution ayarlarının snapshot'ı.
- Diğer bütün paper tablolarında `season_id` foreign key.

Bu küçük ek, reset atmayı, ikinci deneyi, fon hesabının farklı başlangıcını ve eski
sonuçların karışmamasını çözer. Sonradan veri taşımaktan daha ucuzdur.

### Claude'un sentezinde eksik kalan birkaç teknik nüans

#### "T+1" yerine "sonraki geçerli seans" diyelim

`T+1` finans dünyasında settlement/valör anlamına da gelebilir. Biz burada emrin
karar gününden sonraki işlem seansının açılış fiyatıyla doldurulmasını kastediyoruz.
Kod ve şemada daha açık isimler kullanalım:

- `signal_session`
- `scheduled_fill_session`
- `next_valid_session_open`

Settlement ise ayrı bir konudur ve `unsettled_cash` ile modellenir. Böylece iki
farklı T+1 kavramını birbirine karıştırmayız.

#### Karar ve fill için iki ayrı saat gerekir

Günlük motor aslında tek cron değil, iki mantıksal adımdır:

1. **Kapanış sonrası karar işi:** tamamlanmış EOD veriyi dondurur ve pending emir
   üretir.
2. **Sonraki seans fill işi:** açılış verisi erişilebilir olduğunda pending emirleri
   atomik olarak doldurur.

GitHub Actions birkaç dakika gecikebilir; paper simülasyonda problem değildir.
Fill işi saat 10:00'da çalışmış gibi davranmamalı, gerçekten hangi açılış barını
kullandığını ve veriyi ne zaman aldığını ayrı kaydetmelidir. Açılış verisi henüz
gelmemişse fill uydurmak yerine retry etmelidir.

#### Resmî equity ile canlı tahmini ayıralım

Saatlik mark-to-market kullanıcı için güzel, fakat gün içindeki tamamlanmamış günlük
barlar resmî performans serisine karışmamalıdır.

- **Resmî equity:** Günde bir, tamamlanmış seans kapanışıyla; Sharpe/drawdown/CAGR
  bunun üzerinden hesaplanır.
- **Canlı tahmini equity:** Saatlik veya kullanıcı açtığında; yalnız ekranda
  `estimated/intraday` etiketiyle gösterilir, model eğitimine ve resmî istatistiğe
  girmez.

Bu ayrım yapılmazsa aynı günlük hareket saatlik tekrarlarla örnek sayısını ve risk
istatistiklerini şişirebilir.

#### Model kalite kapısının metrikleri doğru ölçülmeli

"Skill > 0, AUC ve Brier eşiği" yön olarak doğru ama sabit rakamları hemen kilitlemek
yerine şu sözleşmeyi öneriyorum:

- AUC ve skill yalnız zaman sıralı, örnek-dışı tahminlerden hesaplanır.
- Kalibratörün eğitildiği tahminlerle aynı satırlarda "kalibre Brier" raporlanmaz;
  kalibrasyon sonrası performans ayrı, dokunulmamış test penceresinde ölçülür.
- Brier mutlak bir sayıdan ziyade naif sınıf olasılığına karşı **Brier Skill Score**
  ile değerlendirilir.
- Modelin hem genel OOS kalitesi hem son kayan pencere kalitesi izlenir.
- `model_version`, `trained_until`, `prediction_asof` ve metrik penceresi saklanır.
- Eşikler ilk diagnostic/replay sonuçlarına göre konfigürasyonda belirlenir; kodun
  içine sihirli sayı olarak gömülmez.

Mevcut `probability.py` Brier'ı isotonic kalibrasyondan önceki OOS olasılıklarda
hesaplıyor ve isotonic modeli aynı OOS havuzunda fit ediyor. Bu tamamen işe yaramaz
demek değildir; fakat "kalibre olasılığın dokunulmamış test performansı" olarak
yorumlanmamalıdır. Arena kalite kapısından önce bu ayrım düzeltilmelidir.

#### `p_up` düzeltmesi yalnız `abs()` işaretini değiştirmek değildir

Kural motoruna model katkısı için açık bir yön sözleşmesi öneriyorum:

```text
model_edge  = 2 × (p_up - 0.5)          # -1 … +1
decision_dir = +1 (AL), -1 (SAT), 0 (nötr)
uyum_katkısı = decision_dir × model_edge × model_ağırlığı
```

- Kural kararı AL ve model yükseliş diyorsa katkı pozitif.
- Kural kararı AL ve model düşüş diyorsa katkı negatif.
- Kural kararı SAT olduğunda yön ters çevrilir.
- Nötr kararda model, kural güvenini yapay biçimde yükseltmez.
- Model kalite kapısını geçmediyse katkı tam olarak sıfırdır.

ML-öncelikli profil ise bu uyum skorundan ayrı tasarlanmalıdır; onun yön kararı
doğrudan kaliteli `p_up`'tan gelir, ancak likidite/veri/makro/portföy emniyet kapıları
yine uygulanır.

#### Corporate action için en azından sert koruma şimdi olmalı

Temettü, bedelsiz/split veya veri hatası bir gecede dev fiyat değişimi gibi görünüp
sahte stop, sahte P&L ve yanlış ML etiketi üretebilir. Tam kurumsal aksiyon motoru
sonraya kalabilir; V1'de en azından olağandışı fiyat oranını yakalayan bir kontrol,
`SUSPENDED_DATA_REVIEW` durumu ve manuel inceleme olmadan fill'i durduran kill switch
olmalıdır.

### Güvenli ama sade MVP şeması

İlk sürüm için aşağıdaki yedi tablo yeterli ve dengeli görünüyor:

```
arena_seasons
  Yarış kimliği, ortak başlangıç ve execution ayarlarının değişmez snapshot'ı

paper_accounts
  season_id, profil, başlangıç kasa, durum, strateji/rules snapshot

paper_runs
  Karar/fill/EOD işlerinin benzersiz kimliği, as-of zamanı ve tamamlanma durumu

paper_orders
  PENDING/FILLED/CANCELLED/REJECTED; V1'de fill alanları aynı tabloda

paper_cash_ledger
  Alış, satış, komisyon ve düzeltme gibi değişmez nakit olayları

paper_positions
  Atomik fill sonrası güncellenen materyalize pozisyon durumu

paper_equity
  Resmî EOD equity; benchmark ve veri-tazeliği bilgisi
```

İlk sürümde ayrıca:

- Tahmin detayları sürümlü `analysis_snapshots.signals` içinde tutulabilir.
- Kısmi fill yoktur; emir ya tamamen dolar ya dolmaz.
- `paper_orders` ile ayrı `paper_fills` tablosu, gerçek broker/kısmi fill gündeme
  geldiğinde ayrılır.
- Nakit ledger + account cache + position güncellemesi tek RPC transaction'ında
  yapılır.
- Her tabloda `season_id`; para alanlarında `numeric`; kod tarafında `Decimal`.
- Unique/idempotency anahtarı en az sezon + hesap + signal session + ticker + side +
  strategy version bileşenlerini kapsar.

Bu, Claude'un üç tablosundan daha güvenli; benim ilk dokuz tabloluk hedefimden daha
hafif bir uzlaşmadır.

### Artık önerdiğim nihai uygulama sırası

#### Sinyal düzeltmeleri ve sözleşme

- `p_up` yön hatasını test yazarak düzelt.
- Makro şok ve model cache kullanımını manuel/cron/günlük motor için tek analiz
  yolunda birleştir.
- Model kalite ölçümünü kalibrasyon sonrası dokunulmamış test penceresiyle düzelt.
- ML hesabı için karar ver: hedefi şimdi hizala veya hesabı `OBSERVER` başlat.
- Dondurulmuş EOD snapshot'a as-of, model, strateji ve evren sürümü ekle.
- Resmî EOD veri ile intraday tahmini veriyi ayır.

#### Küçük çalışan dikey dilim

- Supabase'siz saf execution çekirdeğini önce tek hesap ve iki ticker ile çalıştır.
- Sonraki geçerli seans fill'i, gap, OHLC stop/hedef, commission, slippage, lot ve
  nakit kurallarını uygula.
- Aynı günlük barda stop+hedef durumunda muhafazakâr ve işaretli sonuç üret.
- Basit yerel çıktı/equity tablosuyla zinciri erkenden görünür yap.

#### Minimal şema ve atomik ledger

- Yedi tabloluk MVP SQL'ini önce kullanıcıya göster ve onaylat.
- `season_id`, `run_id`, unique constraint ve global pause sözleşmesini ekle.
- Fill + cash ledger + position update işlemini tek RPC transaction'ı yap.
- Önce tek hesapla uçtan uca doğrula, sonra dört aktif kural hesabını seed et.
- ML hesabını hedef uyumuna göre aktif veya observer olarak seed et.

#### Replay ve shadow doğrulama

- Aynı execution çekirdeğini kısa bir historical replay'de çalıştır.
- İlk replay'de güncel evren kullanılıyorsa survivorship sınırlamasını açıkça yaz;
  point-in-time evreni araştırma olgunluğu fazına bırak.
- Duplicate run, stale data, gap, ambiguous bar, negative cash ve corporate-action
  testlerini geçir.
- Ardından birkaç gerçek seans yalnız pending emir üret; fill uygulamadan zaman ve
  fiyat eşleşmesini gözlemle.

#### Canlı paper arena V1

- Dört kural hesabında veya hedefi hizalanmışsa beş hesapta atomik fill'i aç.
- UI'da önce lig, hesap detayı, resmî EOD equity, tahmini canlı equity ve emir
  günlüğünü göster.
- CSV dışa aktarma ekle.
- Yalnız başarılı/tamamlanmış run'lardan Telegram özeti gönder.
- Laboratuvar motorunun config desteğini koru; ayrı lab UI'ını V1 sonrasına bırak.

#### Araştırma olgunluğu

- Point-in-time BIST üyelikleri ve survivorship-düzeltilmiş replay.
- Lig yanında tek-değişkenli laboratuvar UI'ı.
- ADV/order büyüklüğüne bağlı slippage.
- Pooled panel veri, maliyet-sonrası XU100-relative hedef ve champion/challenger.
- Klasik modeller kalıcı OOS edge gösterirse ayrı TCN/1D-CNN/GRU araştırması.
- Fon veri/valör motoru ve bütün hesaplar için ortak başlangıçlı yeni sezon.

### Şimdi yapılacaklar ve sonraya bırakılacaklar

| Konu | Karar | Neden |
|------|-------|-------|
| `p_up` yön bug'ı | Şimdi | Mevcut canlı güveni de yanlış etkiliyor |
| Model kalite ölçümü | Şimdi | ML sinyalinin işlem yetkisi için zorunlu |
| ML hedef uyumu | Aktif ML hesabı varsa şimdi; observer ise sonra | Yanlış soruyu optimize eden hesabı para ile yarıştırmamak için |
| Sonraki seans fill + OHLC stop | Şimdi | Look-ahead ve sahte P&L'i önler |
| Idempotency + atomik RPC | Şimdi | Çift emir ve bozuk kasayı önler |
| Minimal `season_id` | Şimdi | Deneyleri ve başlangıçları karıştırmamak için |
| Resmî EOD / intraday ayrımı | Şimdi | Performans örneklerini şişirmemek için |
| Dört kural hesabı | V1 | Erken çalışan arena sağlar |
| ML hesabı | Hedef/kalite durumuna bağlı | Observer başlatmak güvenli varsayılan |
| Ayrı laboratuvar UI'ı | V1 sonrası | Motor desteği baştan, ekranı sonra |
| Point-in-time BIST evreni | Araştırma fazı | Forward arena blocker'ı değil |
| Gelişmiş ADV slippage | Araştırma fazı | İlk sürümde ortak muhafazakâr sabit yeterli |
| Fon hesabı | Ayrı sezon/faz | Veri, NAV ve valör sözleşmesi farklı |
| RNN/CNN/GRU | En son araştırma | Önce klasik OOS edge kanıtlanmalı |

### Son sözüm

Claude'un son sentezi **iyi ve uygulanabilir**; özellikle çekirdek doğruluk ile
araştırma olgunluğunu ayırması benim ilk planımı daha gerçekçi hâle getirmiş. Yani
"Codex mantıklı söylemiş mi?" sorusunun cevabı evet; ama Claude'un dediği gibi bazı
doğru maddeleri ilk çalışan sürümü geciktirmeyecek şekilde katmanlamak gerekiyor.

Claude'un yanıldığı ya da fazla ertelediği yer ise ML hedef uyumunu, aktif ML hesabı
varken Faz 3'e bırakması ve minimal sezon kimliğini gereksiz ağır bir alt-sistem gibi
görmesi. Bunları yukarıdaki koşullu çözümle sadeleştirdim: **ML ya doğru hedefle işlem
açar ya observer kalır; sezon yönetimi ise yalnız bir tablo ve foreign key kadar
hafif başlar.**

Ben olsam bir sonraki adımda doğrudan dokuz tablo veya bütün UI'ı yazmaya başlamam.
Önce `p_up` bug'ı + model kalite ölçümü için testleri ve beklenen davranış
sözleşmesini çıkarırım. Hemen arkasından yedi tabloluk MVP şemasını SQL olarak önüne
koyarım. Sen şemayı onayladıktan sonra tek hesap/iki ticker dikey dilimini kurar,
çalıştığını birlikte görür ve sonra arenayı genişletiriz. Bu sıra hem Claude'un hız
kaygısını hem benim finansal doğruluk kaygımı karşılıyor.

---

## Claude — 2026-06-29 00:52 (TR) · Codex'in 2. turuna cevap (yakınsama)

> **Ne yaptım:** Codex'in 00:47'deki ikinci turunu baştan sona okudum, yine bir
> somut kod iddiasını (`probability.py` kalibrasyon ölçümü) açıp **doğruladım**, ve
> "nerede anlaştık, nerede hâlâ ekleyeceğim bir şey var" diye değerlendirdim. Kısa
> versiyon: **yakınsadık.** Aşağısı sohbet gibi; sonunda planın iki tur sonrası
> birleşik nihai hâli var.

Abi en baştan söyleyeyim: bu turda Codex'le **kavga edecek bir şey kalmadı.** Kapsamı
fazla geniş tuttuğunu kendi kabul etti, benim "çekirdek doğruluk vs araştırma
olgunluğu" ayrımımı benimsedi, üstüne de gerçekten işe yarar birkaç teknik nüans
ekledi. Bu noktada "dengeli görüneyim" diye yapay itiraz üretmek sahtekârlık olur —
o yüzden net konuşacağım: çoğunu **olduğu gibi alıyorum**, bir tanesini *daha da
iyi* buldum, ve sadece **senin vermen gereken bir kararı** ikimizin de yeterince öne
çıkarmadığını söyleyeceğim.

### 1) Doğruladığım 3. kod iddiası — Codex yine haklı (`probability.py`)

Kodu açtım. Codex'in dediği birebir doğru:
- `probability.py` (81. satır) `brier = brier_score_loss(oos_y, oos_p)` → Brier'ı
  **kalibrasyon öncesi ham** OOS olasılıklarda hesaplıyor.
- 87. satır `IsotonicRegression().fit(oos_p, oos_y)` → kalibratörü **aynı** OOS
  havuzunda fit ediyor; kalibre çıktıyı ölçecek dokunulmamış bir test penceresi yok.
- Sonuç: docstring "isotonic ile kalibre eder ve Brier ile güvenilirliğini ölçer"
  diyor, ama raporlanan Brier **döndürülen kalibre olasılığın** değil, ham olasılığın
  güvenilirliği. İkisi farklı sayı. "Tamamen çöp" değil (ham model OOS Brier'ı geçerli
  bir ölçüdür) ama "kalibre olasılığın dokunulmamış-test performansı" diye
  yorumlanamaz. Arena kalite kapısı buna bakacaksa, bakmadan önce düzelteceğiz.

Şunu da dürüstçe kaydedeyim: Codex iki turda **üç gerçek kod sorunu** çıkardı —
`abs(p_up−0.5)` yön bug'ı, çeyrek-Kelly ile dokümanımın çelişmesi, ve şimdi bu
kalibrasyon-ölçüm tutarsızlığı. Üçü de gerçek. Bu, review'a güvenimi artırıyor; laf
olsun diye yazmamış.

### 2) Bu turda eklediği ve aynen aldığım nüanslar

- **ML hesabı v1'de `OBSERVER` olsun (Codex'in A seçeneği).** İşte bunu sadece kabul
  etmiyorum, *daha iyi* buluyorum. Önceki turda biz iki uçta kalmıştık: ben "modeli
  şimdilik yönlü kullanalım, etiketi sonra düzeltiriz", Codex "etiketi şimdi yeniden
  tanımla yoksa ML parayla yarışmasın". Observer çözümü ikisini de aşıyor: ML hesabı
  ilk sezonda **tahmin üretir, track record biriktirir ama para harcamaz**; kalite +
  hedef-uyumu kanıtlanınca bir sonraki sezon başında parayla devreye girer. Hem ilk
  ışığı geciktirmez, hem "yanlış tanımlı hedefi para ile ölçme" hatasına düşmeyiz.
  Temiz.
- **İlk günden basit `season_id`.** Önceki turda `arena_seasons`'ı ağır bir
  alt-sistem gibi gördüm; Codex haklı olarak "bu sadece bir tablo + foreign key"
  dedi. Reset, ikinci deney, fonun farklı başlangıcı ve eski sonuçların karışmaması
  bu kadar ucuza çözülüyorsa, baştan koyalım. Kabul.
- **İsimlendirme: "T+1" deme.** Çünkü finans dünyasında T+1 *takas/valör* demek.
  Biz "karar gününden sonraki **geçerli seansın açılışı**"nı kastediyoruz →
  `signal_session`, `scheduled_fill_session`, `next_valid_session_open`. Takas ayrı
  konu, `unsettled_cash` ile modellenir. Haklı, net isim daha iyi.
- **Karar işi ile fill işi iki ayrı saat/iş olmalı.** (1) Kapanış sonrası karar →
  pending emir; (2) sonraki seans açılış verisi gelince → atomik fill. Fill işi
  "saat 10:00'da çalıştım" diye davranmamalı, *hangi açılış barını* kullandığını ve
  veriyi *ne zaman* aldığını kaydetmeli; açılış gelmemişse uydurmak yerine retry
  etmeli. Operasyonel olarak çok doğru; benim "günlük cron"umu ikiye bölüyor.
- **Resmî EOD equity ≠ canlı tahmini equity.** Bu benim "saatlik mark-to-market"
  fikrimi düzeltiyor: saatlik değer ekranda *güzel* ama tamamlanmamış günlük barlar
  **resmî performans serisine karışmamalı** (yoksa aynı hareket saatlik tekrarlarla
  örnek sayısını ve risk istatistiğini şişirir). Çözüm: resmî equity günde bir
  (kapanış), Sharpe/drawdown/CAGR ondan; saatlik olan `estimated/intraday` etiketiyle
  yalnız gösterimde, istatistiğe ve model eğitimine girmez. Kabul — benim fikrimin
  eksik yanını kapatmış.
- **`p_up` düzeltmesi sadece `abs()`'ı kaldırmak değil.** Codex açık bir yön
  sözleşmesi verdi: `model_edge = 2·(p_up−0.5)`, `katkı = karar_yönü · model_edge ·
  ağırlık`; nötr kararda model güveni yapay şişirmez; kalite kapısını geçmezse katkı
  **tam sıfır**. ML-öncelikli profil ise bu uyum skorundan ayrı, yönünü doğrudan
  kaliteli `p_up`'tan alır (ama likidite/veri/makro/portföy emniyet kapıları yine
  geçerli). Benim "yönlü olsun" tek satırımdan daha eksiksiz. Kabul.
- **Corporate-action için en azından sert koruma şimdi.** Temettü/bedelsiz/split veya
  veri hatası bir gecede dev fiyat sıçraması gibi görünüp sahte stop + sahte P&L +
  yanlış ML etiketi üretir. Tam motor sonraya kalsın ama V1'de: olağandışı fiyat-oranı
  kontrolü + `SUSPENDED_DATA_REVIEW` durumu + manuel inceleme olmadan fill'i durduran
  kill switch. Ben bunu "sonraya" koymuştum; Codex haklı, *sert koruma* ucuz ve çöp
  P&L'i baştan engelliyor → şimdiye alıyorum.
- **7 tabloluk MVP** (arena_seasons, paper_accounts, paper_runs, paper_orders [fill
  alanları içeride], paper_cash_ledger, paper_positions, paper_equity). Benim 3'ümle
  Codex'in 9'unun ortası; her tablonun bir gerekçesi var (runs = idempotency, cash_ledger
  = denetlenebilirlik, season = deney ayrımı). Makul uzlaşma, kabul.

### 3) Nerede ayrışıyoruz? Neredeyse hiçbir yerde — ve bunu saklamayacağım

Dürüst olmak gerekirse bu turda gerçek bir anlaşmazlık kalmadı. Codex benim hız
kaygımı, ben de onun doğruluk kaygısını içselleştirdik; üstelik onun bu turda
çıkardığı **"dikey dilim" (tek hesap + iki ticker + Supabase'siz çekirdek)** fikri,
benim "ilk ışığı geciktirme" derdimin tam da doğru çözümü: çekirdeği gevşetmeden
**evreni ve UI'ı küçültmek**. Bunu ben söylemeliydim, Codex söyledi; alıyorum.

### 4) Benim eklediğim: sessizce gömülmemesi gereken bir KULLANICI kararı var

İkimiz de "ML observer olsun" diye teknik olarak hemfikiriz. Ama bunun senin için bir
sonucu var ve onu sana açıkça sormadan geçmeyelim: **sen ilk turda ML'in kurallarla
*parayla* yarışmasına heveslenmiştin.** Observer kararı, V1'de yarışı *"4 para
hesabı + 1 gözlemci ML (+ sonra fon)"* yapıyor — yani ML ilk sezonda madalya
mücadelesine para koymadan, kenardan tahmin üreterek katılıyor. İki seçenek:

- **(A) ML gözlemci (güvenli, ikimizin önerisi):** ilk ışık hızlı gelir, ML haksız
  yere kazanıp/kaybedip seni yanıltmaz; bir sonraki sezon parayla girer.
- **(B) ML ilk günden parayla:** o zaman Faz 0'da ML hedefini execution'a göre
  yeniden tanımlama işini *önce* yapmamız gerekir → ilk ışık biraz gecikir ama 5
  hesabın beşi de baştan parayla yarışır.

Bu teknik değil, **senin tercih** meselesi; aşağıda sana soruyorum.

### 5) Bütün bu titizlik aslında senin İLK derdinin çözümü

Şunu da bağlayayım: sen bana "güzel sinyal üretemiyor, kullanamıyorum, takip
edemiyorum" demiştin. Bu turlardaki her doğruluk maddesi (look-ahead'siz fill,
intraday'i istatistiğe katmama, kalibre ML ölçümü, corporate-action koruması) tek bir
şeye hizmet ediyor: **sana sahte/şişik kâr göstermeyen, dolayısıyla güvenebileceğin
bir sistem.** Çünkü şişik P&L gösteren bir bot er geç "bu çalışmıyor / güvenemiyorum"
hissini verir — yani senin baştaki şikâyetinin ta kendisini. Codex'in titizliği
akademik bir lüks değil; **kullanılabilirliğin önkoşulu.** Bu yüzden direnmeden alıyorum.

### 6) İki tur sonrası birleşik nihai durum (güncel kararlar)

| Konu | Nihai karar (yakınsanmış) |
|------|---------------------------|
| Plan durumu | **Taslak — teknik onay bekliyor** |
| Fill | Sonraki geçerli seans açılışı (`next_valid_session_open`); aynı kapanış fill YOK |
| İsimlendirme | `signal_session` / `scheduled_fill_session`; takas ayrı (`unsettled_cash`) |
| Cron | İki iş: kapanış-sonrası karar + sonraki-seans fill (retry'li, hangi barı kullandığını kaydeder) |
| Stop/hedef | Gün-içi OHLC (Low≤stop / High≥target) + gap + ambiguous-bar muhafazakâr |
| Equity | Resmî = günlük kapanış (istatistik buradan); intraday = `estimated`, yalnız gösterim |
| p_up | Yön sözleşmesi (`karar_yönü · 2(p_up−0.5) · ağırlık`); kalite kapısını geçmezse katkı 0 |
| Kalite ölçümü | `probability.py` kalibrasyon-sonrası dokunulmamış pencere + Brier Skill Score; eşikler konfigte |
| ML hesabı | **V1'de OBSERVER** (kullanıcı B derse: Faz 0'da hedef hizalanır, parayla girer) |
| Kelly | Çeyrek Kelly (kod gerçeği); ½ istenirse ayrı ablation profili |
| Komisyon/slippage | Tüm arenada ortak, `execution_config`'te; slippage önce sabit, sonra ADV |
| Şema | **7 tablo** (season/accounts/runs/orders[+fill alanları]/cash_ledger/positions/equity); para `numeric`/`Decimal`; SQL önce onaya |
| Idempotency | `run_id` + unique (season+hesap+signal_session+ticker+side+strategy_version); fill+cash+pozisyon tek RPC transaction |
| Kill switch | Profilden bağımsız; + corporate-action `SUSPENDED_DATA_REVIEW` + global PAUSE |
| Kıyas | Lig (ürün, V1) + Laboratuvar (tek-değişkenli; motor desteği baştan, UI sonra) |
| season | İlk günden `arena_seasons` + her tabloda `season_id` (hafif) |
| Replay/survivorship | Forward arena blocker değil; replay güncel evrenle ise survivorship uyarısı göster; point-in-time sonra |
| Fon | Ayrı sezon/faz; ortak başlangıçlı replay olmadan lige sokma |
| RNN/CNN | En son; önce klasik pooled model kalıcı OOS edge göstermeli |

### 7) Üzerinde anlaştığımız uygulama sırası (kesişim)

1. **Faz 0 — Sinyal & veri sözleşmesi:** `p_up` yön bug'ını testle düzelt ·
   `probability.py` kalite ölçümünü (kalibrasyon-sonrası, BSS) düzelt · makro şoku tek
   analiz yoluna bağla · tek dondurulmuş EOD snapshot (as-of/model/strateji/evren
   sürümü) · resmî/intraday equity ayrımı · **ML için karar: observer mı, hedef hizala mı**.
2. **Dikey dilim:** Supabase'siz saf execution çekirdeği, **1 hesap + 2 ticker** ile
   `EOD snapshot → pending emir → sonraki açılış fill → atomik nakit/pozisyon → equity`
   zincirini çalışır gör.
3. **Minimal şema:** 7 tabloluk MVP SQL'ini **önce sana göster/onaylat** · `season_id`/
   `run_id`/unique/atomik RPC · tek hesapla uçtan uca doğrula → 4 kural hesabını seed et
   (ML observer/parayla, kararına göre).
4. **Replay & shadow:** kısa historical replay (survivorship uyarılı) + duplicate/stale/
   gap/ambiguous/negatif-kasa/corporate-action testleri → birkaç gerçek seans yalnız
   pending emir (fill'siz) gözlem.
5. **Canlı arena V1:** atomik fill aç · UI (lig + hesap detayı + resmî/tahmini equity +
   emir günlüğü + CSV) · tamamlanmış-run Telegram.
6. **Araştırma olgunluğu:** point-in-time evren · lab UI · ADV slippage · pooled ML +
   champion/challenger · (şart sağlanırsa) TCN/1D-CNN ayrı kol · fon motoru + sezon.

### Son söz

Codex'in 2. turu **dürüst, yapıcı ve büyük ölçüde benimle aynı yerde.** "Codex mantıklı
mı?" → kesinlikle evet; üstelik bu turda benim eksiğimi (intraday equity'nin
istatistiği şişirmesi, dikey dilim fikri) kapattı. Benim son katkım iki şey: (1)
doğrulanmış üçüncü kod sorununu kayda geçirmek, (2) "ML observer" kararının aslında
*senin* vermen gereken bir tercih olduğunu masaya koymak — çünkü bu, hayalini kurduğun
"ML parayla yarışsın" beklentisini doğrudan etkiliyor.

Plan artık **kararlı ve inşa edilebilir.** Sıradaki somut iş ya Faz 0 düzeltmeleri
(ki `p_up` ve `probability.py` zaten *canlı* sistemini etkileyen gerçek hatalar) ya da
7 tabloluk MVP SQL'ini onayına sunmak. Ve §4'teki ML kararını senden bekliyorum.

---

## Claude — 2026-06-29 00:58 (TR) · Faz 0 detaylı planı (KOD YOK, yalnız tasarım)

> **Önemli:** Bu bölüm **plandır**, kod değildir. Kullanıcı "henüz kodlama yok,
> düşünme adımı bitmedi" dedi — doğru uyarı. Aşağısı, "başla" denildiğinde *ne*
> yapacağımın spec'i + yorumum. Hiçbir kaynak dosyaya dokunulmadı.

Bu turda iki karar kilitlendi:
- **ML-öncelikli hesap → `OBSERVER`** (V1'de para harcamaz; tahmin + karne biriktirir,
  kalite + hedef-uyumu kanıtlanınca sonraki sezon parayla girer).
- **Sıradaki adım → Faz 0** (ama bu "şimdi kodla" değil; Faz 0'ın *ne olduğunu* burada
  netleştiriyoruz, kodlama ayrı bir "başla" onayına bağlı).

Faz 0'ın amacı tek cümle: **arena tek satır işlem açmadan önce, sinyal motorunun
çıktısı her çağrı yolunda aynı, yönü doğru ve kalitesi dürüst ölçülmüş olmalı.** Altı
maddesi var; ikisi *doğrulanmış canlı bug*, dördü *sözleşme/altyapı*.

### F0.1 — `p_up` yön düzeltmesi (DOĞRULANMIŞ CANLI BUG)

**Nerede:** `tradingagents/analysis/confidence.py`, 108–112. satır.
**Şu an ne yapıyor:**
```python
conviction = abs(p_up - 0.5) * 2     # yön bilgisini siliyor
score += conviction * 12
```
`abs()` yüzünden model kararın *tersini* güçlü söylese bile güven artıyor (örnek:
karar AL, p_up=0.10 → +9.6).

**Planlanan sözleşme (Codex'in formülü):**
```text
model_edge   = 2 × (p_up − 0.5)              # −1 … +1  (+ = yukarı, − = aşağı)
decision_dir = +1 (AL/GÜÇLÜ AL) | −1 (SAT/KAÇIN) | 0 (nötr)
katkı        = decision_dir × model_edge × 12
kalite kapısı geçilmediyse → katkı = 0 (model güveni yapay şişirmez)
```
**Beklenen davranış (test edilecek):**

| Karar | p_up | model_edge | katkı | doğru mu? |
|-------|------|-----------|-------|-----------|
| AL | 0.90 | +0.8 | **+9.6** | ✅ model teyit ediyor |
| AL | 0.10 | −0.8 | **−9.6** | ✅ model çelişiyor, güven düşmeli |
| SAT | 0.10 | −0.8 | **+9.6** | ✅ model SAT'ı teyit |
| SAT | 0.90 | +0.8 | **−9.6** | ✅ model SAT'a çelişiyor |
| TUT/İZLE | herhangi | — | **0** | ✅ nötrde şişirme yok |

**Yorumum:** Bu sadece arena için değil — **şu an çalışan saatlik cron'un confidence
skorunu ve Supabase snapshot'larını da düzeltir.** Yani Faz 0'ın bu maddesi, paper
motoru hiç yazılmasa bile mevcut sistemine değer katan bir bugfix. Ölçek değişmiyor
(katkı yine ±12 aralığında), sadece işaret doğru yere oturuyor.
**Dikkat / açık iş:** `unified_confidence`'a şu an bir "kalite bayrağı" gelmiyor;
`p_up`'ı `analyze_ticker` doğrudan `model_cache`'ten besliyor. "Kalite kapısını
geçmediyse katkı 0" şartını uygulamak için ya yeni opsiyonel parametre
(`p_up_quality_ok=True`) ekleyeceğiz ya da çağıran taraf yalnız kaliteli `p_up`
geçecek. Kodlamadan önce bu küçük arayüz kararını netleştirmek lazım.
**Mevcut test riski:** `unified_confidence`'ın `p_up` davranışını sabitleyen bir test
varsa güncellenecek (taramayı henüz bitirmedim; kodlama öncesi `tests/` içinde
`p_up`/`conviction` geçen testleri tek tek kontrol edeceğim).

### F0.2 — `probability.py` kalite ölçümü (DOĞRULANMIŞ METODOLOJİ SORUNU)

**Nerede:** `tradingagents/analytics/probability.py`.
**Sorun:** Brier, isotonic kalibrasyon *öncesi* ham OOS olasılıklarda hesaplanıyor
(81. satır) ve isotonic *aynı* OOS havuzunda fit ediliyor (87. satır). Yani raporlanan
Brier, döndürülen **kalibre** olasılığın dokunulmamış-test güvenilirliği değil; ham
olasılığın güvenilirliği. Docstring'in iddiası ile hesap örtüşmüyor.

**Planlanan düzeltme:**
- **İç içe zaman bölmesi (nested time split):** OOS tahminlerini zaman sırasına göre
  ikiye ayır → erken pencerede isotonic *fit*, geç (dokunulmamış) pencerede **kalibre**
  Brier ölç. Böylece raporlanan sayı gerçekten döndürülen olasılığın performansı olur.
- **Brier Skill Score (BSS):** mutlak Brier yerine `BSS = 1 − brier_model / brier_naive`
  (naive = taban oranı). "0.18 Brier" tek başına anlamsız; BSS "naive'den ne kadar iyi"
  der.
- **Sürümleme:** `model_version`, `trained_until`, `prediction_asof` ve metrik
  penceresini sakla; eşikler (skill>0, AUC, BSS) konfigürasyonda, koda gömülü değil.

**Yorumum:** Bu sayı, arena ML kalite kapısının ve (sonra) observer→para geçişinin
okuyacağı sayı. Düzeltilmeden gate yanlış kararla açılır/kapanır. **Ama dikkat:** bu
değişiklik gecelik model çıktısını (`model_cache.brier/auc`) ve raporlanan rakamları
*değiştirir* → eski snapshot'larla birebir kıyas kesintiye uğrar (bunu sürümleme ile
işaretleyeceğiz). Bir de veri kıtlığı: zaten n≥320 / OOS≥30 sınırı dar; iç içe bölme
örneği daha da azaltır → minimum eşikleri yeniden ayarlayıp, yetmezse "kalibrasyon
ölçülemedi" diye graceful döneceğiz (asla istisna fırlatma ilkesi korunur).

### F0.3 — Makro şoku tek analiz yoluna bağla

Saatlik cron, manuel tarama ve (gelecekteki) günlük paper motoru **aynı orchestrator**
(`analyze_ticker`/`analyze_universe`) yolundan geçmeli ki makro-şok kapısı her yerde
aynı uygulansın. **Yorum:** Bugün `macro_shock` saatlik akışta hesaplanıyor; paper
motoru ayrı bir yol kurarsa kapı atlanabilir. Tek yol = tek davranış.

### F0.4 — Tek dondurulmuş EOD snapshot + sürümleme

Kapanış sonrası evren **bir kez** analiz edilip dondurulacak; tüm hesaplar (lig'in
adaleti için) **aynı** snapshot'ı tüketecek. Snapshot'a eklenecek alanlar:
`signal_asof`, `last_bar`, `strategy_version`, `model_version`, `health`. **Yorum:**
Bu, senin "hangisi kazandırıyor" yarışının adil olmasının teknik şartı — her hesap
kendi yfinance çağrısını yaparsa veri/zaman kayar, yarış bozulur.

### F0.5 — Resmî EOD equity ≠ intraday tahmini

İlke kaydı (kod paper motorunda): **resmî** equity günlük kapanışta (Sharpe/drawdown/
CAGR buradan); **intraday** saatlik değer yalnız ekranda `estimated` etiketiyle,
istatistiğe ve model eğitimine *girmez*. **Yorum:** Benim önceki "saatlik
mark-to-market" fikrimin Codex'le düzeltilmiş hâli; aksi halde aynı günlük hareket
saatlik tekrarlarla örnek sayısını şişirir.

### F0.6 — ML kararı: OBSERVER (KİLİTLENDİ)

ML-öncelikli hesap V1'de para harcamaz; `paper_accounts.status = 'OBSERVER'`. Tahmin
ve gerçekleşen sonuçları biriktirir. **Yorum:** Bu sayede "ML kuralları yendi mi?"
sorusunu, *yanlış tanımlı hedefle para yakmadan*, dürüst karneyle bekletiyoruz.

### Faz 0 çıkış kriteri (Codex'le ortak)

> Aynı veri snapshot'ı + aynı profil ayarıyla tekrar çalıştırıldığında **aynı planlanan
> emirler** oluşmalı; **kalitesiz/bayat model hiçbir otomatik işleme katkı vermemeli.**

### Faz 0 için düşündüğüm test listesi (kodlama gelince yazılacak)

- `p_up` yön: AL+yüksek→güven artar · AL+düşük→azalır · SAT'ta tersi · nötr→etki 0 ·
  kalite kapısı yoksa katkı 0.
- `probability`: kalibre Brier **dokunulmamış** pencerede ölçülüyor · BSS hesaplanıyor ·
  az veride graceful "ölçülemedi".
- Makro şok: üç çağrı yolunda da aynı kapı tetikleniyor.
- Snapshot: tek dondurulmuş çıktı, iki hesap aynı `signal_asof`'u görüyor.

### Yorum: neden Faz 0 "kodla" demeden önce konuşulmalı

İki açık arayüz kararı var ve bunlar kodlama başlamadan netleşmeli:
1. **Kalite bayrağı `unified_confidence`'a nasıl girecek?** (yeni parametre mi, çağıran
   filtresi mi)
2. **`probability.py` düzeltmesi `model_cache` sayılarını değiştireceği için**, eski
   gecelik çıktılarla kıyas kopacak — sürümleme bunu yönetir ama "ne zaman migrate
   ediyoruz" kararı sende.

Bunlar "düşünme adımı"nın hâlâ açık olan parçaları. Benden sonra istersen bunu yine
Codex'e okutursun; ya da "şu iki arayüz kararını şöyle ver, sonra kodla" dersin. **Ben
kendiliğimden koda geçmiyorum** — bir önceki turda erken davrandım, bu sefer net:
karar senin.

---

## Codex — 2026-06-29 01:05 (TR) · Acımasız nihai karar ve kaynaklı plan

> **Bu turun amacı:** Artık yeni fikir ekleyip planı sonsuza kadar büyütmek değil;
> tartışmayı kapatmak, yanlışları açıkça elemek ve uygulanacak V1 sözleşmesini
> dondurmak. Aşağıdaki kararlar hem mevcut kod incelemesine hem resmî araç/borsa
> dokümantasyonuna hem de seçilmiş akademik çalışmalara dayanıyor. Paper-trading'in
> gerçek piyasanın kusursuz kopyası olmadığı özellikle kabul ediliyor.

Abi bu sefer en başta hükmü vereyim: **Plan koşullu olarak GO.** Daha fazla genel
mimari tartışmasına ihtiyacımız yok. Claude'un 00:52 yakınsaması ve 00:58 Faz 0
taslağı genel olarak iyi; önceki iki turda bulduğumuz ana risklerin çoğunu doğru
biçimde içeri almış. Fakat Faz 0 metninde bir gerçek kod hatası, bir metodoloji
tercihi ve birkaç kapsam bulanıklığı hâlâ var. Bunları aşağıda düzeltiyorum; sonra
V1'i donduruyorum.

### Claude'un son ekinde katıldığım şeyler

Şunlara artık şerhsiz katılıyorum:

- ML hesabının V1'de **OBSERVER** olması doğru. Yanlış hedefle para yarıştırmaktan
  iyidir; kullanıcıya erken ekran göstermeyi de geciktirmez.
- `p_up` yön bug'ı gerçek ve arena öncesi düzeltilmelidir.
- Kalibrasyon metriğinin mevcut döndürülen olasılığı dürüstçe ölçmediği teşhisi
  doğrudur.
- Tek dondurulmuş EOD snapshot, bütün profiller için adil yarışın önkoşuludur.
- Sonraki geçerli seans açılışı, aynı kapanış fill'inden daha dürüsttür.
- Karar ve fill iki farklı mantıksal iştir.
- Resmî EOD equity ile intraday tahmini equity ayrılmalıdır.
- Basit `season_id`, idempotent `run_id`, unique constraint ve atomik RPC ilk
  sürümden bulunmalıdır.
- Dikey dilim yaklaşımı doğru: çekirdek gevşetilmez, ilk gösterilen evren küçültülür.
- Corporate-action/veri sıçraması için en azından sert durdurma V1'de bulunmalıdır.
- Yedi tabloluk sade MVP, üç tablo ile dokuz tablo arasında makul dengedir.

Bu kararların neden doğru olduğunu yalnız "biz öyle hissettik" diye söylemiyoruz:

- Zaman sıralı veride klasik rastgele CV geleceği geçmişe sızdırabilir;
  scikit-learn de zaman sıralı örneklerde `TimeSeriesSplit` kullanılmasının nedenini
  açıkça böyle anlatıyor
  ([scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)).
- QuantConnect'in fill modeli, order timestamp'iyle aynı veya bayat veriden market
  open fill üretmeyerek look-ahead'i engelliyor; trade bar için alışta
  `open + slippage`, satışta `open - slippage` kullanıyor
  ([Latest Price Fill Model](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/supported-models/latest-price-model)).
- Postgres unique constraint'leri bir kolon kombinasyonunun ikinci kez yazılmasını
  veritabanı seviyesinde engeller; duplicate cron koruması yalnız Python'daki `if`
  kontrolüne bırakılamaz
  ([PostgreSQL Constraints](https://www.postgresql.org/docs/current/ddl-constraints.html)).
- Supabase, Postgres fonksiyonlarını API üzerinden çağırmayı destekliyor; fill,
  nakit ve pozisyon güncellemesini tek DB fonksiyonunda toplama kararı platformun
  doğal yeteneğiyle uyumlu
  ([Supabase Database Functions](https://supabase.com/docs/guides/database/functions)).

### Claude'un Faz 0 metnindeki somut hata

Claude, F0.3 yorumunda "Bugün `macro_shock` saatlik akışta hesaplanıyor" demiş.
**Bu doğru değil.** Mevcut kodda:

- `scripts/run_hourly_analysis.py` yalnız `analysis_run.index_regime()` çağırıyor.
- Hisse bazında `analyze_ticker(tk, regime_state=..., p_up=...)` çağrılıyor.
- `macro_shock` verilmediği için `analyze_ticker` içindeki varsayılan `False` kalıyor.
- Makro şoku otomatik hesaplayan yol `analyze_universe()`; saatlik script bugün bu
  yolu kullanmıyor.
- Manuel scanner da doğrudan `analyze_ticker(tk)` çağırdığı için aynı kapıyı atlıyor.

Dolayısıyla F0.3 bir "paper motoru ileride atlamasın" önlemi değil; **mevcut saatlik
ve manuel akışta zaten atlanan canlı bir entegrasyon boşluğunun düzeltilmesidir.**
Bu, Faz 0 blocker'ı olarak kalıyor ve üç çağrı yolunun tek orchestrator'a alınması
gerekiyor.

### Kalibrasyon konusunda nihai teknik karar

Claude'un "nested time split + isotonic" yönü, aynı veride fit edip aynı veride
ölçmekten daha doğru; ama per-ticker veri miktarı için **isotonic'i varsayılan yapmak
hâlâ savunulamaz.** scikit-learn, kalibrasyon örneği yaklaşık 1.000'in belirgin
altındaysa isotonic yöntemin overfit etmeye yatkın olduğunu açıkça söylüyor; küçük
örneklerde sigmoid/Platt yaklaşımı daha uygun olabilir
([Probability calibration](https://scikit-learn.org/stable/modules/calibration.html),
[CalibratedClassifierCV](https://scikit-learn.org/1.4/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)).

Bizim per-ticker veri setinde uzun SMA ısınması, train penceresi, calibration penceresi,
dokunulmamış test penceresi ve 10 günlük horizon sonrası isotonic'e kalan bağımsız
örnek sayısı çoğu hissede 1.000'in altında olacaktır. Bu yüzden karar:

1. **Per-ticker V1 kalibratörü varsayılan olarak sigmoid/Platt olacak.**
2. Zaman sırası korunacak; rastgele `KFold` kullanılmayacak.
3. Dış walk-forward test penceresi kalibratör ve base model tarafından görülmeyecek.
4. Ham ve kalibre Brier ayrı raporlanacak.
5. Naif sınıf oranına karşı Brier Skill Score raporlanacak.
6. AUC, BSS, skill, örnek sayısı ve kayan-son-pencere kalitesi birlikte tutulacak.
7. Isotonic yalnız pooled modelde veya gerçekten yeterli calibration örneği varsa
   challenger olarak denenecek; otomatik olarak "daha gelişmiş" sayılmayacak.
8. Kalibrasyon için veri yetmiyorsa olasılık uydurulmayacak; `quality_passed=false`
   ve ML katkısı sıfır olacak.

Bu noktada akademik görünmek için nested kelimesini yazmak yetmez. Zaman sırası,
kalibrasyon/test ayrımı ve örnek sayısı gerçekten uygulanmazsa model yalnız daha
sofistike biçimde overfit olur.

### Model kalite bayrağının arayüz kararı

Claude iki seçenek bırakmış: `unified_confidence` fonksiyonuna
`p_up_quality_ok=True` eklemek veya çağıranın yalnız kaliteli `p_up` geçmesi. Nihai
kararım: **çıplak boolean eklemiyoruz.** Boolean'ın yanlış varsayılanı ileride aynı
bug'ı başka isimle geri getirir.

Şu sözleşme kullanılacak:

```text
ModelEvidence
  p_up
  model_version
  prediction_asof
  trained_until
  n_train / n_calibration / n_test
  auc_raw
  brier_raw
  brier_calibrated
  brier_skill_score
  recent_skill
  quality_passed
  rejection_reasons[]
```

- Kaliteyi ayrı saf fonksiyon `validate_model_evidence(...)` belirler.
- Orchestrator, `quality_passed=false` ise `unified_confidence`'a `p_up=None` geçirir.
- Reddedilen modelin metrikleri ve nedenleri yine snapshot'a yazılır; veri kaybolmaz.
- `unified_confidence` yalnız yönlü katkı hesabından sorumlu kalır.
- `quality_passed` varsayılanı hiçbir zaman `True` olmaz.

Bu ayrım hem testi kolaylaştırır hem de "model kullanılmadı çünkü bayattı mı, AUC
zayıftı mı, calibration ölçülemedi mi?" sorusunu cevaplar.

### Akademi gerekli, fakat borsa akademik bir tablo değil

Burada senin soruna doğrudan cevap vereyim: **Evet, bilimsel olarak güzel bir yöntem
gerçek alım-satımda tek başına gerçekçi olmayabilir.** Bunun birkaç nedeni var:

- Backtest çok sayıda varyant arasından en iyiyi seçtiğinde şişer. Deflated Sharpe
  Ratio, çoklu deneme/selection bias ve non-normal getiriyi düzeltmeye çalışır; bu
  yararlıdır ama veri sızıntısını, yanlış fill'i veya olmayan likiditeyi düzeltmez
  ([Bailey & López de Prado — Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).
- PBO/CSCV, denenen konfigürasyonlar arasındaki overfit ihtimalini ölçmeye çalışır;
  kötü veri veya yanlış execution sözleşmesini iyi hâle getirmez
  ([Bailey, Borwein, López de Prado & Zhu — Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322)).
- Gerçek işlem verisine dayanan araştırma, yatırımcı için önemli olanın gross değil
  **işlem maliyeti sonrası net getiri** olduğunu; maliyet ve kapasitenin stratejiye
  göre değiştiğini vurgular
  ([Frazzini, Israel & Moskowitz — Trading Costs of Asset Pricing Anomalies](https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini%2C%20Israel%20and%20Moskowitz.pdf)).
- Paper broker işleten Alpaca bile simülasyonun market impact, emir bilgisinin
  piyasaya sızması, gecikme kaynaklı slippage, limit emir kuyruk pozisyonu, fiyat
  iyileştirmesi, bazı ücretler ve temettüyü hesaba katmadığını açıkça yazıyor
  ([Alpaca Paper vs Live](https://docs.alpaca.markets/us/docs/paper-trading)).

O yüzden bu arenanın etiketi **"gerçek para kazanma kanıtı" değil, "strateji eleme
ve operasyon doğrulama ortamı"** olacak. Paper'da kötü olan stratejiyi eleriz;
paper'da iyi olanı ise ancak daha sıkı ileri test için aday yaparız. Paper kârını
gelecek kâr vaadi gibi göstermeyiz.

### BIST açılış fiyatı hakkında gerçekçi karar

Borsa İstanbul prosedürü "Opening Auction Price" ile gün içindeki ilk işlemin
"Opening Price" bilgisini ayrı kavramlar olarak yayımlıyor
([Borsa İstanbul Equity Market Procedure](https://borsaistanbul.com/files/equity-market-procedure.pdf)).
Dolayısıyla yfinance günlük `Open` alanına bakıp buna otomatik olarak "resmî açılış
müzayedesi fill'i" diyemeyiz.

V1 sözleşmesi:

- Kullanılan fiyat `yfinance_daily_open_proxy` olarak açıkça etiketlenecek.
- Bunun resmî açılış müzayedesi fiyatı olduğu iddia edilmeyecek.
- Alış fill'i `open_proxy + slippage`, satış fill'i `open_proxy - slippage` olacak.
- Açılış verisi yoksa, hacim sıfırsa, fiyat bayatsa veya corporate-action kontrolü
  takılmışsa emir doldurulmayacak.
- Kasa fiyat gap'i sonrası yetmiyorsa emir miktarı sessizce eksi kasaya sokulmayacak;
  ortak bir buying-power buffer uygulanacak ve gerekirse emir reddedilecek.
- BIST100/az likit evrende "tam fill garantisi" verilmeyecek. İleride resmî/intraday
  veri kaynağı geldiğinde ayrı fill modeli sürümü açılacak.

QuantConnect de market-on-open emrinin fiyatının açılıştan önce bilinmediğini, gap
nedeniyle buying power yetmeyebileceğini ve emrin reddedilebileceğini belirtiyor
([Market On Open Orders](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/order-types/market-on-open-orders)).
Bu yüzden başlangıç bütçesinin yüzde yüzünü emirlerde tüketmek gerçekçi değildir.

### GitHub Actions konusunda romantik olmayalım

GitHub Actions bu paper sistem için ucuz ve yeterli bir worker olabilir; fakat saat
garantili piyasa altyapısı değildir. GitHub'ın kendi dokümanı scheduled işlerin yoğun
zamanlarda gecikebildiğini, saat başlarında yükün arttığını ve bazı queued işlerin
düşebileceğini söylüyor
([GitHub Actions scheduled workflows](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows)).

Nihai operasyon kararı:

- Cron'lar tam saat `:00` yerine örneğin `:07`, `:17` gibi dakikalara konur.
- Karar işi ve fill reconciliation işi idempotent olur; "tam saatinde koştu"ya
  güvenmez.
- Bir run kaçırılırsa sonraki run eksik session'ı bulup tamamlayabilir.
- Emirde `scheduled_fill_session`; run'da gerçek `started_at/completed_at` saklanır.
- Paper fill, job'ın çalıştığı duvar saatinden değil hedef seansın kayıtlı open
  proxy'sinden üretilir.
- İki ardışık başarısızlık veya bayat session varsa yeni emir durur ve Telegram hata
  mesajı gönderilir.

Bu düzen paper için yeterlidir. Gerçek broker emri düşünülürse GitHub cron worker'ı
artık yeterli sayılmayacak; sürekli çalışan, gözlemlenebilir bir execution servisi
gerekecek.

### DSR ve PBO hakkında acı gerçek

Projede DSR/PBO fonksiyonlarının bulunması güzel ama bugün UI'da görünmeleri, gerçek
arena stratejisinin doğrulandığı anlamına gelmiyor:

- Mevcut DSR esas olarak Dip-Al stratejisinin getiri serisine uygulanıyor.
- Mevcut PBO ekranı MA kesişim konfigürasyon ızgarasını ölçüyor.
- Arena ise temel + teknik + confidence gate + rejim + sizing + stop + execution
  maliyetlerinden oluşacak bambaşka bir strateji.

Bu nedenle **mevcut DSR/PBO rakamları arena hesabına kalite rozeti olarak
taşınmayacak.** Arena replay'i kendi günlük net getiri matrisini üretmeden "DSR geçti"
veya "PBO düşük" denmeyecek. Ayrıca hangi parametre denemelerinin yapıldığı ve kaç
varyant arasından seçim yapıldığı kaydedilecek; DSR'deki deneme sayısını keyfî `20`
vermek bilimsel doğrulama değildir.

Bu karar, akademiyi küçümsemek değil; akademik aracın doğru nesneye uygulanmasını
istemektir.

### Nihai V1 kapsamı — artık değiştirilmeyecek kararlar

Bu noktadan sonra V1 için aşağıdaki sözleşmeyi donduruyorum:

| Konu | Nihai V1 kararı |
|------|-----------------|
| Amaç | Gerçek para kârlılığı kanıtı değil; strateji eleme + execution/ledger doğrulama |
| Başlangıç | Her aktif hesap 100.000 TL, aynı `season_id`, aynı başlangıç session'ı |
| Aktif hesaplar | Temkinli, Dengeli, Agresif, Trend-takip — **4 para hesabı** |
| ML hesabı | **OBSERVER**, para harcamaz; tahmin ve gerçekleşen sonuç biriktirir |
| Fon hesabı | V1 dışında; ayrı veri/valör motoru ve yeni sezon |
| Varlık evreni | V1 para hesaplarında **yalnız BIST30** |
| BIST100 | Tarama/gözlem devam eder; paper execution ancak BIST30 motoru kararlı olduktan sonra yeni sezon/ablation |
| Yön | Long-only; short, kaldıraç ve margin yok |
| LLM rolü | Emir yetkisi yok; rapor/açıklama ve ileride opsiyonel veto araştırması |
| Sinyal | Tamamlanmış EOD snapshot, tek kez hesaplanır ve bütün hesaplarla paylaşılır |
| Fill | Sonraki geçerli seansın `yfinance_daily_open_proxy` fiyatı + yönlü slippage |
| Emir | V1'de tam fill ya da reject; partial fill yok |
| Equity | Resmî günlük EOD; intraday yalnız `estimated` gösterim |
| Maliyet | Bütün hesaplarda ortak commission/slippage config; profile göre değişmez |
| Sizing | Çeyrek Kelly yalnız geçerli olasılıkla; aksi hâlde profilde tanımlı sabit risk/weight |
| Güvenlik | Bayat veri, negatif kasa, duplicate emir, corporate action, veri sıçraması ve global pause kapıları kapatılamaz |
| Şema | 7 tabloluk MVP + atomik Postgres RPC + unique constraint |
| Worker | GitHub Actions idempotent/recovery özellikli; tam saat dışında planlanır |
| Bildirim | Yalnız completed run özeti; failure/stale ayrıca alarm |

**BIST100'ü V1 para hesaplarından çıkarıyorum.** Acımasız ama doğru karar bu.
Sebebi BIST100'ün kötü olması değil: aynı anda yeni ledger, yeni fill modeli, dört
profil ve daha zayıf likidite/veri kalitesini devreye almak hata ayıklamayı gereksiz
zorlaştırır. Önce BIST30'da motorun doğru çalıştığını kanıtlarız. BIST100 daha sonra
aynı stratejinin tek-değişkenli evren deneyi olarak girer; o zaman gerçekten
"geniş evren katkı sağladı mı?" diyebiliriz.

### Profil kuralları hakkında son uyarı

Mevcut 68/60/55 güven eşikleri ve pozisyon tavanları bilimsel olarak kanıtlanmış
doğa sabitleri değil; tasarım hipotezleri. Sezon başladıktan sonra kötü sonuç gördük
diye bu eşikler sessizce değiştirilmez.

- Her profil `rules_snapshot` ile kilitlenir.
- Her değişiklik yeni strategy version ve tercihen yeni sezon/ablation üretir.
- Aynı sezon içinde geriye dönük kural değiştirmek yasaktır.
- "En iyi görünen profili" tekrar tekrar ayarlamak selection bias üretir. DSR'nin
  ele almaya çalıştığı sorunlardan biri zaten çoklu deneme sonrası en iyi sonucu
  seçmektir.
- Lig eğlenceli bir ürün görünümüdür; neden-sonuç iddiası laboratuvar testinden gelir.

Kazanan ilanı da acele yapılmaz. Önceden belirlenmiş değerlendirme penceresi ve
minimum tamamlanmış işlem sayısı dolmadan yalnız "mevcut lider" denir; "en iyi
strateji" denmez. Kesin seans/işlem eşikleri replay dağılımını görmeden sihirli sayı
olarak yazılmayacak, sezon başlamadan önce kilitlenecektir.

### Faz 0 için düzeltilmiş nihai görev listesi

Faz 0 artık şu altı teslimattan oluşuyor:

1. **ModelEvidence sözleşmesi ve kalite doğrulayıcı**
   - Kalite metriği/rejection reason tek yerde.
   - Kalitesiz/bayat evidence, confidence'a `p_up=None` olarak gider.
2. **Yönlü model katkısı**
   - AL/SAT yön testi, nötrde sıfır katkı, kalite yoksa sıfır katkı.
3. **Zaman duyarlı kalibrasyon/evaluation**
   - Sigmoid varsayılan; ayrı calibration ve untouched walk-forward test.
   - Raw/calibrated Brier + BSS + AUC + recent skill + örnek sayıları.
4. **Tek analiz orchestrator'ı**
   - Saatlik, manuel ve gelecekteki EOD paper aynı universe yolunu kullanır.
   - Makro şok gerçekten hesaplanır ve bütün yollarda aynı uygulanır.
5. **Sürümlü EOD snapshot sözleşmesi**
   - `signal_session`, `signal_asof`, `last_bar`, `strategy_version`,
     `model_version`, `universe_version`, `health`, model rejection reasons.
6. **Observer karne sözleşmesi**
   - ML tahmininin hangi horizon ve hangi gerçekleşen fiyatla değerlendirileceği
     açıkça kaydedilir; observer sonucu para P&L'i gibi gösterilmez.

Claude'un F0.5 resmî/intraday equity ayrımı doğru bir karardır ama henüz equity
motoru olmadığı için Faz 0'da kod teslimatı değil, paper engine acceptance
criterion'ıdır. Faz 0 dokümanında sözleşme olarak kalır; kodu execution/equity
fazında yazılır. Bu ayrım kapsamı temiz tutar.

### Uygulama sırası — son kez

1. **Faz 0:** Yukarıdaki altı sinyal/veri teslimatı ve testleri.
2. **Dikey dilim:** Tek hesap + iki BIST30 ticker; saf execution motoru ve yerel
   equity çıktısı.
3. **MVP SQL onayı:** 7 tablo, RLS, unique constraint, atomik fill RPC.
4. **Tek hesap entegrasyonu:** Supabase'e pending → fill → ledger → position → EOD
   equity zinciri.
5. **Kısa replay:** Aynı engine ile gap/stop/maliyet/idempotency ve veri sıçraması
   doğrulaması; bugünkü evren kullanılırsa survivorship uyarısı.
6. **Shadow:** Gerçek seanslarda pending emir üret, para/pozisyon değiştirme;
   scheduled session ve open proxy eşleşmesini izle.
7. **Arena V1:** Dört BIST30 para hesabı + bir ML observer, UI/CSV/Telegram.
8. **V1 sonrası:** BIST100 ablation sezonu, lab UI, gelişmiş slippage, point-in-time
   evren, ML target hizalama/activation.
9. **Daha sonra:** Fon motoru ve ancak klasik ML kalıcı OOS edge gösterirse derin
   model araştırması.

Bu sıra artık yeterince güvenli ve yeterince küçüktür. Bundan sonra aynı planı başka
kelimelerle yeniden tartışmak ilerleme değil erteleme olur.

### Kesin NO-GO koşulları

Aşağıdakilerden biri varsa canlı paper fill açılmaz:

- Aynı kapanıştan fill üreten herhangi bir yol kalmışsa.
- Saatlik/manuel/EOD yolları farklı makro veya model gate davranışı gösteriyorsa.
- Kalibrasyon test penceresi model/kalibratör tarafından görülmüşse.
- Isotonic küçük per-ticker örnekte varsayılan kalmışsa.
- Duplicate workflow aynı emri veya nakit hareketini iki kez üretebiliyorsa.
- Fill + ledger + position tek transaction değilse.
- Açılış fiyat kaynağı/as-of bilgisi kaydedilmiyorsa.
- Resmî EOD ile intraday tahmin aynı performans serisine yazılıyorsa.
- Corporate-action/veri sıçramasında emir otomatik devam ediyorsa.
- Profil ayarları sezon ortasında geçmişi değiştirecek biçimde güncellenebiliyorsa.
- Public Streamlit arayüzü service-role yetkisiyle kimlik doğrulamasız mutasyon
  yapabiliyorsa.

Son madde önceki turlarda yeterince öne çıkmadı: mevcut tek-kullanıcılı RLS yapısı
service-role ile bypass ediliyor. Uygulama internete açıksa yalnız UI butonlarına
güvenilemez. Public deploy öncesinde en az uygulama erişim koruması veya gerçek
kullanıcı bazlı auth/RLS gerekir.

### Kaynakça — hangi kararı neden destekliyor?

- [Borsa İstanbul Equity Market Procedure (2026)](https://borsaistanbul.com/files/equity-market-procedure.pdf)
  — opening auction price, opening price, seans ve devre kesici gerçeklerinin resmî
  kaynağı; günlük `Open` alanının neyi temsil ettiğini varsaymamamız gerektiğini
  destekliyor.
- [QuantConnect — Market On Open Orders](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/order-types/market-on-open-orders)
  — açılış fill fiyatının önceden bilinmediğini, gap ve buying-power reddi olabileceğini
  gösteriyor.
- [QuantConnect — Latest Price Fill Model](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/supported-models/latest-price-model)
  — stale/same-timestamp veriden fill üretmeme ve yönlü slippage yaklaşımına pratik
  referans.
- [Alpaca — Paper Trading / Paper vs Live](https://docs.alpaca.markets/us/docs/paper-trading)
  — paper sonuçlarının market impact, latency, queue, dividend ve başka gerçek piyasa
  unsurlarını tam modellemediğini açıkça belirtiyor.
- [scikit-learn — TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
  — zaman sıralı veride gelecekle geçmişi karıştırmayan validation ihtiyacı.
- [scikit-learn — Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
  — küçük calibration örneğinde sigmoid/isotonic trade-off'u ve isotonic overfit
  riskinin kaynağı.
- [Bailey & López de Prado — The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
  — selection bias, çoklu deneme ve non-normal getiri nedeniyle Sharpe şişmesini
  düzeltme motivasyonu.
- [Bailey, Borwein, López de Prado & Zhu — The Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322)
  — çok sayıda varyant arasından seçilen stratejinin overfit olma ihtimali ve PBO
  yaklaşımı.
- [Frazzini, Israel & Moskowitz — Trading Costs of Asset Pricing Anomalies](https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini%2C%20Israel%20and%20Moskowitz.pdf)
  — canlı işlemlerden maliyet, market impact, kapasite ve net-getiri gerçekliği.
- [Brown, Goetzmann, Ibbotson & Ross — Survivorship Bias in Performance Studies](https://terpconnect.umd.edu/~wermers/ftpsite/FAME/Brown_Goetzmann_Ibbotson_Ross.pdf)
  — yalnız hayatta kalan varlıklarla yapılan geçmiş analizin sahte öngörülebilirlik
  yaratabilmesi.
- [PostgreSQL — Constraints](https://www.postgresql.org/docs/current/ddl-constraints.html)
  — composite unique constraint ile duplicate emir/run koruması.
- [Supabase — Database Functions](https://supabase.com/docs/guides/database/functions)
  — atomik ledger mantığını Postgres fonksiyonuna/RPC'ye taşımanın platform desteği.
- [GitHub Actions — Scheduled workflow gecikmeleri](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows)
  — cron gecikmesi/düşmesi ihtimali; recovery ve idempotency gereğinin operasyonel
  kaynağı.

### Nihai söz

Claude'un son planı kötü değil; aksine tartışmayı uygulanabilir seviyeye taşıdı.
Katıldığım bölümleri yukarıda açıkça aldım. Yanlış olan makro-şok durumunu düzelttim;
küçük örnekte isotonic varsayımını reddettim; mevcut DSR/PBO'nun arena doğrulaması
sayılmayacağını netleştirdim; BIST100'ü ve paralı ML hesabını V1'den çıkardım.

Acımasız nihai karar şu: **Önce dürüst ve sıkıcı bir BIST30 paper motoru kuracağız.**
Gösterişli RNN, yüz hisselik evren, fon rotasyonu ve LLM trader sonra. İlk sistemin
görevi yüksek kâr göstermek değil; tek bir emrin neden üretildiğini, hangi bilgiyle
hangi fiyattan doldurulduğunu ve her kuruşun kasayı nasıl değiştirdiğini açıklamak.
Bu zincir kusursuz değilse eklenen her "AI" katmanı yalnız hatayı daha pahalı ve daha
anlaşılmaz yapar.

Bu bölümle planlama tartışmasını kapatıyorum. Bir sonraki anlamlı adım yeni bir plan
yazmak değil; kullanıcı onayı geldiğinde Faz 0 testleri ve `ModelEvidence`
sözleşmesidir.

---

## Claude — 2026-06-29 01:13 (TR) · NİHAİ KARAR (acımasız)

> **Bu bölüm planlamayı KAPATIR.** Kullanıcı acımasız olmamı istedi; olacağım —
> acımasızlık burada "kabalık" değil, *hoşa gitmeyen gerçeği saklamamak, kesilmesi
> gerekeni kesmek ve net hüküm vermek* demek. Kararları, Codex gibi, kaynaklara
> dayandırıyorum. Hükmü en sona saklamıyorum, başa koyuyorum.

**HÜKÜM: Plan GO — ama Codex'in donduruşunu olduğu gibi değil, "önce-ucuz-test,
edge-kapısı, 1→4 hesap, kill-criterion" şartlarıyla onaylıyorum. Ve bu projenin
gerçek teslimatı bir kâr eğrisi değil; "sinyallerimin net edge'i var mı?" sorusuna
kanıtlı bir EVET/HAYIR.**

### 1) Önce kendi hatam — macro_shock'ta Codex haklı, ben yanlıştım

Kabul ediyorum, kanıtıyla: `run_hourly_analysis.py:69` `analyze_ticker`'ı
`macro_shock` olmadan çağırıyor, satır 110 yalnız `index_regime()` hesaplıyor,
`scanner_page.py:62` çıplak `analyze_ticker(tk)`. `macro_shock_state()`'i otomatik
çağıran tek yol `analyze_universe()` ve onu **ne saatlik cron ne manuel tarama**
kullanıyor. Yani USDTRY sistemik-şok kapısı **şu an canlıda ölü.** Benim F0.3'teki
"bugün saatlik akışta hesaplanıyor" cümlem yanlıştı. Bu, F0.3'ü "ileride atlamasın"
önleminden **"şu an dürüstçe çalışmayan bir emniyet kapısını dirilt"** seviyesine
yükseltir. Ayrıca **küçük per-ticker örnekte isotonic yerine sigmoid/Platt** kararında
da Codex haklı (scikit-learn küçük örnekte isotonic'in overfit ettiğini açıkça yazıyor);
ben "nested isotonic" derken örnek sayısının isotonic'e uygun olup olmadığını
sorgulamamıştım.

Dürüst skor tahtası: Codex bu turlar boyunca **dört gerçek sorun** çıkardı —
`abs(p_up−0.5)` yön bug'ı, doküman ½-Kelly ↔ kod ¼-Kelly çelişkisi, `probability.py`
kalibrasyon-ölçüm tutarsızlığı, ve şimdi macro_shock entegrasyon boşluğu. Dördü de
gerçek, dördünü de kodda doğruladım. Bu, review'a güvenimi hak ediyor; ben de aynı
acımasızlığı kendime uyguluyorum.

### 2) Codex'in nihai V1 dondurmasını KABUL ediyorum

Şu maddeleri şerhsiz alıyorum: BIST30-only para hesapları · 4 para hesabı + ML
observer · long-only · sonraki seans `open_proxy + yönlü slippage` · tam-fill-ya-reject ·
resmî EOD ≠ intraday equity · ortak maliyet config · çeyrek Kelly yalnız geçerli
olasılıkla · kapatılamaz emniyet kapıları · 7 tablo + atomik RPC + unique · idempotent/
recovery'li worker · `ModelEvidence` sözleşmesi (çıplak boolean yerine) · DSR/PBO'nun
mevcut hâliyle arenaya kalite rozeti olarak taşınmaması · fon/BIST100/RNN'in V1 dışı.
Bunlar doğru ve iyi argümanlı. Tartışmayı tekrar açmak erteleme olur; açmıyorum.

### 3) Şimdi acımasız kısım — Codex'in bile fazla kibar kaldığı üç gerçek

**(a) Asıl teslimat "çalışan bot" değil. Senin kök derdin hâlâ çözülmedi.**
Sen bana en başta "güzel sinyal üretemiyor" dedin. Bütün bu 2000 satır — ledger, RPC,
sezon, atomik fill, kill switch — **o sorunu çözmüyor.** Hiçbir execution titizliği
edge'siz bir sinyali kâra çeviremez; sadece sonucu *dürüstçe ölçer.* Ve dürüst ölçümün
en olası cevabı şu: **bu sinyallerin maliyet-sonrası net edge'i muhtemelen yok.**
Bireysel yatırımcıların aktif işlemde maliyet sonrası piyasanın altında kaldığı
ampirik olarak güçlü biçimde gösterilmiştir (Barber & Odean, *Trading Is Hazardous to
Your Wealth*, Journal of Finance 2000). Buna duygusal olarak hazır ol; çünkü sistemi
"doğru" kurmak, sana hoş olmayan bu cevabı **net** verecek.

**(b) En pahalı altyapıyı kurmadan önce en ucuz testi yap — fail cheap.**
Codex'in sırası altyapıyı öne, edge sorusunu (replay) 5. adıma koyuyor: 7 tablo, RPC,
cron, UI… *sonra* "acaba kâr var mıydı?". Bunu **tersine çeviriyorum.** Faz 0'dan
sonra ilk iş **historical replay = edge kapısı**: Supabase yok, cron yok, UI yok —
yalnız mevcut sinyaller + gerçekçi maliyet + sonraki-seans fill, equity vs XU100,
ve PBO/DSR **arenanın kendi getiri serisine** (Dip-Al'a değil). Çünkü çoğu backtest,
çok sayıda varyant arasından en iyiyi seçtiği için şişer ve out-of-sample çöker
(Bailey, Borwein, López de Prado & Zhu, *Pseudo-Mathematics and Financial
Charlatanism*, Notices of the AMS 2014). Execution çekirdeği replay ve canlıda zaten
ortak (Codex de böyle dedi) → **çekirdeği yaz, replay'i koştur, KARAR VER.** Replay
hiçbir profilde XU100'ü net geçemiyorsa, 7 tabloluk kaleyi **hiç kurmayız.** Haftalarca
ledger cilalayıp sonunda "zaten edge yokmuş" demek, kaynak israfının ta kendisidir.

**(c) Akademik cila en tehlikeli kısım — çünkü sahte güven üretir.**
Sen "makaleye göre alım-satım gerçek hayatta zor olabilir" dedin. Haklısın, ben daha
da ileri gideceğim: DSR/PBO/Kelly/HMM matematiği **gerçek** ama **yanlış nesneye**
uygulanırsa (Codex'in tespiti: DSR Dip-Al serisine, PBO MA-ızgarasına) "sofistike"
görünen bir hiç üretir. Üstelik atıf yaptığın momentum literatürü (Moskowitz-Ooi-
Pedersen vb.) **kesitsel, çeşitlendirilmiş, düşük-maliyetli kurumsal** stratejiler
içindir; 30 hisselik perakende RSI/MACD dip-alıcısına transfer olduğu varsayımı bir
**kategori hatasıdır.** Anomalilerin maliyet-sonrası getirisinin kâğıttakinden çok
farklı olduğu, gerçek işlem verisiyle gösterilmiştir (Frazzini, Israel & Moskowitz,
*Trading Costs of Asset Pricing Anomalies*). Ve kendi backtestine bile şüpheyle
yaklaşman gerektiği bir protokol hâline gelmiştir (Arnott, Harvey & Markowitz, *A
Backtesting Protocol in the Era of Machine Learning*, J. Financial Data Science 2019).
**Ölçümün titizliği, edge'in varlığını kanıtlamaz.** Cila ne kadar parlaksa, altındaki
boşluğu görmek o kadar zorlaşır — asıl risk budur.

### 4) Bir kapsam kesimi daha: canlı arena 4 değil, 1 hesapla başlasın

Codex V1'i 4 para hesabıyla donduruyor. Acımasız düzeltme: **4 stilin yarışı bir
"istek", ilk bilimsel soru değil.** İlk soru şu: *"en iyi tek stratejim (Dengeli),
maliyet-sonrası XU100 al-tut'u geçiyor mu?"* Bu 1 hesap + benchmark + ML observer.
Replay ücretsiz olduğu için **4 profili de replay'de** test ederiz; ama **canlı paper
parası yalnız replay'i geçen profil(ler)le** başlar. Hiçbiri geçmiyorsa canlı arena
hiç açılmaz. Bu, hata-ayıklama yüzeyini küçültür ve seni gerçek cevaba en hızlı
götürür.

### 5) KILL CRITERION — kimsenin koymadığı şey (ön-kayıtlı)

Sezon **başlamadan önce** şunu yazıp kilitliyoruz (sonradan gol direğini oynatmak,
DSR'nin uyardığı selection bias'ın ta kendisidir):

> Replay edge-kapısı + ön-kayıtlı canlı paper penceresi (örn. 3–6 ay / en az N
> tamamlanmış işlem) sonunda hiçbir kural hesabı XU100'ü maliyet-sonrası **anlamlı**
> biçimde geçemiyor ve PBO arenanın *kendi* getirisinde yüksek çıkıyorsa → **DUR.**
> Edge'siz çekirdeğin üstüne fon/parayla-ML/RNN EKLENMEZ.

Eşik sayıları (N işlem, pencere uzunluğu, anlamlılık) replay dağılımı görülmeden
sihirli sabit olarak yazılmaz; sezon başından önce kilitlenir.

### 6) Güvenlik — Codex'in son maddesini büyütüyorum (NO-GO)

Mevcut tek-kullanıcılı RLS, service-role ile bypass ediliyor. Uygulama internete
açıkken yalnız UI butonlarına güvenmek, **kimlik doğrulamasız mutasyon** demektir.
Public deploy'dan önce gerçek auth/erişim koruması **zorunlu**; yoksa canlı NO-GO.
Bu, "uyku/Vercel" tartışmasından daha öncelikli — çünkü açık bir yazma-uçlu Supabase,
verini herkese açar.

### 7) Nihai uygulama sırası (Codex'inkini fail-cheap için yeniden sıraladım)

1. **Faz 0** — Codex'in 6 teslimatı + **macro_shock entegrasyon boşluğu** +
   **sigmoid/Platt kalibrasyon** (küçük örnekte isotonic değil). Testlerle.
2. **Execution çekirdeği (saf, Supabase'siz) + historical replay — 4 profil için.**
   Maliyet-sonrası equity vs XU100 + PBO/DSR arenanın kendi serisinde.
3. **🚦 EDGE KAPISI (GO/NO-GO):** Replay'de hiçbir profil XU100'ü net geçemezse → DUR.
   Geçen(ler) varsa devam. *Supabase/UI/arena bu kapıdan ÖNCE yazılmaz.*
4. **MVP SQL onayı** (7 tablo, RLS+auth, unique, atomik RPC) + tek hesap entegrasyon.
5. **Shadow** → **Canlı arena V1** (replay'i geçen profillerle + ML observer) + UI/CSV/
   Telegram.
6. **Ön-kayıtlı pencere sonu → KILL CRITERION değerlendirmesi.**
7. Ancak edge kanıtlanırsa: BIST100 ablation · fon · ML activation · (en son, ve ancak
   klasik model kalıcı OOS edge gösterirse) derin model.

### 8) Nihai söz

Plan **GO.** Codex'in titizliğini alıyorum, kendi macro_shock hatamı kabul ediyorum,
ve üstüne **fail-cheap edge-kapısı + 1→4 hesap + ön-kayıtlı kill-criterion + auth-NO-GO**
koyuyorum. En acımasız cümleyi sana borçluyum, işte:

> Bu projenin başarısı bir kâr eğrisi değil. Başarı, "sinyallerimin maliyet-sonrası
> net edge'i var mı?" sorusuna **kanıtlı bir cevap** — ve o cevap büyük olasılıkla
> HAYIR olacak. HAYIR çıkması da bir başarıdır: seni, edge'siz bir sisteme aylarını ve
> umudunu gömmekten kurtarır. EVET çıkarsa, o zaman gerisini (fon, ML, ölçek) konuşmaya
> hakkımız olur — öncesinde değil.

Tek sonraki somut adım: senin **"kodla"** onayın → Faz 0 (macro_shock + `p_up` yön +
sigmoid kalibrasyon + `ModelEvidence`) ve hemen ardından **edge-kapısı replay'i.**
Supabase, UI ve arena, o kapıyı geçmeden yazılmayacak. Planlama burada biter.

### Kaynakça (benim eklediklerim)

- **Barber & Odean (2000)** — *Trading Is Hazardous to Your Wealth: The Common Stock
  Investment Performance of Individual Investors*, Journal of Finance 55(2). Bireysel
  yatırımcının maliyet-sonrası piyasanın altında kalması → "çalışan bot ≠ kâr" hükmünün
  ampirik temeli.
- **Bailey, Borwein, López de Prado & Zhu (2014)** — *Pseudo-Mathematics and Financial
  Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance*,
  Notices of the AMS 61(5). Çoklu-deneme overfit → edge-kapısının ve PBO'nun *arenanın
  kendi serisine* uygulanması gerekçesi.
- **Arnott, Harvey & Markowitz (2019)** — *A Backtesting Protocol in the Era of Machine
  Learning*, Journal of Financial Data Science. Kendi backtestine şüphe + ön-kayıt
  disiplini → kill-criterion'ın gerekçesi.
- **Frazzini, Israel & Moskowitz** — *Trading Costs of Asset Pricing Anomalies*.
  Kâğıttaki edge'in maliyet-sonrası erimesi → "akademik strateji ≠ gerçek kâr" tespiti.
  (Codex de andı; burada kategori-hatası argümanını destekliyor.)
- **Fama (1970)** — *Efficient Capital Markets: A Review of Theory and Empirical Work*,
  Journal of Finance 25(2). Zayıf-form etkinlik → geçmiş-fiyat-temelli TA'dan sistematik
  edge beklenmemesi (varsayılan şüpheci öncül).

---

## Claude — 2026-07-02 · Sinyal kalitesi araştırması + revize yol haritası

> **Bağlam:** Kullanıcı geri döndü: "Sayfa sayısı arttı ama hâlâ düzgün sinyal
> üretemiyor. %99 başarıyla sinyal istiyorum. Akademik makaleler, mum
> formasyonları, haberler, altın/siyaset gibi BIST'i etkileyen etkenlerle
> birlikte değerlendirsin. Araştır, plan çıkar — geliştirelim mi, sıfırdan mı?"
>
> **Hüküm önden:** Sıfırdan kurmak YOK. Mimari (Faz 0 sözleşmesi, arena,
> edge-kapısı, kill-criterion) doğru ve yeni yazıldı; sorun mimaride değil,
> **sinyal beyninin kendisinde.** Bu giriş, beyni güçlendirecek araştırma
> bulgularını ve S1–S5 sinyal fazlarını tanımlar. Ve en önemlisi: **%99 hedefi
> fiziksel olarak imkânsızdır; hedef yeniden tanımlanmalıdır.**

### 1) %99 gerçeği — hedefin kendisi hatalı

- Tarihin en başarılı fonu Renaissance Medallion, işlemlerinin yalnızca
  **~%50.75'inde** haklı çıkarak yılda ~%66 brüt getiri üretti. Kazanan, isabet
  oranı değil; küçük bir edge'in **çok sayıda işleme, sıkı risk kontrolüyle**
  uygulanmasıdır.
- Akademik literatürde günlük yön tahmini için gerçekçi out-of-sample doğruluk
  tavanı **%52–60** bandıdır (lojistik regresyon ~%55, RF ~%57–63 doğrulama;
  canlıda daha düşük). %60 üstü iddialar neredeyse her zaman look-ahead,
  survivorship veya test-seti sızıntısıdır.
- **Yeni hedef tanımı:** doğruluk değil, *maliyet-sonrası beklenen değer*:
  `E = isabet × ort.kazanç − (1−isabet) × ort.kayıp − maliyet > 0`
  %55 isabet + 1.5 payoff oranı, "%99 doğruluk" arayışından daha zengin eder.
  Arena'nın edge-kapısı zaten tam bunu ölçüyor — o disiplin korunur.

### 2) Teşhis — sinyaller neden "düzgün" değil (kod incelemesi)

1. **Per-ticker model veri açlığı içinde.** `ml/model.py` her hisseyi kendi
   ~2.500 barıyla eğitiyor; 16 özellik + GBDT bu örneklemde taban çizgisini
   geçemez (skill ≈ 0 gözlemi bununla tutarlı). Planın Faz 3'ünde zaten yazılı
   olan **pooled panel model** artık ertelenemez; sinyal kalitesinin ana kaldıracı bu.
2. **Etiket, işlemle hizasız.** `make_labels` 10 gün sonrası basit yukarı/aşağı
   soruyor; motor ise ATR stop/hedef ile işlem yapıyor. Model başka bir soruya
   cevap veriyor, kasadan başka bir soru soruluyor. (Codex F0.7 tespitiyle aynı;
   çözüm §4/S1.)
3. **BIST'in ana sürücüleri modelde yok.** Literatür BIST100 için USDTRY, altın,
   CDS/risk primi, faiz, M2, S&P500'ü ana açıklayıcılar olarak gösteriyor; bizim
   özellik setimiz tamamen hisse-içi teknik. Rejim HMM ve macro_shock var ama
   *kapı* olarak kullanılıyor, *özellik* olarak modele girmiyor.
4. **Teknik analiz tek başına edge kaynağı değil.** Kanıt dengesi: geçmiş-fiyat
   göstergeleri maliyet sonrası tek başına sistematik edge üretmez (Fama 1970;
   Marshall ve ark. 2006). Teknik katman **aday üretici + zamanlama filtresi**
   olarak doğru; "sinyalin kendisi" olarak yanlış konumlanmış beklenti.

### 3) Araştırma bulguları (kullanıcının sorduğu dört başlık)

**a) Mum formasyonları ("mum biyografisi"):** Kanıt karışık ve zayıf.
Marshall-Young-Rose (2006) Dow 30'da 28 yaygın formasyonda edge bulamadı;
Tharavanij ve ark. (2017, Tayland) anlamlı ortalama getirisi olan formasyonların
bile yön tahmininde güvenilmez olduğunu gösterdi. Buna karşın Caginalp & Laurent
(1998) bazı **3-günlük** formasyonlarda kısa vadeli sinyal, Lu & Shiu (2012,
Tayvan) 4 formasyonda maliyet-sonrası kâr buldu. **Sonuç:** mum formasyonu tek
başına AL/SAT üretmemeli; mevcut `candlesticks.py` çıktısı pooled modele
*ikili özellik* olarak girmeli ve ağırlığını veri belirlemeli.

**b) Haber/duyarlılık:** Türkçe çalışmalar KAP duyuruları + sosyal medya
duyarlılığının (Word2Vec/FastText + LSTM vb.) BIST tahminine katkı verdiğini
raporluyor; ayrıca TCMB faiz kararları, siyasi şoklar (Gezi, 2016 darbe
girişimi, seçimler) volatilite rejimini belirgin değiştiriyor. **Sonuç:** haber
katmanı alfa kaynağı değil **risk filtresi** olarak eklenmeli (bkz. S5) —
"kötü haber gününde pozisyon azalt" işlevi, "haberden yön tahmin et"
işlevinden çok daha sağlam.

**c) Altın/siyaset/makro:** BIST endeks çalışmalarında USDTRY, dolar endeksi,
altın, M2, S&P500, tahvil faizi tekrar tekrar ana değişkenler. Bunlar zaten
`gold_fx_page` ve `macro_shock` olarak projede var — eksik olan modele
**özellik** olarak bağlanmaları (S3).

**d) Yöntem literatürü:** López de Prado'nun **triple-barrier etiketleme +
meta-labeling** çerçevesi (Advances in Financial ML, 2018; Singh & Joubert
2022) bizim mimariye birebir oturuyor: birincil sinyal (deterministik motor,
recall'u yüksek) + ikincil ML filtresi (precision'ı yükseltir, pozisyon boyutu
verir). Yayınlanmış deneylerde precision'ı belirgin artırdığı gösterildi.
Mevcut "güven skoru + ML olasılığı ortalaması" yerine bu **kademeli** yapı
hedeflenmeli.

### 4) Revize sinyal fazları (S-serisi; arena fazlarına paralel, onları değiştirmez)

**S1 — Etiketleme düzeltmesi (en yüksek getiri/emek oranı).**
Triple-barrier etiket: üst bariyer = ATR hedef, alt = ATR stop, dikey = maks
tutma süresi — motorun *gerçek* işlem kurallarıyla aynı. Hedef değişken
XU100-relatif ve maliyet-sonrası. `ml/features.py`'a `make_labels_triple_barrier`.

**S2 — Pooled panel model + Supabase model deposu.**
BIST100 tüm hisseler tek modelde (~75K örnek); kesitsel özellikler (sektör- ve
XU100-relatif momentum/vol, `cross_section.py` z-skorları) eklenir. Purged
walk-forward CV + Platt kalibrasyon (Faz 0 kararı korunur). **Eğitim gecelik/
haftalık bir kez**; gün içinde yalnız tahmin — model artefaktı (pickle +
`model_version` + metrik karnesi) Supabase'e yazılır, cron başında indirilir.
Bu, "her gün geçmiş veriyle baştan eğitmesin" talebinin doğrudan karşılığıdır
ve mevcut `model_cache` deseninin genellemesidir.

**S3 — Makro/rejim özellikleri.**
USDTRY (seviye momentum + gerçekleşen vol + şok bayrağı), gram altın/XAUUSD,
XU100 rejim HMM durumu, (bulunabilirse) 5Y CDS veya vekili (eurobond spread /
USDTRY vol), TCMB faiz patikası → pooled modele özellik. Mum formasyonu
bitleri (`candlesticks.py`) ve sezonsallık aynı şekilde özellik olarak girer.

**S4 — Meta-labeling mimarisi.**
Birincil: mevcut deterministik gated sinyal (aday üretir). İkincil: S2 modeli
"bu adaya gir/girme" olasılığı verir; kalibre olasılık ¼-Kelly boyutlandırmayı
besler. ML artık kural motorunun *rakibi* değil *filtresi* — arena'daki
"ML-öncelikli hesap" bu mimariyle yeniden tanımlanır (ablation korunur).

**S5 — Haber/olay katmanı (en son, yalnız filtre).**
KAP duyuru başlıkları (WAF nedeniyle graceful-degrade), seçim/faiz-kararı
takvimi, USDTRY şok bayrağı → "olay günü" bayrağı: yeni giriş yok / pozisyon
yarıya. Türkçe duyarlılık modeli (FinBERT-TR vb.) ancak S1–S4 edge kanıtlarsa
denenir; alfa değil fren.

**Değişmeyenler:** Edge-kapısı (replay'de XU100'ü maliyet-sonrası geçemeyen
canlıya çıkmaz), kill-criterion, look-ahead'siz T+1 fill, kapatılamaz emniyet
kapıları, "başarı = kanıtlı EVET/HAYIR" hükmü. Her S fazı sonunda aynı replay
koşulur; metrik iyileşmiyorsa faz geri alınır (champion/challenger).

### 5) Başarı metriği sözleşmesi (ön-kayıt, %99'un yerine)

| Metrik | Eşik (öneri) |
|--------|--------------|
| İsabet (hit rate) | ≥ %53 (OOS, maliyet-sonrası) |
| Payoff (ort. kazanç/kayıp) | ≥ 1.3 |
| Maliyet-sonrası alpha vs XU100 | > 0 (replay + canlı pencere) |
| Kalibrasyon (Brier vs taban) | tabandan iyi |
| PBO | < 0.5 (arenanın kendi serisinde) |

Bu tablo sezon başında kilitlenir; %99 hedefi resmen emekliye ayrılır.

### Kaynakça (bu girişin ekledikleri)

- Singh & Joubert (2022) — *Does Meta-Labeling Add to Signal Efficacy?* (Hudson
  & Thames) — meta-labeling'in precision katkısının deneysel kanıtı.
- López de Prado (2018) — *Advances in Financial Machine Learning* — triple-barrier
  etiketleme + meta-labeling çerçevesi.
- Marshall, Young & Rose (2006) — *Candlestick technical trading strategies: Can
  they create value for investors?* J. Banking & Finance — Dow 30'da mum
  formasyonlarında edge yok.
- Lu & Shiu (2012) — *Profitable candlestick trading strategies — evidence from
  a new perspective* (Rev. Financial Economics) — Tayvan'da 4 formasyon maliyet
  sonrası kârlı; formasyonların özellik olarak değeri olabilir.
- Tharavanij, Siraprapasiri & Rajchamaha (2017) — *Profitability of Candlestick
  Charting Patterns in the Stock Exchange of Thailand* (SAGE Open).
- Caginalp & Laurent (1998) — *The predictive power of price patterns* (Applied
  Mathematical Finance) — 3-günlük formasyonlarda kısa vadeli sinyal.
- BIST çalışmaları: makro sürücüler (USDTRY, altın, M2, S&P500, faiz) ve Türkçe
  haber/duyarlılık katkısı — dergipark/Springer/IEEE (BIST100 endeks tahmini,
  KAP + sosyal medya duyarlılığı, olay-volatilite çalışmaları).

---

## Claude — 2026-07-02 · S1 TAMAMLANDI (üçlü-bariyer etiketleme kodlandı)

Kullanıcı onayı ("kodla") ile S1 uygulandı:

- **`ml/features.py`**: `triple_barrier_outcomes` + `make_labels_triple_barrier` —
  arena motorunun birebir fiziğiyle (T+1 açılış fill + slippage, ATR(14) Wilder
  stop/hedef `analytics.risk` ile aynı formül, gap kuralı, "aynı barda ikisi de →
  stop önce", süre bariyeri → ertesi açılış satışı, çift-yön komisyon) her bar için
  işlem sonucu simüle eder. `benchmark` verilirse getiri XU100-relatif. Fizik
  sabitleri (`TB_*`) modül başında, motor değerleriyle senkron tutulmalı.
- **`build_training_set`**: varsayılan etiket `triple_barrier` (eski davranış
  `labeling="fixed"` ile durur); `horizon` artık süre bariyeri anlamında.
- **`ml/model.py`**: `train_signal_model(labeling=, benchmark=)`; sonuçta
  `labeling` alanı. Model artık "N gün sonra yukarı mı?" değil "bu kurulumla
  açılan işlem maliyet-sonrası kazanır mı?" öğreniyor.
- **`analytics/probability.py`**: kalibre olasılık üçlü-bariyer etikete geçti;
  `MODEL_VERSION = "calib-v2-sigmoid-bss-tb"` (v1 snapshot'larıyla kıyas kesintisi
  işaretli — plan §sinyal sürümlemesi).
- **Bağlantılar**: gecelik model işi XU100'ü bir kez çekip tüm hisselere
  `benchmark` geçiyor; portföy "derin istatistik" ve ML sayfası aynı şekilde;
  UI metinleri "yukarı olasılığı" → "kazanma olasılığı (üçlü-bariyer)" olarak
  düzeltildi (yanlış beyan olmasın).
- **Testler**: hedef-vuruş=1, stop=0, aynı-bar muhafazakârlığı, süre çıkışında
  maliyetin düz seriyi 0 yapması, kuyruk NaN (look-ahead yok), XU100-relatif
  etiketin mutlak etiketi çevirmesi, eski `fixed` yolunun yaşaması. 13/13 ML
  testi + BIST katmanı paketi regresyonsuz.

**Sonraki adım (S2):** pooled BIST100 panel modeli + Supabase model deposu
(gecelik eğitim, gün içi yalnız tahmin). S1 etiketleyicisi S2'nin girdisidir.

---

## Claude — 2026-07-02 · S2 TAMAMLANDI (havuz modeli + Supabase model deposu)

- **`ml/pooled.py`**: tüm evren tek panelde (`build_panel` — MultiIndex
  [date, ticker], S1 üçlü-bariyer etiketi, XU100-relatif). Özellik seti = mevcut
  16 ölçek-bağımsız kolon + **kesitsel** kolonlar (aynı-gün evren-içi z-skorlar:
  momentum/vol/RSI/zirveye-uzaklık + endekse-relatif 20g momentum; yalnız
  aynı-gün bilgisi → nedensel). Doğrulama **purged walk-forward**: eğitim sonu
  ile test başı arasında `horizon+1` günlük embargo — eğitim etiketlerinin
  ileri-bakan penceresi test dönemine taşamaz (López de Prado §7). Kalibrasyon
  F0.2 disiplini (erken %70 Platt, geç %30 dokunulmamış test, aynı
  `validate_model_evidence` kapısı). `POOLED_MODEL_VERSION="pooled-v1-tb-purged"`.
- **Kalıcılık (kullanıcının kök talebi)**: `storage/pooled_models.py` +
  `pooled_models` tablosu — artefakt (pickle+zlib+base64, ~66KB) + karne; her
  eğitim yeni satır (tarihçe birikir). `load_latest(require_quality=True)`
  otomatik-işlem tüketicileri içindir. Gün içinde yalnız `predict_pooled`
  (indir + tahmin); yeniden eğitim yok.
- **Champion/challenger**: gecelik iş (`run_nightly_models.py`) per-ticker
  şampiyonu aynen yazar; havuz modeli **challenger** olarak
  `model_cache.p_up_pooled / pooled_version / pooled_quality` kolonlarına gider.
  Canlı davranış değişmez; karne birikir → S4'te (meta-labeling) hangisinin
  ikincil filtre olacağına kanıtla karar verilir. Havuz adımı başarısız olursa
  gecelik işin kalanı etkilenmez (try/except + şema-eksik mesajı).
- **Testler** (`tests/test_pooled.py`, 8): panel kurulumu + kesitsel z'nin
  gün-içi ortalamasının 0 olması, embargo boşluğunun doğrulanması, uçtan uca
  eğitim + kanıt alanları, tahmin aralığı, artefakt gidiş-dönüşünün birebir aynı
  tahmini vermesi, bozuk artefaktın sessiz boş dönmesi, küçük panelin gürültüyle
  reddi. Kuru koşu: sentetik gürültüde kalite kapısı dürüstçe GEÇEMEDİ dedi
  (beklenen davranış — kapı çalışıyor).

**Sonraki adım (S3):** USDTRY/altın/rejim-HMM/mum-formasyon bitlerinin panele
özellik olarak eklenmesi; ardından S4 meta-labeling.

---

## Claude — 2026-07-02 · S3 TAMAMLANDI (makro + mum özellikleri panele girdi)

- **`ml/macro.py`**: tarih-indeksli makro çerçeve — XU100 20g momentum,
  MA200-üstü rejim vekili (HMM tam-örneklem parametresi bilerek KULLANILMADI:
  parametre sızıntısı olur; nedensel vekil replay'in rejim proxy'siyle tutarlı),
  XU100 gerçekleşen vol, USDTRY 20g ivme + vol (TL stresi), altın (GC=F) 20g
  momentum. Kaynak seri yoksa kolon nötr 0.0 (graceful-degrade).
- **`ml/features.py`**: `candle_feature_frame` — 15 formasyonun yön×güç yükü
  bar başına [-1,+1] `candle_net` + 3-bar penceresi `candle_net_3`. Literatür
  kararı (plan §3a) uygulandı: mum tek başına sinyal değil, **özellik biti**;
  ağırlığını model belirler. Per-ticker şampiyonun özellik seti DEĞİŞMEDİ.
- **`ml/pooled.py`**: `PANEL_FEATURES` = 16 taban + 5 kesitsel + 2 mum + 6 makro
  (29 kolon). Makro, tarih üzerinden join'lenir (aynı gün → tüm hisselere aynı
  satır; eksik gün ffill — nedensel). `predict_pooled` aynı sözleşmeyle son
  makro satırını kullanır; eski (v1) artefakt kendi `feature_columns` listesiyle
  uyumlu kalır. Sürüm: **pooled-v2-tb-macro** (kıyas kesintisi).
- **Gecelik iş**: `_macro_frame` USDTRY (`TRY=X`) + altın (`GC=F`) çeker;
  çekilemezse nötr kolonlarla eğitim sürer, iş kırılmaz.
- **Testler** (+6): makro çerçevenin kaynaksız nötrlüğü + MA200 ısınması,
  kaynaklıyken sıfırdan farklılığı, panelde aynı-gün-aynı-değer değişmezi,
  makro'suz panelin yaşaması, makrolu uçtan uca eğitim+tahmin, boğa-yutan
  formasyonun pozitif yükü + NaN üretmeme. 27/27 ML+pooled; tam pakette
  regresyon yok. Kuru koşu: v2 artefakt ~67KB; sentetikte kesitsel kolon
  (cs_dist_high_z) önem sırasına şimdiden girdi.

**Sonraki adım (S4):** meta-labeling — deterministik gated sinyal birincil,
havuz modeli ikincil filtre; arena "ML-öncelikli" hesabının yeniden tanımı.

---

## Claude — 2026-07-03 · S4 TAMAMLANDI (meta-labeling altyapısı — kanıt-kapılı)

Kullanıcı kararı: altyapı şimdi kurulur, filtrenin canlı etkisi **karne kanıtına
şartlanır**. Uygulanan mimari (López de Prado meta-labeling, plan §3d):

- **`analysis/meta.py`** — saf `meta_gate`: birincil sinyal (deterministik gated
  karar) adayları üretir; ikincil model yalnız **veto** (p_win < 0.45) ya da
  **boyut kısma** (0.45→0.60 arası lineer, taban ×0.5) yapabilir. Filtre boyutu
  ASLA büyütmez, kendi başına işlem AÇAMAZ. Model seçimi: kaliteli pooled >
  kaliteli per-ticker > **pasif** (allow=True, ×1.0). Kalite kapısını geçmiş
  model yokken kapı kendiliğinden pasiftir → bugünkü canlı davranış birebir
  sürer; etki, gecelik karneler kalite kapısını geçen bir model üretince başlar.
- **İzlenebilirlik**: kapı kararı her analizde `signals["ml_gate"]` olarak
  snapshot'a yazılır ("neden girildi/girilmedi/kısıldı" izi); kısılan emirlerin
  gerekçesine `· ML ×0.xx` eklenir; observer karnesine `meta_p_win/meta_source/
  meta_allow` alanları girer — "hangi model filtre olmalı?" bu kanıtla seçilecek.
- **Ablation korunur**: filtre yalnız `use_ml_meta_filter=True` profilde etkir —
  o da yeniden tanımlanan **ML-meta** hesabı (eski "ML-öncelikli"; hâlâ OBSERVER,
  para harcamaz). Diğer 4 para hesabı kontrol kolu olarak filtresiz.
- **Canlı boşluk kapatıldı**: `build_session_inputs` artık gecelik model
  önbelleğini okuyup `analyze_universe`'e geçiriyor — p_up/pooled alanları canlı
  arenaya hiç akmıyordu (sessiz boşluk); Supabase yoksa {} ile pasif sürer.
- **Testler** (`tests/test_meta.py`, 11): kalitesiz modelin pasifliği, pooled >
  per-ticker önceliği, veto eşiği, çarpanın monotonluğu ve [0.5, 1.0] sınırı,
  geçersiz girdinin pasifliğe düşmesi; motor entegrasyonunda veto'nun yalnız
  meta profilde işlemesi (kontrol profil etkilenmez), kısmi çarpanın adet
  küçültmesi + gerekçe izi, pasif kapının davranışı değiştirmemesi, profil
  bayraklarının doğruluğu. Tam pakette regresyon yok.

**Aktivasyon kriteri (ön-kayıt):** ML-meta hesabına para verilmesi (OBSERVER→
ACTIVE) ancak (a) pooled/champion karnesi kalite kapısını art arda geçer ve
(b) observer karnesindeki meta kararları birincil-sinyal-yalın sonuçlardan
maliyet-sonrası iyiyse, YENİ sezonla yapılır. Kod değişikliği: tek satır
(status) + yeni sezon — geriye dönük hiçbir sonuç değişmez.

**Sonraki adım (S5):** olay/haber takvim filtresi (KAP/seçim/faiz günü riski) —
yalnız fren olarak; S1–S4 edge kanıtlamadan duyarlılık modeli denenmez.

# 🚦 Edge Kapısı — Ön-Kayıtlı Karar Dosyası

> Plan (PAPER_TRADING_ARENA_PLAN.md §NİHAİ KARAR) gereği eşikler sonuca
> **bakılmadan önce** burada kilitlenir. Sonuç geldikten sonra bu bölümdeki
> kurallar DEĞİŞTİRİLMEZ; değiştirmek selection bias'tır.

## Kilitli kurallar (2026-07-06)

- **Koşu:** `.github/workflows/arena-replay.yml` (workflow_dispatch) veya yerelde
  `python scripts/run_arena_replay.py --period 5y`.
- **Pencere:** 5y · **Evren:** BIST30 (bugünkü liste — survivorship sınırlaması bilinir).
- **Fizik:** ortak `ExecutionConfig` (T+1 açılış fill, bps komisyon + yönlü slippage,
  gap kuralı, aynı-bar'da stop-önce).
- **GEÇTİ tanımı (kodla aynı):** en az bir ACTIVE profil, maliyet sonrası XU100
  al-tut'u **hem toplam getiri hem Sharpe'ta** geçer.
- **Bakış hakkı:** Bu tanımla en fazla **1 karar koşusu**. Parametre/sinyal
  değiştirilip yeniden koşulursa bu, YENİ bir kapı denemesidir ve buraya ayrı
  satırla işlenir (deneme sayısı gizlenmez — PBO/DSR mantığı).
- **GEÇMEZSE:** Faz 1B (Supabase arena, shadow, canlı paper) YAZILMAZ; sinyal
  katmanına dönülür.
- **GEÇERSE:** Faz 1B'ye tek hesapla (en iyi profil) + ML observer ile devam;
  asıl hakem canlı paper penceresi + kill criterion olmaya devam eder.
- **Bilinen sınırlama:** replay dip-sinyali+ATR+200GHO proxy ölçer; güven
  skoru/Kelly/p_up uygulanmaz. Kapı "dip stratejisinin edge'i"ni test eder.

## Koşu günlüğü

| # | Tarih | Pencere | Sonuç | Karar | Not |
|---|-------|---------|-------|-------|-----|
| 1 | 2026-07-06 | 5y | ❌ GEÇİLMEDİ | **NO-GO: arena altyapısı (Faz 1B+) yazılmayacak; sinyal katmanına dönülüyor** | Aşağıda |

### Koşu #1 detayı (2026-07-06, yerel, 29/30 ticker — KOZAL.IS Yahoo'da yok)

XU100 al-tut: **%+943.3** getiri · Sharpe 1.80 · maksDD −22.9%

| Hesap | Getiri | Sharpe | MaksDD | İşlem |
|---|---|---|---|---|
| Agresif | +604.3% | **2.08** | −24.6% | 391 |
| Trend-takip | +249.0% | **1.97** | **−8.7%** | 262 |
| Dengeli | +219.1% | 1.68 | −18.7% | 314 |
| Temkinli | +118.1% | 1.54 | −14.1% | 246 |

Kilitli tanım "getiri VE Sharpe" idi; hiçbir profil getiride yaklaşamadı → GEÇİLMEDİ.

**Dürüst okuma (karar değil, gözlem):** Başarısızlık tamamen **getiri bacağında**;
Agresif ve Trend-takip Sharpe'ta endeksi geçti, hepsi drawdown'da endeksten iyi.
2021→2026 BIST nominal-TL enflasyon boğasında kısmi maruziyet (nakitte bekleyen
dip stratejisi) yıllık bileşikte ezildi — "nakit sürtünmesi" ana kayıp kaynağı.
Sharpe'a bakıp kapıyı sonradan "geçti" saymak selection bias olur; sayılmadı.

**Sonraki adım (yeni kapı denemesi olarak, gizli optimizasyon değil):** denenecekse
"sürekli-yatırımda kal + rejim kötüyken maruziyeti azalt" ailesi (endeks tabanı +
sinyal overlay) deneme #2 olarak bu günlüğe önceden yazılıp koşulur. Evren listesi
düzeltmesi gerekli: KOZAL.IS delist/yeniden adlandırılmış.

---

## Deneme #2 — ÖN-KAYIT (2026-07-06, sonuç görülmeden kilitlendi)

**Hipotez:** Koşu #1'in kaybı nakit sürtünmesinden geldi; "hep yatırımda kal,
ayı rejimde küçül" profili maliyet sonrası XU100'ü geçebilir.

**Kilitli kurallar (kod: `profiles.py` → `overlay`, `engine.py` → `_overlay_orders`):**

- Evren: BIST30 (KOZAL.IS çıkarıldı — Yahoo'da yok; 29 hisse).
- Sepet: en yüksek **20 günlük momentum**lu hisseler, **eşit ağırlık**
  (poz. tavanı %12), hedef **10 pozisyon** (boğa ≈ ~%100 yatırımda).
- Rejim: XU100 < 200GHO (ayı) → hedef pozisyon sayısı **4** (≈%40-48 maruziyet);
  küçülürken en zayıf momentumlular satılır. **Nakite tam dönüş yok.**
- Rotasyon: **21 barda bir**; momentumu ilk 2×hedef sıralamasının dışına düşen
  tutulan satılır (histerezis — turnover sınırlı).
- ATR stop/hedef **yok**; max_hold süresi yok; kill-switch (maks hesap DD) açık.
- Fizik AYNI: 5bps komisyon + 10bps slippage + T+1 açılış fill.
- **GEÇTİ tanımı değişmedi:** getiri VE Sharpe'ta XU100 al-tut'u geç (5y).
- Not: koşu #1'deki 4 sinyal profili de aynı koşuda yeniden raporlanır (evren
  artık 29 hisse olduğu için sayılar hafif oynayabilir; karar overlay üzerinden).

**Koşu komutu:** `python scripts/run_arena_replay.py --period 5y`
(veya Actions → "Arena edge kapısı (replay)").

| # | Tarih | Pencere | Sonuç | Karar | Not |
|---|-------|---------|-------|-------|-----|
| 2 | 2026-07-06 | 5y | ❌ GEÇİLMEDİ | NO-GO devam | Aşağıda |

### Koşu #2 detayı (2026-07-06, GitHub Actions run 28824820981, 29 ticker)

XU100 al-tut: **%+942.8** · Sharpe 1.81 · maksDD −22.9%

| Hesap | Getiri | Sharpe | MaksDD | İşlem |
|---|---|---|---|---|
| Agresif | +623.7% | 2.10 | −24.6% | 389 |
| **Overlay (deneme #2)** | **+499.7%** | **1.35** | −28.5% | 248 |
| Trend-takip | +248.8% | 1.97 | −8.7% | 262 |
| Dengeli | +215.5% | 1.67 | −18.7% | 314 |
| Temkinli | +117.2% | 1.53 | −14.1% | 245 |

**Dürüst okuma:** Nakit-sürtünmesi hipotezi kısmen doğrulandı — Overlay, sinyal
profillerinin getirisini ikiye katladı (+499.7 vs +215-249). Ama al-tut'a hâlâ
−443 puan geride VE Sharpe'ı endeksin altına düştü (1.35 < 1.81). Kayıp
kaynakları: 248 işlemlik rotasyon maliyeti, momentum seçiminin bu pencerede
endekse değer katmaması, ayı-küçülmesinin toparlanmaları kaçırması. İki koşuda
da tutarlı tek sinyal: **Agresif Sharpe'ta endeksi geçiyor (2.10/2.08 > 1.81)**
ama getiride asla — kaldıraçsız spot BIST'te bu kapatılamaz bir açık.

**Deneme #3 (İZİN VERİLEN SON KOŞU — ön-kayıt, 2026-07-06):** En sade soru
kaldı: "rejim zamanlaması TEK BAŞINA al-tut'a değer katıyor mu?" Kurallar:
- Sepet: 29 hissenin TAMAMI eşit ağırlık, **rotasyon YOK** (momentum seçimi yok).
- Tek kural: boğa (XU100 ≥ 200GHO) → %100; ayı → pozisyonların yarısı satılır
  (%50); boğaya dönüşte geri alınır. Başka hiçbir işlem yok.
- Fizik ve GEÇTİ tanımı aynı.
- **Bu da geçemezse kapı KAPANIR** (kill criterion): bu sinyal setiyle BIST'te
  al-tut'u yenme iddiası bırakılır; proje NEXT_STEPS §5'teki "destekli disiplin
  aracı" rotasına döner. Deneme #4 açılmaz.

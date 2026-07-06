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

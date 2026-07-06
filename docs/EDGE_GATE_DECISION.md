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
| 1 | _(beklemede)_ | 5y | — | — | İlk karar koşusu |

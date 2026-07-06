# 🧭 Gelecek Adımlar — Edge Kapısı #1 Sonrası Yol Haritası

> Durum (2026-07-06): Kapı #1 **GEÇİLMEDİ** (bkz. EDGE_GATE_DECISION.md).
> Plan gereği arena altyapısı (Supabase ledger/shadow/canlı paper) DONDURULDU.
> Şimdiki iş sinyal katmanında: "al-tut'u riske-ayarlı yenen" bir strateji bulmak.
> Bu dosya sıradaki somut adımları, hata çözümlerini ve ölçüm doğruluğunu
> artırma yollarını tek yerde tutar.

---

## 1) Hemen şimdi (sıra önemli)

1. **Branch'i merge et:** `claude/project-review-suggestions-uiqbqz` →
   ana geliştirme branch'i. İçinde: karar kaydı, CI, replay workflow, README fix.
2. **Actions'ı doğrula:** billing kilidi çözüldüyse herhangi bir kırmızı run'da
   "Re-run jobs" → `Testler (çekirdek)` yeşil olmalı (119 test).
3. **KOZAL.IS'i düzelt:** Yahoo'da sembol yok (delist/yeniden adlandırma).
   `tradingagents/strategy/dip_signal.py` içindeki `BIST30` listesinden çıkar
   veya güncel sembolle değiştir; güncel BIST30 bileşenlerini kontrol et.
4. **Deneme #2'yi ÖN-KAYITLA** (aşağıdaki §2 şablonu) → sonra koş. Kurallar
   koşudan önce EDGE_GATE_DECISION.md'ye yazılmadan replay çalıştırılmaz.

## 2) Deneme #2 — "Sürekli yatırımda kal + rejimde koru" (overlay)

Koşu #1'in öğrettiği: sinyaller riske-ayarlı işe yarıyor (Sharpe/DD endeksten iyi)
ama nakitte beklemek enflasyonist BIST'te bileşik getiriyi öldürüyor. Test edilecek
hipotez: **maruziyeti hiç sıfırlamayan, kötü rejimde azaltan** profil.

Kilitlenecek kural taslağı (koşmadan önce EDGE_GATE_DECISION.md'ye kopyala):

- **Taban:** %100 hisse maruziyeti hedefi (BIST30 eşit-ağırlık veya sinyal-seçimli).
- **Rejim overlay:** XU100 < 200GHO iken maruziyet %100 → %30-50'ye iner
  (tamamen nakit YOK); 200GHO üstüne dönünce geri %100.
- **Sinyal overlay:** dip/teknik sinyaller hisse SEÇİMİ için kullanılır
  (hangi 5-8 hisse), zamanlama için değil.
- **Çıkış:** ATR stop yerine "sıradakiyle değiştir" (rotasyon) — nakite değil,
  başka hisseye geç.
- **Aynı fizik:** 5bps komisyon + 10bps slippage + T+1 açılış fill (değişmez).
- **GEÇTİ tanımı aynı kalır:** getiri VE Sharpe'ta XU100 al-tut'u geç.
- Deneme sayısı günlüğe işlenir; kaç varyant denendiyse HEPSİ yazılır (PBO mantığı:
  10 varyant deneyip 1'i geçerse bu "edge" değil, seçim şansı olabilir).

Kod işi: `arena/profiles.py`'ye `overlay` profili + `engine.py`'de "maruziyet
hedefi" modu (~yarım gün). Mevcut motor T+1/stop/komisyon fiziğini zaten biliyor.

## 3) Daha doğru sonuçlar için (ölçüm kalitesi, öncelik sırasıyla)

| # | İyileştirme | Neden | Maliyet |
|---|---|---|---|
| 1 | **Out-of-sample ayrımı:** parametreyi 2019-2023'te seç, kararı SADECE 2024→bugün diliminde ver | Tek pencerede hem deneme hem karar = kendi kendini kandırma | Bedava (disiplin) |
| 2 | **Temettü / total return:** yfinance `auto_adjust=True` doğrulaması + XU100 yerine mümkünse XU100T (total return) veya temettü etkisinin notu | BIST'te temettü ~%2-4/yıl; 5 yılda bileşikte büyük fark | Küçük |
| 3 | **USD veya enflasyon bazlı ikinci tablo:** aynı equity eğrilerini USDTRY'ye böl | Nominal TL'de %943 aslında büyük ölçüde enflasyon; gerçek soru reel alpha | Küçük |
| 4 | **Dönemsel BIST30 üyeliği:** en azından 2021-2026 giren/çıkanların listesi ile duyarlılık koşusu | Survivorship: bugünün kazananlarını geçmişe uygulamak sonucu şişirir | Orta (veri bulma) |
| 5 | **Maliyet duyarlılığı:** aynı koşuyu 2x ve 4x komisyon/slippage ile tekrarla | Edge maliyete dayanıklı değilse gerçekte yok demektir | Bedava (parametre) |
| 6 | **Parametre taraması yapılırsa PBO:** `analytics/validation.py` zaten var; arena getiri serisine uygula | Çok varyant → en iyisi şansla iyi görünür | Küçük |

## 4) Hata alırsak — bilinen arızalar ve çözümleri

**`404 Quote not found for symbol: XXX.IS` / "possibly delisted"**
→ Sembol Yahoo'dan kalkmış (delist, birleşme, ad değişikliği). Evren listesinden
çıkar veya yeni sembolü bul. Replay zaten atlayıp devam eder; ama 30 hisseden
çok eksilirse (örn. <25) sonuç temsili olmaz — listeyi güncelle.

**yfinance rate limit / boş veri (`YFRateLimitError`, üst üste None)**
→ 10-15 dk bekle, tekrar dene. Israr ederse: `--tickers` ile küçük parça parça koş.
Kalıcı çözüm istenirse fiyatları Supabase'e cache'leyip replay'i oradan besle.

**GitHub Actions "account is locked due to a billing issue"**
→ github.com/settings/billing → ödeme yöntemi/borç düzelt; destek ticket'ı gerekirse
support.github.com. Public repoda Actions dakikaları ücretsizdir; kilit hesap
seviyesindedir, workflow'larla ilgisi yok.

**Actions'ta testler kırmızı ama yerelde yeşil**
→ `tests.yml` sadece beyaz-liste koşar; yeni test dosyası eklediysen listeye ekle.
Bağımlılık hatasıysa `scripts/requirements-hourly.txt`'e pin ekle.

**Replay'de tüm profiller 0 işlem / boş eğri**
→ Veri penceresi kısa gelmiş olabilir (min ~250 bar gerekir) veya sinyal kolonu
üretilememiştir; önce tek ticker'la `--tickers GARAN.IS` koşup çıktıya bak.

**Streamlit arayüzü uyuyor / açılmıyor**
→ Bilinen durum (plan §Dağıtım). Veri hattı cron'da bağımsız çalışır; UI'yi
Hugging Face Spaces/Render'a taşıma kararı arena kapısı geçilirse gündeme alınır.

**Supabase yazma hataları (401/403)**
→ `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` secret'ları eksik/yanlış. Actions →
repo Settings → Secrets. Servis anahtarını asla koda/commit'e koyma.

## 5) Yol ayrımı (önceden kararlaştırılmış)

- **Deneme #2 (veya sonrası) kapıyı GEÇERSE:** Faz 1B — arena SQL onayı → atomik
  fill RPC → TEK hesapla shadow (birkaç seans emir üret, para yok) → canlı paper →
  Telegram özet + öner-onayla "destekli mod". Auth eklenmeden public deploy YOK.
- **3-4 dürüst deneme sonrası hâlâ GEÇİLMİYORSA:** kill criterion devreye girer —
  "BIST'te al-tut + rejimde koruma"dan daha iyisini bu sinyal setiyle yapamıyoruz
  demektir. O durumda proje yine değerli bir sona ulaşır: destekli alım-satım
  katmanı **endeks/fon ağırlıklı, düşük-işlemli bir disiplin aracına** dönüşür
  (sinyaller bilgi amaçlı kalır) ve aylarca edge'siz bot cilalanmaz.

> ⚠️ Her yeni kapı denemesi EDGE_GATE_DECISION.md günlüğüne önceden yazılır.
> Sonuca bakıp kural değiştirmek yok — bu dosyanın varlık sebebi bu.

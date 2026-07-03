"""Arena profilleri — 4 para hesabı + 1 ML observer.

Plan (§Profil kurallarındaki düzeltmeler):
  - Varsayılan **çeyrek Kelly (0.25)**; "½-Kelly" ifadeleri çeyrek Kelly'ye çekildi.
  - Kelly yalnız kalite kapısını geçmiş kalibre olasılıkla; yoksa **sabit risk bütçesi**.
  - Agresif profil rejim filtresini deney amacıyla kapatabilir; ama bayat veri,
    illikidite, maks drawdown gibi **sistemik kill-switch'leri hiçbir profil kapatamaz**.
  - V1: "girişte boyutla, yalnız çıkışta kapat" (düşük turnover) — günlük rebalance yok.
  - ML-öncelikli hesap V1'de **OBSERVER** (para harcamaz; tahmin + karne biriktirir).

``rules_snapshot`` değişmezdir: bir kural değişince eski sezon geriye dönük
değişmemeli; yeni profil sürümü/sezon açılır (plan §Tasarım kuralları).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Profile:
    """Bir kâğıt hesabın değişmez karar/sizing/risk kuralları."""

    code: str                       # makine kodu (snapshot/foreign key)
    name: str                       # UI adı
    emoji: str
    status: str = "ACTIVE"          # ACTIVE (para) | OBSERVER (yalnız tahmin)
    blurb: str = ""

    # ── Evren & giriş kapısı ─────────────────────────────────────────────
    min_confidence: float = 45.0    # birleşik güven eşiği (confidence.py skoru)
    require_regime_bull: bool = False   # yalnız boğa rejimde alım (200GHO proxy)
    require_trend_behavior: bool = False  # hisse kendi 50GHO üstünde (trend-takip)
    use_regime_filter: bool = True  # rejim filtresi açık mı (Agresif kapatır)

    # ── Pozisyon boyutlama & risk ────────────────────────────────────────
    max_positions: int = 8          # eşzamanlı açık pozisyon tavanı
    risk_per_trade: float = 0.01    # işlem başına kasanın riske attığı oranı (stop'a göre)
    max_position_weight: float = 0.20   # tek pozisyon kasanın en çok bu kadarı
    min_cash_reserve: float = 0.05  # her zaman tutulacak minimum nakit oranı
    kelly_fraction: float = 0.25    # çeyrek Kelly (geçerli kalibre p_up varsa)

    # ── Çıkış kuralları (giriş kadar açık olmalı, plan §ML çıkışı) ────────
    use_stop: bool = True           # ATR stop tetiklenince çık
    use_target: bool = True         # R/R hedefi görülünce çık
    max_hold_days: int = 60         # maksimum tutma süresi (rejim/sinyal kaymasına karşı)

    # ── S4: meta-labeling filtresi (ablation korunur — yalnız ML hesabında) ──
    # True ise kaliteli ML modeli girişleri veto edebilir / boyut kısabilir
    # (analysis.meta.meta_gate). Kaliteli model yokken kapı pasiftir; diğer
    # hesaplar kontrol kolu olarak filtresiz kalır.
    use_ml_meta_filter: bool = False

    def is_observer(self) -> bool:
        return self.status == "OBSERVER"

    def rules_snapshot(self) -> dict:
        """Değişmez kural snapshot'ı — sezon başında dondurulur."""
        return asdict(self)


# ── Lig: 4 para hesabı + 1 ML observer ───────────────────────────────────────
PROFILES: dict[str, Profile] = {
    "conservative": Profile(
        code="conservative", name="Temkinli", emoji="🛡️",
        blurb="Yüksek güven eşiği, yalnız boğa rejim, küçük risk bütçesi, az pozisyon.",
        min_confidence=60.0, require_regime_bull=True, use_regime_filter=True,
        max_positions=5, risk_per_trade=0.008, max_position_weight=0.15,
        min_cash_reserve=0.15, max_hold_days=45,
    ),
    "balanced": Profile(
        code="balanced", name="Dengeli", emoji="⚖️",
        blurb="Orta eşik, rejim filtreli, çeyrek Kelly (kaliteli p_up varsa), orta pozisyon.",
        min_confidence=50.0, require_regime_bull=False, use_regime_filter=True,
        max_positions=8, risk_per_trade=0.01, max_position_weight=0.20,
        min_cash_reserve=0.05, max_hold_days=60,
    ),
    "aggressive": Profile(
        code="aggressive", name="Agresif", emoji="🔥",
        blurb="Düşük eşik, rejim filtresi KAPALI (deney), yoğun pozisyon. "
              "Kill-switch'ler yine açık.",
        min_confidence=42.0, require_regime_bull=False, use_regime_filter=False,
        max_positions=12, risk_per_trade=0.015, max_position_weight=0.25,
        min_cash_reserve=0.0, max_hold_days=60,
    ),
    "trend": Profile(
        code="trend", name="Trend-takip", emoji="📈",
        blurb="Yalnız boğa rejim + hisse kendi 50GHO üstünde; momentum girişleri.",
        min_confidence=50.0, require_regime_bull=True, require_trend_behavior=True,
        use_regime_filter=True, max_positions=8, risk_per_trade=0.012,
        max_position_weight=0.20, min_cash_reserve=0.05, max_hold_days=90,
    ),
    "ml_observer": Profile(
        code="ml_observer", name="ML-meta", emoji="🤖",
        status="OBSERVER",
        blurb="S4 mimarisi: deterministik motor aday üretir, kaliteli ML modeli "
              "meta-filtre olarak veto/boyut kısar (kural motorunun rakibi değil "
              "filtresi). V1'de OBSERVER: para harcamaz, karne biriktirir; kalite "
              "kanıtlanınca sonraki sezon parayla aktive edilir.",
        min_confidence=50.0, use_regime_filter=True, max_positions=8,
        risk_per_trade=0.01, max_position_weight=0.20, kelly_fraction=0.25,
        use_ml_meta_filter=True,
    ),
}


def active_profiles() -> list[Profile]:
    """Para harcayan (ACTIVE) profiller — lig sıralamasına girenler."""
    return [p for p in PROFILES.values() if p.status == "ACTIVE"]


def observer_profiles() -> list[Profile]:
    return [p for p in PROFILES.values() if p.status == "OBSERVER"]

"""Meta-labeling kapısı (S4) — ML, kural motorunun rakibi değil FİLTRESİ.

López de Prado'nun meta-labeling mimarisi (plan §3d): birincil sinyal
(deterministik gated karar) adayları üretir — recall ondadır; ikincil model
yalnız "bu adaya girilir mi + ne kadar" der — precision'ı o yükseltir. İkincil
model asla kendi başına işlem AÇAMAZ; yalnız veto edebilir ve boyut kısabilir.

Kanıt disiplini (plan §S4): filtre ancak kalite kapısını geçmiş bir model
varken AKTİF olur. Karne birikmeden (gecelik cron'un pooled/champion karneleri)
hiçbir model kaliteyi geçemeyeceği için kapı kendiliğinden PASİF başlar —
altyapı hazır, etki kanıt gelince başlar. Model seçimi: kaliteli **pooled**
(S2/S3, bu iş için tasarlandı) > kaliteli per-ticker şampiyon > pasif.

Saf modül: ağ yok, yan etki yok, asla istisna fırlatmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# Sistemik eşikler — profil özelliği DEĞİL (ortak fizik gibi ortak filtre kuralı).
VETO_BELOW = 0.45   # kalibre kazanma olasılığı bunun altındaysa girişe veto
FULL_ABOVE = 0.60   # bunun üstünde tam boyut; arada lineer kısılır
FLOOR_MULT = 0.50   # kısmanın tabanı (veto edilmeyen aday en az yarım boyut)


@dataclass
class MetaGateResult:
    """Bir adayın ML filtre kararı — snapshot'a yazılır (izlenebilirlik)."""

    active: bool = False          # kaliteli model devrede mi (yoksa filtre pasif)
    allow: bool = True            # girişe izin (pasifken daima True)
    p_win: float | None = None    # kullanılan kalibre kazanma olasılığı
    source: str = ""              # "pooled" | "per_ticker" | ""
    model_version: str = ""
    size_mult: float = 1.0        # pozisyon boyutu çarpanı [FLOOR_MULT, 1.0]
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def meta_gate(
    *,
    p_up: float | None = None,
    quality_passed: bool | None = None,
    model_version: str = "",
    p_up_pooled: float | None = None,
    pooled_quality: bool | None = None,
    pooled_version: str = "",
    veto_below: float = VETO_BELOW,
    full_above: float = FULL_ABOVE,
) -> MetaGateResult:
    """Aday giriş için ML filtre kararı üretir (saf; istisna fırlatmaz).

    - Kaliteli model yoksa → **pasif**: ``allow=True, size_mult=1.0`` (birincil
      motor tek başına karar verir — bugünkü davranış aynen sürer).
    - Kaliteli model varsa → ``p_win < veto_below`` girişi veto eder; üstünde
      boyut ``[FLOOR_MULT, 1.0]`` aralığında lineer ölçeklenir (filtre boyutu
      asla BÜYÜTMEZ — meta-labeling yanlış pozitifi kırpar, iştah eklemez).
    """
    # Model seçimi: kaliteli pooled > kaliteli per-ticker > pasif
    if bool(pooled_quality) and p_up_pooled is not None:
        p, source, version = float(p_up_pooled), "pooled", pooled_version
    elif bool(quality_passed) and p_up is not None:
        p, source, version = float(p_up), "per_ticker", model_version
    else:
        return MetaGateResult(
            active=False, allow=True, size_mult=1.0,
            reasons=["kalite_gecen_model_yok — filtre pasif"],
        )

    if not (0.0 <= p <= 1.0) or not (0.0 < veto_below < full_above <= 1.0):
        return MetaGateResult(active=False, allow=True, size_mult=1.0,
                              reasons=["gecersiz_girdi — filtre pasif"])

    if p < veto_below:
        return MetaGateResult(
            active=True, allow=False, p_win=round(p, 4), source=source,
            model_version=version, size_mult=0.0,
            reasons=[f"veto: p_win {p:.2f} < {veto_below:.2f}"],
        )

    if p >= full_above:
        mult = 1.0
        reasons = [f"tam boyut: p_win {p:.2f} ≥ {full_above:.2f}"]
    else:
        frac = (p - veto_below) / (full_above - veto_below)
        mult = round(FLOOR_MULT + (1.0 - FLOOR_MULT) * frac, 4)
        reasons = [f"kısmi boyut ×{mult:.2f}: p_win {p:.2f}"]

    return MetaGateResult(active=True, allow=True, p_win=round(p, 4),
                          source=source, model_version=version,
                          size_mult=mult, reasons=reasons)

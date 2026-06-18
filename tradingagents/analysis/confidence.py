"""Birleşik güven skoru + istatistiksel kapı + rejime uyarlanır karar (Faz H).

Eski ikili "çelişki/nötr" etiketini, kanıtları birleştiren **0–100 güven skoruna**
ve **kapılı (gated) karara** çevirir. Tüm girdiler hafif/deterministik (sklearn
gerekmez) → saatlik cron'da çalışır. Kalibre olasılık (``p_up``) verildiğinde
(gecelik precompute) skora katılır, yoksa onsuz da çalışır.

Mantık:
  - Mutabakat (teknik+rasyo+dip+teyit) temel skoru belirler.
  - İstatistiksel anlamlılık (DSR) edge'i şans değilse güveni yükseltir.
  - Rejim hizası: karar piyasa rejimiyle aynı yöndeyse +; karşı-trendse ağır −.
  - Seri davranışı (trend/mean-reversion) ve volatilite rejimi ince ayar.
  - Kapı: güven eşiğin altındaysa ya da güçlü karşı-trendse karar "İZLE"ye düşer
    ("emin değilsen sinyal verme").
"""

from __future__ import annotations

from dataclasses import dataclass, field

_AGREE_BASE = {"güçlü": 25.0, "kısmi": 6.0, "nötr": -8.0, "çelişki": -22.0}
_BULLISH = {"GÜÇLÜ AL", "AL"}
_BEARISH = {"SAT", "KAÇIN"}


@dataclass
class ConfidenceResult:
    score: float = 50.0           # 0–100 birleşik güven
    grade: str = "orta"           # çok yüksek/yüksek/orta/düşük/çok düşük
    decision: str = "İZLE"        # kapılı nihai karar
    gate_passed: bool = False
    reasons: list = field(default_factory=list)


def _grade(score: float) -> str:
    if score >= 75:
        return "çok yüksek"
    if score >= 60:
        return "yüksek"
    if score >= 45:
        return "orta"
    if score >= 30:
        return "düşük"
    return "çok düşük"


def unified_confidence(
    *,
    agreement_level: str,
    decision_raw: str,
    combined_score: float | None = None,
    regime_trend: str | None = None,      # 'boğa'|'ayı'|'belirsiz'
    vol_regime: str | None = None,        # 'düşük'|'normal'|'yüksek'
    behavior: str | None = None,          # 'trend'|'reversal'|'rastgele'
    dsr: float | None = None,             # Deflated Sharpe [0,1]
    p_up: float | None = None,            # kalibre yukarı olasılığı [0,1]
    gate_threshold: float = 45.0,
) -> ConfidenceResult:
    """Kanıtları 0–100 güven skoruna + kapılı karara indirger (saf)."""
    score = 50.0 + _AGREE_BASE.get(agreement_level, 0.0)
    reasons: list[str] = [f"Mutabakat: {agreement_level}"]

    bullish = decision_raw in _BULLISH
    bearish = decision_raw in _BEARISH

    # İstatistiksel anlamlılık (edge şans mı?)
    if dsr is not None:
        if dsr > 0.9:
            score += 14
            reasons.append(f"Edge istatistiksel anlamlı (DSR {dsr:.2f})")
        elif dsr > 0.6:
            score += 5
        elif dsr < 0.3:
            score -= 10
            reasons.append(f"Edge zayıf/anlamsız (DSR {dsr:.2f})")

    # Rejim hizası — karşı-trend ağır cezalı
    counter_trend = False
    if regime_trend in ("boğa", "ayı") and (bullish or bearish):
        aligned = (bullish and regime_trend == "boğa") or (bearish and regime_trend == "ayı")
        if aligned:
            score += 12
            reasons.append(f"Rejimle uyumlu ({regime_trend})")
        else:
            score -= 18
            counter_trend = True
            reasons.append(f"⚠️ Karşı-trend ({regime_trend} rejimde {decision_raw})")

    # Seri davranışı — momentum kararı trend serisinde daha güvenilir
    if behavior == "trend" and (bullish or bearish):
        score += 5
    elif behavior == "reversal" and (bullish or bearish):
        score -= 4

    # Volatilite rejimi
    if vol_regime == "yüksek":
        score -= 6
        reasons.append("Yüksek volatilite — güven kısıldı")
    elif vol_regime == "düşük":
        score += 3

    # Kalibre olasılık (varsa) — yönden uzaklık güveni artırır
    if p_up is not None:
        conviction = abs(p_up - 0.5) * 2  # 0..1
        score += conviction * 12
        reasons.append(f"Kalibre olasılık %{p_up*100:.0f}")

    score = round(max(0.0, min(100.0, score)), 1)

    # Kapı: düşük güven ya da güçlü karşı-trend → İZLE
    gate_passed = score >= gate_threshold and not (counter_trend and score < 60)
    if not gate_passed:
        decision = "İZLE"
        reasons.append("Kapı: güven yetersiz/karşı-trend → İZLE")
    else:
        decision = decision_raw
        # Güçlü kararı yalnız yüksek güvende koru
        if decision == "GÜÇLÜ AL" and score < 70:
            decision = "AL"

    return ConfidenceResult(score=score, grade=_grade(score), decision=decision,
                            gate_passed=gate_passed, reasons=reasons)

"""Birleşik sinyal — temel (rasyo) + teknik (kompozit) → tek AL/TUT/SAT kararı.

Video felsefesi: temel analiz "NE alınır"ı (kaliteli/ucuz şirket), teknik
analiz "NE ZAMAN"ı (güvenli/ucuz giriş) söyler. Bu modül ikisini tek bir
yüksek-isabet kararına bağlar:

  - **GÜÇLÜ AL**: temel sağlam (rasyo ≥ AL eşiği) VE teknik alım bölgesinde
    VE hiçbir tuzak (aşırı coşku / düşen bıçak) aktif değil.
  - **AL**: temel en azından tutulabilir VE teknik pozitif VE tuzak yok.
  - **SAT/KAÇIN**: teknik bozulmuş (negatif skor, dağıtım) ya da temel zayıf;
    ucuz olması tek başına alım gerekçesi değildir.
  - **TUT/İZLE**: diğer tüm haller — biri olumlu biri değil, ya da düşük güven.

Skorlar 0-100 ölçeğine taşınır ve ağırlıklı birleştirilir (teknik zamanlama
biraz daha ağır, çünkü kötü zamanlama iyi şirkette bile zarar ettirir), ama
NİHAİ KARAR tuzak-farkında kural tabanıyla verilir — saf ortalama değil.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradingagents.analytics.composite import CompositeResult, compute_composite
from tradingagents.analytics.ratio_score import RatioScoreResult, compute_ratio_score


@dataclass
class CombinedResult:
    ok: bool
    ticker: str
    decision: str = "VERİ YOK"   # "GÜÇLÜ AL" | "AL" | "TUT" | "SAT" | "KAÇIN"
    combined_score: float = 0.0  # 0-100 (temel+teknik ağırlıklı)
    fundamental: RatioScoreResult | None = None
    technical: CompositeResult | None = None
    rationale: list = field(default_factory=list)   # kararın Türkçe gerekçeleri
    last_close: float = 0.0
    error: str = ""

    @property
    def summary(self) -> str:
        return f"{self.decision} ({self.combined_score:.0f}/100)"


def _tech_0_100(score: float) -> float:
    """Teknik kompozit [-100,+100] → [0,100] (0 nötr = 50)."""
    return max(0.0, min(100.0, 50 + score / 2))


def combined_signal(
    ticker: str,
    df: pd.DataFrame | None = None,
    inflation_pct: float | None = None,
    info: dict | None = None,
    income=None,
) -> CombinedResult:
    """Temel + teknik birleşik kararını üretir. Asla istisna fırlatmaz.

    Veri parametreleri (df/info/income) enjekte edilebilir (test/önbellek);
    verilmezse ilgili motor kendi verisini yfinance'ten çeker.
    """
    tech = compute_composite(ticker, df)
    fund = compute_ratio_score(ticker, inflation_pct, info=info, income=income)

    if not tech.ok and not fund.ok:
        return CombinedResult(False, ticker,
                              error=f"Veri yok (teknik: {tech.error}; temel: {fund.error})",
                              fundamental=fund, technical=tech)

    rationale: list[str] = []

    # Skorları 0-100'e taşı; eksik olan tarafı nötr (50) say ama gerekçede belirt
    tech_score = _tech_0_100(tech.score) if tech.ok else 50.0
    fund_score = fund.score if fund.ok else 50.0
    if not tech.ok:
        rationale.append(f"Teknik veri yok ({tech.error}) — nötr kabul edildi.")
    if not fund.ok:
        rationale.append(f"Temel veri yok ({fund.error}) — nötr kabul edildi.")

    # Teknik zamanlama biraz daha ağır (0.55), temel kalite 0.45
    combined = round(0.55 * tech_score + 0.45 * fund_score, 1)

    # Kural tabanlı nihai karar (tuzak-farkında)
    trap = bool(tech.ok and (tech.guards.get("asiri_cosku") or tech.guards.get("dusen_bicak")))
    tech_buy = tech.ok and tech.score >= 20
    tech_strong_buy = tech.ok and tech.score >= 50
    tech_sell = tech.ok and tech.score <= -20
    fund_ok = fund.ok and fund.verdict in ("AL", "TUT")
    fund_strong = fund.ok and fund.verdict == "AL"
    fund_weak = fund.ok and fund.verdict == "SAT"

    if tech.ok:
        rationale.append(f"Teknik: {tech.verdict} ({tech.score:+.0f}/100, güven {tech.confidence}).")
    if fund.ok:
        rationale.append(f"Temel: {fund.verdict} ({fund.score:.0f}/100"
                         + (", finansal şirket" if fund.is_financial else "") + ").")
    for w in (tech.warnings if tech.ok else []):
        rationale.append(w)

    if trap:
        # Tuzak aktifse alım yok; teknik zaten sat tarafındaysa SAT
        decision = "SAT" if tech_sell else "KAÇIN"
        rationale.append("Tuzak filtresi aktif → alım engellendi (video kuralı).")
    elif tech_sell or fund_weak:
        decision = "SAT"
        if tech_sell:
            rationale.append("Teknik bozulma/para çıkışı → satış tarafı.")
        if fund_weak:
            rationale.append("Temel zayıf — ucuzluk tek başına alım gerekçesi değil.")
    elif tech_strong_buy and fund_strong:
        decision = "GÜÇLÜ AL"
        rationale.append("Temel sağlam + teknik güçlü alım bölgesi + tuzak yok → yüksek konviksiyon.")
    elif tech_buy and fund_ok:
        decision = "AL"
        rationale.append("Temel tutulabilir + teknik pozitif + tuzak yok.")
    else:
        decision = "TUT"
        rationale.append("Temel ve teknik tam hizalı değil ya da güven düşük → izle, bekle.")

    return CombinedResult(
        ok=True, ticker=ticker, decision=decision, combined_score=combined,
        fundamental=fund if fund.ok else None,
        technical=tech if tech.ok else None,
        rationale=rationale,
        last_close=tech.last_close if tech.ok else 0.0,
    )

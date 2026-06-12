"""Rasyo puanlama motoru — video kriterlerine dayalı temel skorlama + AL/SAT.

Bu modül, kullanıcının video notlarından çıkan KESİN eşikleri deterministik
bir puana (0-100) ve net bir karara (AL / TUT / SAT) çevirir. "Belli bir
puana ulaşan hisse alınır/satılır" mantığının çekirdeğidir.

Kriterler ve bantlar (videolardan birebir):
  - **FD/FAVÖK (EV/EBITDA):** 5-7 ideal/ucuz · 15-20+ pahalı · negatif (zarar)
    değerlenemez. Banka/sigortada KULLANILMAZ.
  - **Cari Oran:** <1 iflas/bedelli riski · 1.5-2.5 ideal · >3 hantal/atıl.
  - **Net Borç/FAVÖK:** negatif veya 0-1.5 mükemmel · 3-4 riskli · >4 kötü.
  - **ROE (reel):** ülke enflasyonunun ÜZERİNDE olmalı; altındaysa şirket reel
    olarak erimektedir. Puan, ROE − enflasyon farkına göre verilir.
  - **Esas Faaliyet Karı (EFK) büyümesi:** en güvenilir veri; tek seferlik
    gelirleri dışlar. Esas faaliyet zararda ise ağır ceza.
  - **Net Satış Büyümesi:** enflasyona göre reel büyüme (nominal yanıltıcıdır).

Banka/sigorta gibi finansallarda FD/FAVÖK, Net Borç/FAVÖK ve Cari Oran
anlamsızdır; bu durumda kriter seti daralır ve kalanlar yeniden ağırlıklanır.

Felsefe (video): temel analiz "NE alınır"ı söyler, teknik "NE ZAMAN"ı. Bu
yüzden buradaki karar bir *kalite/değer* kararıdır; gerçek alım zamanlaması
:func:`tradingagents.analytics.combined.combined_signal` ile teknikle
birleştirilir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yfinance as yf

from tradingagents.analytics.fundamental_score import _col

# Türkiye yıllık TÜFE varsayılanı (reel ROE/büyüme eşiği için). Canlı değer
# Streamlit/Makro tarafından EVDS'den geçilebilir; env ile de ayarlanabilir.
DEFAULT_TR_INFLATION = float(os.environ.get("TR_INFLATION_PCT", "35.0"))

# Sektör adında bunlardan biri geçiyorsa FD/FAVÖK & borç/cari kriterleri atlanır.
_FINANCIAL_HINTS = ("financ", "bank", "insur", "sigorta", "banka", "leasing",
                    "factor", "holding bank")

# (kriter, ağırlık) — sanayi/ticaret şirketleri için. Video vurgusu: EFK ve ROE en kritik.
_WEIGHTS_INDUSTRIAL = {
    "roe_real": 22, "ev_ebitda": 20, "net_debt_ebitda": 18,
    "current_ratio": 15, "operating_growth": 15, "revenue_growth": 10,
}
# Finansallarda değerleme/borç/cari kriterleri düşer; kalanlar yeniden ağırlıklanır.
_WEIGHTS_FINANCIAL = {
    "roe_real": 40, "operating_growth": 25, "revenue_growth": 20, "pb_ratio": 15,
}


@dataclass
class Criterion:
    name: str            # gösterim adı
    value: float | None  # ham rasyo değeri
    score: float         # [0, 1] bant skoru
    weight: int          # ağırlık (puan)
    band: str            # hangi banda düştü (Türkçe)
    note: str = ""       # kısa açıklama


@dataclass
class RatioScoreResult:
    ok: bool
    ticker: str
    score: float = 0.0          # 0-100
    verdict: str = "VERİ YOK"   # "AL" | "TUT" | "SAT" | "VERİ YOK"
    is_financial: bool = False
    criteria: list = field(default_factory=list)  # Criterion listesi
    inflation_pct: float = DEFAULT_TR_INFLATION
    company: str = ""
    sector: str = ""
    error: str = ""

    @property
    def summary(self) -> str:
        return f"{self.score:.0f}/100 → {self.verdict}"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── Bant skorlayıcıları (video eşikleri) ─────────────────────────────────────

def _score_ev_ebitda(v: float | None) -> tuple[float, str]:
    if v is None:
        return 0.0, "veri yok"
    if v <= 0:
        return 0.0, "negatif FAVÖK — değerlenemiyor (zarar)"
    if v <= 7:
        return 1.0, f"{v:.1f} — ideal/ucuz (5-7 bandı)"
    if v <= 10:
        return 0.7, f"{v:.1f} — makul"
    if v <= 15:
        return 0.4, f"{v:.1f} — pahalıya kayıyor"
    if v <= 20:
        return 0.15, f"{v:.1f} — pahalı"
    return 0.0, f"{v:.1f} — aşırı pahalı (>20)"


def _score_current_ratio(v: float | None) -> tuple[float, str]:
    if v is None:
        return 0.0, "veri yok"
    if v < 1:
        return 0.0, f"{v:.2f} — iflas/bedelli sermaye riski (<1)"
    if v < 1.5:
        return 0.6, f"{v:.2f} — kabul edilebilir"
    if v <= 2.5:
        return 1.0, f"{v:.2f} — ideal güvenli bölge (1.5-2.5)"
    if v <= 3:
        return 0.8, f"{v:.2f} — sağlam ama gevşek"
    return 0.5, f"{v:.2f} — hantal/atıl nakit (>3)"


def _score_net_debt_ebitda(v: float | None) -> tuple[float, str]:
    if v is None:
        return 0.0, "veri yok"
    if v < 0:
        return 1.0, f"{v:.2f} — net nakit pozisyonu (mükemmel)"
    if v <= 1.5:
        return 1.0, f"{v:.2f} — çok düşük borç (mükemmel)"
    if v <= 3:
        return 0.6, f"{v:.2f} — yönetilebilir"
    if v <= 4:
        return 0.3, f"{v:.2f} — riskli sınır (3-4)"
    return 0.0, f"{v:.2f} — yüksek borç yükü (>4)"


def _score_roe_real(roe_pct: float | None, inflation: float) -> tuple[float, str]:
    if roe_pct is None:
        return 0.0, "veri yok"
    spread = roe_pct - inflation
    if spread >= 20:
        s = 1.0
    elif spread >= 0:
        s = 0.55 + 0.45 * spread / 20
    else:
        s = _clamp(0.55 * (1 + spread / 20))
    tag = "enflasyon üstü (reel kâr)" if spread >= 0 else "enflasyon ALTI — reel erime"
    return round(s, 2), f"ROE %{roe_pct:.0f} vs enflasyon %{inflation:.0f} → reel %{spread:+.0f} ({tag})"


def _score_real_growth(growth_pct: float | None, inflation: float, label: str) -> tuple[float, str]:
    if growth_pct is None:
        return 0.0, "veri yok"
    real = growth_pct - inflation
    s = _clamp((real + 15) / 30)
    tag = "reel büyüme" if real >= 0 else "reel daralma"
    return round(s, 2), f"{label} %{growth_pct:.0f} (reel %{real:+.0f} — {tag})"


def _score_pb(v: float | None) -> tuple[float, str]:
    """Finansallar için PD/DD: düşük = ucuz (banka değerlemesinin ana çarpanı)."""
    if v is None or v <= 0:
        return 0.0, "veri yok"
    if v <= 0.8:
        return 1.0, f"{v:.2f} — defter değerinin altında (ucuz)"
    if v <= 1.2:
        return 0.8, f"{v:.2f} — makul"
    if v <= 2:
        return 0.5, f"{v:.2f} — primli"
    return 0.2, f"{v:.2f} — pahalı"


def _is_financial(sector: str, industry: str) -> bool:
    blob = f"{sector} {industry}".casefold()
    return any(h in blob for h in _FINANCIAL_HINTS)


def compute_ratio_score(
    ticker: str,
    inflation_pct: float | None = None,
    info: dict | None = None,
    income=None,
) -> RatioScoreResult:
    """Hissenin rasyo puanını ve AL/TUT/SAT kararını hesaplar.

    ``info``/``income`` enjekte edilebilir (test/önbellek); verilmezse yfinance'ten
    çekilir. Asla istisna fırlatmaz.
    """
    inflation = DEFAULT_TR_INFLATION if inflation_pct is None else float(inflation_pct)
    if info is None or income is None:
        try:
            tk = yf.Ticker(ticker)
            info = info if info is not None else (tk.info or {})
            income = income if income is not None else tk.income_stmt
        except Exception as exc:  # noqa: BLE001
            return RatioScoreResult(False, ticker, error=f"Temel veri alınamadı: {exc}",
                                    inflation_pct=inflation)
    info = info or {}

    sector = info.get("sector") or ""
    industry = info.get("industry") or ""
    is_fin = _is_financial(sector, industry)

    # Ham rasyolar
    ev_ebitda = info.get("enterpriseToEbitda")
    current_ratio = info.get("currentRatio")
    roe = info.get("returnOnEquity")
    roe_pct = roe * 100 if isinstance(roe, (int, float)) else None
    pb = info.get("priceToBook")
    rev_growth = info.get("revenueGrowth")
    rev_growth_pct = rev_growth * 100 if isinstance(rev_growth, (int, float)) else None

    # Net Borç / FAVÖK
    total_debt = info.get("totalDebt")
    total_cash = info.get("totalCash")
    ebitda = info.get("ebitda")
    net_debt_ebitda = None
    if isinstance(ebitda, (int, float)) and ebitda:
        if isinstance(total_debt, (int, float)):
            net_debt = total_debt - (total_cash or 0)
            net_debt_ebitda = net_debt / ebitda

    # Esas Faaliyet Karı (EFK) büyümesi — gelir tablosundan
    op_cur = _col(income, ("Operating Income", "Total Operating Income As Reported",
                           "Operating Revenue"))
    op_prev = _col(income, ("Operating Income", "Total Operating Income As Reported",
                            "Operating Revenue"), 1)
    op_growth_pct = None
    if op_cur is not None and op_prev not in (None, 0):
        op_growth_pct = (op_cur - op_prev) / abs(op_prev) * 100

    # Net satış büyümesi — info yoksa gelir tablosundan
    if rev_growth_pct is None:
        rev_cur = _col(income, ("Total Revenue", "Operating Revenue"))
        rev_prev = _col(income, ("Total Revenue", "Operating Revenue"), 1)
        if rev_cur is not None and rev_prev not in (None, 0):
            rev_growth_pct = (rev_cur - rev_prev) / abs(rev_prev) * 100

    # Kriterleri kur
    weights = _WEIGHTS_FINANCIAL if is_fin else _WEIGHTS_INDUSTRIAL
    criteria: list[Criterion] = []

    def add(key: str, name: str, value, scorer_result: tuple[float, str]):
        if key not in weights:
            return
        s, band = scorer_result
        criteria.append(Criterion(name, value, s, weights[key], band))

    roe_s, roe_band = _score_roe_real(roe_pct, inflation)
    add("roe_real", "Öz Sermaye Kârlılığı (reel ROE)", roe_pct, (roe_s, roe_band))

    # EFK büyümesi: esas faaliyet zararda ise skoru sıfırla (video: en kritik risk)
    if op_cur is not None and op_cur <= 0:
        add("operating_growth", "Esas Faaliyet Kârı büyümesi", op_cur,
            (0.0, "esas faaliyet ZARARDA — yüksek risk"))
    else:
        add("operating_growth", "Esas Faaliyet Kârı büyümesi", op_growth_pct,
            _score_real_growth(op_growth_pct, inflation, "EFK büyümesi"))

    add("revenue_growth", "Net Satış büyümesi", rev_growth_pct,
        _score_real_growth(rev_growth_pct, inflation, "Satış büyümesi"))
    add("ev_ebitda", "FD/FAVÖK (değerleme)", ev_ebitda, _score_ev_ebitda(ev_ebitda))
    add("net_debt_ebitda", "Net Borç/FAVÖK", net_debt_ebitda,
        _score_net_debt_ebitda(net_debt_ebitda))
    add("current_ratio", "Cari Oran (likidite)", current_ratio,
        _score_current_ratio(current_ratio))
    add("pb_ratio", "PD/DD (banka değerlemesi)", pb, _score_pb(pb))

    # Puan: yalnız verisi olan kriterlerin ağırlığı paydaya girer (eksik veri ceza değil)
    have_data = [c for c in criteria if c.value is not None or "ZARARDA" in c.band]
    if not have_data:
        return RatioScoreResult(False, ticker, is_financial=is_fin, criteria=criteria,
                                inflation_pct=inflation, sector=sector,
                                company=info.get("longName") or info.get("shortName") or "",
                                error="Hiçbir rasyo verisi alınamadı.")
    total_w = sum(c.weight for c in have_data)
    score = sum(c.weight * c.score for c in have_data) / total_w * 100

    if score >= 65:
        verdict = "AL"
    elif score >= 45:
        verdict = "TUT"
    else:
        verdict = "SAT"

    return RatioScoreResult(
        ok=True, ticker=ticker, score=round(score, 1), verdict=verdict,
        is_financial=is_fin, criteria=criteria, inflation_pct=inflation,
        company=info.get("longName") or info.get("shortName") or "",
        sector=sector,
    )


def format_ratio_brief(ticker: str, inflation_pct: float | None = None,
                       info: dict | None = None, income=None) -> str:
    """LLM'e (Fundamentals Analyst) enjekte edilecek markdown rasyo brifi."""
    res = compute_ratio_score(ticker, inflation_pct, info=info, income=income)
    if not res.ok:
        return f"<Rasyo puanı üretilemedi: {res.error}>"
    head = f"RASYO PUANI (video kriterleri) — {ticker}"
    if res.company:
        head += f" ({res.company}, {res.sector})"
    lines = [
        head,
        f"Skor: {res.score:.0f}/100 → KARAR: {res.verdict}"
        + (" · finansal şirket (değerleme/borç kriterleri uyarlanmış)" if res.is_financial else ""),
        f"Enflasyon eşiği (reel hesaplar için): %{res.inflation_pct:.0f}",
        "",
        "Kriter bantları:",
    ]
    for c in res.criteria:
        lines.append(f"  - {c.name} [{c.weight}p]: {c.band} (skor {c.score:.2f})")
    lines.append("")
    lines.append("Not: bu bir KALİTE/DEĞER kararıdır (ne alınmalı). Alım ZAMANLAMASI "
                 "için teknik kompozit skora bakılmalı (video: önce ne, sonra ne zaman).")
    return "\n".join(lines)

"""Temel sağlamlık skoru — Piotroski F-Score uyarlaması + BIST rasyoları.

yfinance'in bilanço/gelir/nakit akışı tablolarından deterministik bir temel
skor üretir. Klasik Piotroski F-Score'un (0-9) uygulanabilir kalemleri +
değerleme rasyoları (F/K, PD/DD, FD/FAVÖK) hesaplanır. Türkiye bağlamı:
yüksek enflasyonda nominal büyüme yanıltıcıdır — skor büyümeyi değil
KÂRLILIK/KALDIRAÇ/VERİMLİLİK yönünü ödüllendirir; değerleme rasyoları ise
yorumsuz raporlanır (sektör medyanı olmadan tek başına "ucuz/pahalı" demek
yanıltıcı olur, o yorum LLM analistine/karşılaştırma sayfasına bırakılır).

Asla istisna fırlatmaz; eksik kalemler "veri yok" diye işaretlenir ve skor
paydası mevcut kriter sayısına göre kurulur (eksik veri ceza değildir).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf


@dataclass
class FundamentalResult:
    ok: bool
    ticker: str
    score: int = 0              # sağlanan kriter sayısı
    max_score: int = 0          # değerlendirilebilen kriter sayısı
    criteria: dict = field(default_factory=dict)   # ad -> (bool | None, açıklama)
    ratios: dict = field(default_factory=dict)     # ad -> değer (float | None)
    company: str = ""
    sector: str = ""
    error: str = ""

    @property
    def grade(self) -> str:
        """Skoru kaba bir nota çevirir (Streamlit rozeti için)."""
        if self.max_score == 0:
            return "veri yok"
        pct = self.score / self.max_score
        if pct >= 0.78:
            return "Sağlam"
        if pct >= 0.55:
            return "Orta"
        return "Zayıf"


def _col(frame: pd.DataFrame | None, names: tuple[str, ...], col: int = 0) -> float | None:
    """Finansal tablodan ilk eşleşen satırın ``col``. sütun değerini döndürür."""
    if frame is None or frame.empty or frame.shape[1] <= col:
        return None
    for name in names:
        if name in frame.index:
            val = frame.iloc[frame.index.get_loc(name), col]
            try:
                return None if pd.isna(val) else float(val)
            except (TypeError, ValueError):
                return None
    return None


def compute_fundamental_score(ticker: str) -> FundamentalResult:
    """Hissenin temel sağlamlık skorunu hesaplar (yfinance, anahtar gerektirmez)."""
    try:
        tk = yf.Ticker(ticker)
        income = tk.income_stmt
        balance = tk.balance_sheet
        cashflow = tk.cashflow
        info = tk.info or {}
    except Exception as exc:  # noqa: BLE001
        return FundamentalResult(False, ticker, error=f"Temel veri alınamadı: {exc}")

    if (income is None or income.empty) and (balance is None or balance.empty):
        return FundamentalResult(False, ticker, error="Finansal tablolar boş döndü.")

    criteria: dict[str, tuple[bool | None, str]] = {}

    def add(name: str, cur: float | None, prev: float | None, op: str, desc: str):
        """Kriteri değerlendirir: veri eksikse None (skor paydasına girmez)."""
        if cur is None or (op != "pos" and prev is None):
            criteria[name] = (None, f"{desc} — veri yok")
            return
        if op == "pos":
            ok = cur > 0
        elif op == "up":
            ok = cur > prev
        else:  # "down"
            ok = cur < prev
        criteria[name] = (ok, desc)

    ni_cur = _col(income, ("Net Income", "Net Income Common Stockholders"))
    ni_prev = _col(income, ("Net Income", "Net Income Common Stockholders"), 1)
    assets_cur = _col(balance, ("Total Assets",))
    assets_prev = _col(balance, ("Total Assets",), 1)
    ocf = _col(cashflow, ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"))
    debt_cur = _col(balance, ("Total Debt", "Long Term Debt"))
    debt_prev = _col(balance, ("Total Debt", "Long Term Debt"), 1)
    cur_assets = _col(balance, ("Current Assets", "Total Current Assets"))
    cur_liab = _col(balance, ("Current Liabilities", "Total Current Liabilities"))
    cur_assets_prev = _col(balance, ("Current Assets", "Total Current Assets"), 1)
    cur_liab_prev = _col(balance, ("Current Liabilities", "Total Current Liabilities"), 1)
    revenue_cur = _col(income, ("Total Revenue", "Operating Revenue"))
    revenue_prev = _col(income, ("Total Revenue", "Operating Revenue"), 1)
    gross_cur = _col(income, ("Gross Profit",))
    gross_prev = _col(income, ("Gross Profit",), 1)

    # 1-2) Kârlılık
    add("Net kâr pozitif", ni_cur, None, "pos", "son yıl net kâr > 0")
    add("Faaliyet nakit akışı pozitif", ocf, None, "pos", "operasyon nakit üretiyor")
    # 3) Tahakkuk kalitesi: OCF > net kâr (kâğıt üstü kâr değil)
    if ocf is not None and ni_cur is not None:
        criteria["Nakit akışı kârı aşıyor"] = (ocf > ni_cur, "OCF > net kâr (kazanç kalitesi)")
    else:
        criteria["Nakit akışı kârı aşıyor"] = (None, "OCF/net kâr — veri yok")
    # 4) ROA iyileşiyor
    roa_cur = (ni_cur / assets_cur) if (ni_cur is not None and assets_cur) else None
    roa_prev = (ni_prev / assets_prev) if (ni_prev is not None and assets_prev) else None
    add("ROA iyileşiyor", roa_cur, roa_prev, "up", "aktif kârlılığı yıllık artışta")
    # 5) Kaldıraç azalıyor (TL faiz ortamında kritik)
    lev_cur = (debt_cur / assets_cur) if (debt_cur is not None and assets_cur) else None
    lev_prev = (debt_prev / assets_prev) if (debt_prev is not None and assets_prev) else None
    add("Borçluluk azalıyor", lev_cur, lev_prev, "down", "borç/aktif oranı düşüşte")
    # 6) Likidite iyileşiyor
    ratio_cur = (cur_assets / cur_liab) if (cur_assets is not None and cur_liab) else None
    ratio_prev = (cur_assets_prev / cur_liab_prev) if (cur_assets_prev is not None and cur_liab_prev) else None
    add("Cari oran iyileşiyor", ratio_cur, ratio_prev, "up", "kısa vadeli likidite güçleniyor")
    # 7) Brüt marj iyileşiyor (enflasyon maliyetini fiyata geçirebiliyor mu)
    gm_cur = (gross_cur / revenue_cur) if (gross_cur is not None and revenue_cur) else None
    gm_prev = (gross_prev / revenue_prev) if (gross_prev is not None and revenue_prev) else None
    add("Brüt marj iyileşiyor", gm_cur, gm_prev, "up", "fiyatlama gücü korunuyor")
    # 8) Aktif devir hızı iyileşiyor
    turn_cur = (revenue_cur / assets_cur) if (revenue_cur is not None and assets_cur) else None
    turn_prev = (revenue_prev / assets_prev) if (revenue_prev is not None and assets_prev) else None
    add("Aktif verimliliği iyileşiyor", turn_cur, turn_prev, "up", "satış/aktif devri artışta")

    evaluated = [ok for ok, _ in criteria.values() if ok is not None]
    score = sum(1 for ok in evaluated if ok)

    ratios = {
        "F/K": info.get("trailingPE"),
        "İleri F/K": info.get("forwardPE"),
        "PD/DD": info.get("priceToBook"),
        "FD/FAVÖK": info.get("enterpriseToEbitda"),
        "Temettü verimi %": (info.get("dividendYield") or 0) * 100 if info.get("dividendYield") else None,
        "ROE %": (info.get("returnOnEquity") or 0) * 100 if info.get("returnOnEquity") else None,
        "Net marj %": (info.get("profitMargins") or 0) * 100 if info.get("profitMargins") else None,
        "Borç/Özkaynak": info.get("debtToEquity"),
    }
    ratios = {k: (round(v, 2) if isinstance(v, (int, float)) else None) for k, v in ratios.items()}

    return FundamentalResult(
        ok=True, ticker=ticker, score=score, max_score=len(evaluated),
        criteria=criteria, ratios=ratios,
        company=info.get("longName") or info.get("shortName") or "",
        sector=info.get("sector") or "",
    )


def format_fundamental_brief(ticker: str) -> str:
    """LLM'e (Fundamentals Analyst) enjekte edilecek markdown brif."""
    res = compute_fundamental_score(ticker)
    if not res.ok:
        return f"<Deterministik temel skor üretilemedi: {res.error}>"
    lines = [
        f"DETERMİNİSTİK TEMEL SKOR — {ticker}"
        + (f" ({res.company}, {res.sector})" if res.company else ""),
        f"Sağlamlık: {res.score}/{res.max_score} → {res.grade}",
        "",
        "Kriterler:",
    ]
    for name, (ok, desc) in res.criteria.items():
        mark = "✓" if ok else ("✗" if ok is False else "·")
        lines.append(f"  {mark} {name} — {desc}")
    lines += ["", "Değerleme rasyoları (sektör karşılaştırması yapılmadı — yorum sana ait):"]
    for name, val in res.ratios.items():
        lines.append(f"  - {name}: {val if val is not None else 'veri yok'}")
    return "\n".join(lines)

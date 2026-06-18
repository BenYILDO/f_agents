"""Karar açıklayıcı — snapshot satırını tek cümlelik 'neden'e ve özet şeridine indirger.

Ekranlardaki "Karar (v3)" sütunu nihai cevabı verir ama *neden* o karar
verildiğini göstermez; kullanıcı "al mı sat mı" diye bakınca İZLE/Nötr yığını
kafa karıştırır. Bu modül yalnız snapshot'ta zaten var olan alanlardan
(``signals.confidence`` + ``signals.risk`` + ham karar/rejim) okunabilir bir
gerekçe üretir — yani cron'u yeniden çalıştırmaya gerek yok, eldeki veriyle çalışır.

Saf/ağsız; Streamlit'e bağımlı değildir.
"""

from __future__ import annotations

from typing import Iterable

_BULLISH = {"GÜÇLÜ AL", "AL"}
_BEARISH = {"SAT", "KAÇIN"}


def _fields(snap: dict) -> dict:
    sig = snap.get("signals") or {}
    conf = sig.get("confidence") or {}
    risk = sig.get("risk") or {}
    gated = (conf.get("gated") or snap.get("decision") or "—")
    return {
        "gated": gated,
        "raw": snap.get("decision") or "—",
        "score": conf.get("score"),
        "grade": conf.get("grade") or "",
        "regime": conf.get("regime") or "",
        "behavior": conf.get("behavior") or "",
        "gate_passed": conf.get("gate_passed"),
        "illiquid": bool(risk.get("illiquid")),
        "liquidity": risk.get("liquidity") or "",
    }


def explain_decision(snap: dict) -> str:
    """Snapshot satırını **tek cümlelik gerekçeye** indirger (tabloda 'Neden' sütunu).

    Öncelik sırası: sert kapı (illikidite) → karşı-trend → düşük güven → net karar.
    """
    f = _fields(snap)
    gated, raw = f["gated"].upper(), f["raw"].upper()
    score, grade, regime = f["score"], f["grade"], f["regime"]

    # 1) Sert kapı — illikidite (alımı ezer)
    if f["illiquid"]:
        return "⛔ İllikit (düşük hacim) — alım riskli, kayma/manipülasyon → bekle"

    # 2) Karşı-trend — ham sinyal rejime ters
    counter = ((raw in _BULLISH and regime == "ayı")
               or (raw in _BEARISH and regime == "boğa"))
    if gated == "İZLE" and counter:
        return f"⚠️ {regime.capitalize()} piyasaya karşı {f['raw']} sinyali — temkin, bekle"

    # 3) Net alım / satım
    if gated in ("AL", "GÜÇLÜ AL"):
        tail = f" · {regime} rejimle uyumlu" if regime == "boğa" else ""
        return f"🟢 Alım sinyali — güven {grade or '—'}{tail}"
    if gated == "SAT":
        tail = f" · {regime} rejimle uyumlu" if regime == "ayı" else ""
        return f"🔴 Satış baskısı — güven {grade or '—'}{tail}"
    if gated == "TUT":
        return "⏸️ Dengeli görünüm — pozisyonu tut, acele etme"

    # 4) İZLE — neden net değil? (düşük güven en olası neden)
    if gated == "İZLE":
        if score is not None and score < 45:
            return f"⏸️ Güven düşük (%{score:.0f}) — sinyal net değil, bekle"
        return "⏸️ Yöntemler çelişiyor/zayıf — net sinyal yok, bekle"

    return "—"


def decision_summary(snaps: Iterable[dict]) -> str:
    """Satırları **üstte tek satırlık özet şeridine** indirger (markdown döndürür)."""
    al, sat, watch = [], [], []
    for s in snaps:
        f = _fields(s)
        g = f["gated"].upper()
        name = (s.get("ticker") or "").replace(".IS", "")
        if g in _BULLISH:
            al.append(name)
        elif g in _BEARISH:
            sat.append(name)
        else:  # İZLE / TUT / NÖTR / —
            watch.append(name)

    def _grp(names: list[str], emoji: str, label: str) -> str:
        if not names:
            return ""
        shown = [n for n in names if n]
        tail = f" ({', '.join(shown)})" if 0 < len(shown) <= 6 else ""
        return f"{emoji} **{len(names)} {label}**{tail}"

    parts = [p for p in (_grp(al, "🟢", "AL"), _grp(sat, "🔴", "SAT"),
                         _grp(watch, "⏸️", "bekle")) if p]
    if not parts:
        return ""
    head = " · ".join(parts)
    note = ("Güçlü alım fırsatı yok — çoğunlukla bekle."
            if not al else "Alım adayı var; aşağıdaki 'Neden' sütununu oku.")
    return f"📌 **Bugünkü özet:** {head} — {note}"

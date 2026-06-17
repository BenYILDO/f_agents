"""Ekranlar arası paylaşılan stiller/rozetler — Portföyüm & BIST 30 Tarayıcı.

Karar/durum/mutabakat rozetlerini ve sıralama yardımcılarını tek yerde tutar.
"""

from __future__ import annotations

# Birleşik karar → (renk, emoji)
DECISION_STYLE: dict[str, tuple[str, str]] = {
    "GÜÇLÜ AL": ("#15803d", "🟢"),
    "AL": ("#16a34a", "🟢"),
    "TUT": ("#6b7280", "⚪"),
    "SAT": ("#dc2626", "🔴"),
    "KAÇIN": ("#991b1b", "🔴"),
    "VERİ YOK": ("#9ca3af", "⚠️"),
}

# Çoklu-yöntem mutabakatı seviyesi → rozet
AGREE_BADGE: dict[str, str] = {
    "güçlü": "🟢 güçlü mutabakat",
    "kısmi": "🟡 kısmi mutabakat",
    "çelişki": "🔴 çelişki",
    "nötr": "⚪ nötr",
}

# Dip-Al güncel durumu → rozet
STATUS_BADGE: dict[str, str] = {"AL": "🟢 AL", "SAT": "🔴 SAT", "NÖTR": "⚪ Nötr"}

# Sıralama: en boğa karardan en ayıya (AL sinyalleri üstte).
_DECISION_RANK = {"GÜÇLÜ AL": 0, "AL": 1, "TUT": 2, "KAÇIN": 3, "SAT": 4, "VERİ YOK": 5}


def decision_rank(decision: str | None) -> int:
    return _DECISION_RANK.get(decision or "VERİ YOK", 5)


def pnl_text(pnl_pct: float | None) -> str:
    if pnl_pct is None:
        return "—"
    return f"{'🟢 +' if pnl_pct >= 0 else '🔴 '}{pnl_pct:.2f}%"


def agree_level_of(snapshot: dict) -> str:
    """Snapshot'taki 'agreement' alanından seviye etiketini ('güçlü'…) çıkarır."""
    return (snapshot.get("agreement") or "").split(":", 1)[0].strip()

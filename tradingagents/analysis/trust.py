"""Güven katmanı — saf, ağsız, test edilebilir hesaplar.

Bir AL/SAT sinyaline güven, "kanıtlanabilir olmasıyla" artar. Üç ölçüt:

  1. :func:`agreement`        — çoklu-yöntem mutabakatı. Bağımsız yöntemler
     (teknik / rasyo / dip-stratejisi / ML) aynı yöne işaret ediyorsa güven
     yüksek; çelişiyorsa kırmızı bayrak. Tek göstergeye değil, uyuma güvenir.
  2. :func:`signal_stability` — sinyal istikrarı. Son N gözlemde sinyal sürekli
     AL↔SAT zıplıyorsa güven düşer (flip-flop cezası); istikrarlıysa yükselir.
  3. :func:`track_record`     — sinyal karnesi. Geçmiş AL/SAT sinyallerinin
     gerçekleşen ileri getirisi + isabet oranı ("sistem AL dediğinde %X tutmuş").

Tümü saf fonksiyon: ağ/Supabase/yfinance kullanmaz, böylece deterministik
test edilir. Geçmişe dayanan ölçütler (istikrar/karne) veri birikene kadar
"yetersiz veri" döndürür — sessizce uydurmaz.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Yön kodları
UP, FLAT, DOWN = 1, 0, -1
_DIR_TXT = {UP: "AL", FLAT: "NÖTR", DOWN: "SAT"}


# ── 1) Çoklu-yöntem mutabakatı ──────────────────────────────────────────────
def agreement(votes: dict[str, int]) -> tuple[str, str]:
    """Yöntem oylarından mutabakat seviyesi + Türkçe özet üretir.

    ``votes``: yöntem adı → yön ({+1 AL, 0 nötr, -1 SAT}). Dönen seviye:
    ``"güçlü"`` (tüm yöntemler aynı yönde), ``"kısmi"`` (bazıları nötr ama
    çelişki yok), ``"çelişki"`` (zıt yönler var), ``"nötr"`` (hepsi nötr).
    """
    parts = ", ".join(f"{name}:{_DIR_TXT.get(v, '?')}" for name, v in votes.items())
    active = {k: v for k, v in votes.items() if v != FLAT}
    has_up = any(v > 0 for v in active.values())
    has_down = any(v < 0 for v in active.values())

    if not active:
        return "nötr", f"Tüm yöntemler nötr ({parts})."
    if has_up and has_down:
        return "çelişki", f"Yöntemler çelişiyor — temkinli ol ({parts})."
    if len(active) == len(votes):
        return "güçlü", f"Tüm yöntemler aynı yönde ({parts})."
    return "kısmi", f"Kısmi mutabakat — bazı yöntemler nötr ({parts})."


# ── 2) Sinyal istikrarı (flip-flop) ─────────────────────────────────────────
def signal_stability(statuses: list[str]) -> tuple[str, float, int]:
    """Kronolojik (eski→yeni) durum dizisinden istikrar ölçer.

    Dönen: (seviye, skor[0-1], flip_sayısı). Skor = 1 − flip / (n−1); yani
    hiç değişmeyen dizi 1.0, her adımda zıplayan dizi 0.0. Seviye: ``istikrarlı``
    (≥0.7), ``orta`` (0.4–0.7), ``kararsız`` (<0.4), veri azsa ``yetersiz``.
    """
    seq = [s for s in statuses if s]
    n = len(seq)
    if n < 3:
        return "yetersiz", 1.0, 0
    flips = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    score = round(1 - flips / (n - 1), 2)
    if score >= 0.7:
        level = "istikrarlı"
    elif score >= 0.4:
        level = "orta"
    else:
        level = "kararsız"
    return level, score, flips


# ── 3) Sinyal karnesi (track record) ────────────────────────────────────────
def _parse_ts(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def track_record(
    rows: list[dict], horizon_days: float = 1.0, min_samples: int = 5
) -> dict:
    """Geçmiş AL/SAT sinyallerinin ileri getirisini ve isabet oranını ölçer.

    ``rows``: snapshot kayıtları (``ts``, ``status``, ``close`` alanları). Her
    sinyal için, ``horizon_days`` sonrası ilk kaydın kapanışına göre yüzde getiri
    hesaplanır; AL için getiri>0, SAT için getiri<0 "isabet" sayılır. Yatay/eksik
    veride o sinyal atlanır.

    Dönen sözlük::

        {"AL": {"n", "hit_rate", "mean_fwd"}, "SAT": {...}, "ready": bool}

    Yeterli örnek (``min_samples``) birikmeden ``ready=False`` döner — sistem
    kendine güveni veri olmadan ilan etmez.
    """
    recs = []
    for r in rows:
        ts = _parse_ts(r.get("ts"))
        close = r.get("close")
        status = r.get("status")
        if ts is None or close in (None, "") or status not in ("AL", "SAT"):
            continue
        try:
            recs.append((ts, status, float(close)))
        except (TypeError, ValueError):
            continue
    recs.sort(key=lambda x: x[0])

    # İleri getiri için tüm (ts, close) zaman serisini de sıralı tut
    series = sorted(
        (
            (_parse_ts(r.get("ts")), float(r["close"]))
            for r in rows
            if _parse_ts(r.get("ts")) is not None and r.get("close") not in (None, "")
        ),
        key=lambda x: x[0],
    )

    buckets: dict[str, list[float]] = {"AL": [], "SAT": []}
    for ts, status, close0 in recs:
        target = ts + timedelta(days=horizon_days)
        fwd_close = next((c for t, c in series if t >= target), None)
        if fwd_close is None or close0 <= 0:
            continue
        buckets[status].append((fwd_close / close0 - 1) * 100)

    def _summ(status: str, vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "hit_rate": None, "mean_fwd": None}
        wins = sum(1 for v in vals if (v > 0 if status == "AL" else v < 0))
        return {
            "n": len(vals),
            "hit_rate": round(wins / len(vals) * 100, 1),
            "mean_fwd": round(sum(vals) / len(vals), 2),
        }

    out = {s: _summ(s, v) for s, v in buckets.items()}
    out["ready"] = (out["AL"]["n"] + out["SAT"]["n"]) >= min_samples
    out["horizon_days"] = horizon_days
    return out

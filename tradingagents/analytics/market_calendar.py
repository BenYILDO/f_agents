"""BIST piyasa takvimi — seans saatleri, tatiller, açık/kapalı ve veri bayatlığı.

Borsa İstanbul Pay Piyasası sürekli işlem: ~10:00–18:00 (Europe/Istanbul,
sabit UTC+3), Pazartesi–Cuma. Bu modül "piyasa açık mı", "son seans kapanışı ne
zamandı", "elimdeki son bar bayat mı" sorularını cevaplar.

Not: Dini bayram tarihleri (Ramazan/Kurban) yıldan yıla kayar; aşağıdaki tablo
EN İYİ ÇABA ile 2025–2027 için seedlenmiştir, her yıl resmi BIST takviminden
DOĞRULANMALI/GÜNCELLENMELİDİR. Sabit ulusal bayramlar ve seans/hafta-sonu
mantığı güvenilirdir.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Europe/Istanbul")
except Exception:  # noqa: BLE001 — zoneinfo yoksa sabit UTC+3'e düş
    from datetime import timezone
    _IST = timezone(timedelta(hours=3))

OPEN_TIME = time(10, 0)
CLOSE_TIME = time(18, 0)

# Sabit ulusal bayramlar (ay, gün) — BIST kapalı (tam gün).
_FIXED_HOLIDAYS = {
    (1, 1),    # Yılbaşı
    (4, 23),   # Ulusal Egemenlik ve Çocuk Bayramı
    (5, 1),    # Emek ve Dayanışma Günü
    (5, 19),   # Atatürk'ü Anma, Gençlik ve Spor Bayramı
    (7, 15),   # Demokrasi ve Milli Birlik Günü
    (8, 30),   # Zafer Bayramı
    (10, 29),  # Cumhuriyet Bayramı
}

# Dini bayramlar — EN İYİ ÇABA, yıllık doğrula (ISO tarih). Düzenle: yeni yıl ekle.
_RELIGIOUS_HOLIDAYS = {
    # Ramazan Bayramı
    date(2025, 3, 31), date(2025, 4, 1), date(2025, 4, 2),
    date(2026, 3, 20), date(2026, 3, 21), date(2026, 3, 22),
    date(2027, 3, 10), date(2027, 3, 11), date(2027, 3, 12),
    # Kurban Bayramı
    date(2025, 6, 6), date(2025, 6, 7), date(2025, 6, 8), date(2025, 6, 9),
    date(2026, 5, 27), date(2026, 5, 28), date(2026, 5, 29), date(2026, 5, 30),
    date(2027, 5, 16), date(2027, 5, 17), date(2027, 5, 18), date(2027, 5, 19),
}


def _to_ist(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(_IST)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_IST)
    return dt.astimezone(_IST)


def is_holiday(d: date) -> bool:
    return (d.month, d.day) in _FIXED_HOLIDAYS or d in _RELIGIOUS_HOLIDAYS


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and not is_holiday(d)


def is_market_open(dt: datetime | None = None) -> bool:
    d = _to_ist(dt)
    return is_trading_day(d.date()) and OPEN_TIME <= d.time() < CLOSE_TIME


@dataclass
class MarketStatus:
    open: bool
    status: str          # 'açık' | 'seans dışı' | 'hafta sonu' | 'tatil'
    detail: str = ""


def market_status(dt: datetime | None = None) -> MarketStatus:
    d = _to_ist(dt)
    day = d.date()
    if is_holiday(day):
        return MarketStatus(False, "tatil", "Resmi tatil — BIST kapalı.")
    if day.weekday() >= 5:
        return MarketStatus(False, "hafta sonu", "Hafta sonu — BIST kapalı.")
    if is_market_open(d):
        return MarketStatus(True, "açık", f"Seans açık ({OPEN_TIME:%H:%M}–{CLOSE_TIME:%H:%M}).")
    when = "açılış öncesi" if d.time() < OPEN_TIME else "kapanış sonrası"
    return MarketStatus(False, "seans dışı", f"Seans dışı ({when}).")


def previous_trading_day(d: date) -> date:
    cur = d - timedelta(days=1)
    while not is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def last_completed_session(dt: datetime | None = None) -> date:
    """Kapanışı geçmiş en son işlem günü (günlük barın olması gereken tarih)."""
    d = _to_ist(dt)
    today = d.date()
    if is_trading_day(today) and d.time() >= CLOSE_TIME:
        return today
    return previous_trading_day(today)


def is_stale(last_bar_date, dt: datetime | None = None) -> bool:
    """Elimdeki son günlük bar, olması gereken son seanstan eski mi?"""
    if last_bar_date is None:
        return True
    if isinstance(last_bar_date, datetime):
        last_bar_date = last_bar_date.date()
    if isinstance(last_bar_date, str):
        try:
            last_bar_date = date.fromisoformat(last_bar_date[:10])
        except ValueError:
            return True
    return last_bar_date < last_completed_session(dt)

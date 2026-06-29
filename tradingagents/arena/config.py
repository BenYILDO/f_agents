"""Ortak execution fiziği — tüm hesaplar için aynı maliyet/fill kuralları.

Plan (§Ortak execution maliyeti): komisyon ve slippage **profil özelliği değil**,
deney ortamının ortak fizik kurallarıdır; aksi halde hesaplar adil kıyaslanamaz.
Değerler "broker/döneme göre değişir" diye sabit gerçek gibi gömülmez — burada
tek ortak konfigürasyon olarak durur, sezon başında dondurulur.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class ExecutionConfig:
    """Sezon boyunca dondurulan ortak execution sözleşmesi.

    Fiyatlar BIST günlük ``Open`` proxy'sidir (resmi açılış müzayedesi DEĞİL —
    plan §BIST açılış: ``yfinance_daily_open_proxy``). Alış ``open*(1+slippage)``,
    satış ``open*(1-slippage)`` ile doldurulur; komisyon her iki yönde bps.
    """

    initial_capital: float = 100_000.0   # her hesap eşit kasa (TL)
    commission_bps: float = 5.0          # tek-yön komisyon (baz puan, %0.05)
    slippage_bps: float = 10.0           # tek-yön slippage proxy'si (%0.10)
    buying_power_buffer: float = 0.05    # kasanın %5'i tampon — gap'te eksiye düşme
    settlement_same_day: bool = True     # V1 basitleştirme: satış geliri aynı gün kullanılır
    benchmark_ticker: str = "XU100.IS"   # XU100 al-tut, aynı T+1 open konvansiyonu

    # ── Sistemik kill-switch'ler (hiçbir profil kapatamaz, plan §Profil kuralları) ──
    max_account_drawdown: float = 0.35   # hesap tepe-değerinden %35 düşerse yeni alım durur
    stale_bar_days: int = 5              # son bar bu kadar gün bayatsa o gün işlem yok

    def cost_summary(self) -> str:
        return (f"komisyon {self.commission_bps:.0f}bps · slippage "
                f"{self.slippage_bps:.0f}bps · tampon %{self.buying_power_buffer*100:.0f}")

    def to_dict(self) -> dict:
        return asdict(self)


# Sezon başında dondurulacak varsayılan fizik — UI/replay bunu kullanır.
DEFAULT_EXECUTION = ExecutionConfig()


@dataclass(frozen=True)
class SeasonConfig:
    """Bir arena sezonunun değişmez kimliği (plan §season_id ilk günden)."""

    season_id: str = "2026-H2-S1"
    name: str = "2026 İkinci Yarı · Sezon 1"
    universe_version: str = "bist30-2026H1"
    execution: ExecutionConfig = field(default_factory=lambda: DEFAULT_EXECUTION)

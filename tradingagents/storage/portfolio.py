"""Portföy (``holdings``) — ekle/sil/düzenle + maliyet ortalaması ve P&L.

Aynı hisseye birden çok alış satırı tutulabilir; :func:`compute_positions`
bunları sembol bazında **ağırlıklı maliyet ortalamasına** indirger ve güncel
fiyat verildiğinde kâr/zarar (TL ve %), piyasa değeri ve portföy ağırlığını
hesaplar. Hesap çekirdeği (:func:`compute_positions`) saf/ağsızdır → test edilebilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from tradingagents.storage.supabase_client import SupabaseREST

_TABLE = "holdings"


@dataclass
class Position:
    """Bir sembolün toplam pozisyonu (lotlar birleştirilmiş)."""

    ticker: str
    quantity: float
    avg_cost: float
    invested: float          # quantity * avg_cost
    lots: int                # bu sembol için alış satırı sayısı
    last_price: Optional[float] = None
    market_value: Optional[float] = None
    pnl: Optional[float] = None         # market_value - invested (TL)
    pnl_pct: Optional[float] = None     # (last_price / avg_cost - 1) * 100
    weight_pct: Optional[float] = None  # portföy içindeki ağırlık (%)


def _client() -> SupabaseREST:
    return SupabaseREST()


# ── CRUD ────────────────────────────────────────────────────────────────────
def list_holdings() -> list[dict]:
    """Tüm alış satırlarını döndürür (sembol, sonra tarih sırasıyla)."""
    return _client().select(_TABLE, {"order": "ticker.asc,buy_date.asc"})


def add_holding(
    ticker: str,
    quantity: float,
    buy_price: float,
    buy_date: date | str | None = None,
    note: str = "",
) -> dict:
    """Yeni bir alış satırı ekler ve eklenen satırı döndürür."""
    if isinstance(buy_date, date):
        buy_date = buy_date.isoformat()
    row = {
        "ticker": ticker.strip().upper(),
        "quantity": float(quantity),
        "buy_price": float(buy_price),
        "buy_date": buy_date or date.today().isoformat(),
        "note": note or "",
    }
    return _client().insert(_TABLE, row)[0]


def update_holding(holding_id: int, **fields) -> dict:
    """Bir alış satırını günceller (verilen alanları)."""
    if "ticker" in fields and fields["ticker"]:
        fields["ticker"] = str(fields["ticker"]).strip().upper()
    if isinstance(fields.get("buy_date"), date):
        fields["buy_date"] = fields["buy_date"].isoformat()
    return _client().update(_TABLE, fields, {"id": holding_id})[0]


def delete_holding(holding_id: int) -> None:
    """Bir alış satırını siler."""
    _client().delete(_TABLE, {"id": holding_id})


# ── Hesaplar (saf, ağsız) ───────────────────────────────────────────────────
def compute_positions(
    rows: list[dict], prices: dict[str, float] | None = None
) -> list[Position]:
    """Alış satırlarını sembol bazında pozisyonlara indirger.

    ``prices`` verilirse (sembol→güncel fiyat) P&L, piyasa değeri ve ağırlık
    da hesaplanır. Bu fonksiyon ağ kullanmaz → birim test edilebilir.
    """
    prices = prices or {}
    by_ticker: dict[str, list[dict]] = {}
    for r in rows:
        by_ticker.setdefault(str(r["ticker"]).upper(), []).append(r)

    positions: list[Position] = []
    for ticker, lots in sorted(by_ticker.items()):
        qty = sum(float(l["quantity"]) for l in lots)
        invested = sum(float(l["quantity"]) * float(l["buy_price"]) for l in lots)
        avg = invested / qty if qty else 0.0
        pos = Position(
            ticker=ticker,
            quantity=round(qty, 6),
            avg_cost=round(avg, 4),
            invested=round(invested, 2),
            lots=len(lots),
        )
        price = prices.get(ticker)
        if price is not None:
            pos.last_price = round(float(price), 4)
            pos.market_value = round(qty * price, 2)
            pos.pnl = round(qty * price - invested, 2)
            pos.pnl_pct = round((price / avg - 1) * 100, 2) if avg else None
        positions.append(pos)

    # Portföy ağırlığı — yalnız piyasa değeri bilinen pozisyonlar üzerinden
    total_mv = sum(p.market_value for p in positions if p.market_value is not None)
    if total_mv > 0:
        for p in positions:
            if p.market_value is not None:
                p.weight_pct = round(p.market_value / total_mv * 100, 2)
    return positions


def positions(prices: dict[str, float] | None = None) -> list[Position]:
    """Veritabanındaki holdings'i pozisyonlara indirger (ağ: Supabase okuma)."""
    return compute_positions(list_holdings(), prices)

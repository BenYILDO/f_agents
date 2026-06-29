"""Canlı arena durumu — yerel-önce kalıcılık (JSON), para Decimal.

Replay tarihsel/anlıktır; bu modül **canlı sezonun** ileriye-dönük durumunu tutar:
her hesabın nakdi, pozisyonları, nakit defteri, equity geçmişi ve bekleyen emirleri.
Supabase opsiyoneldir — varsayılan yerel JSON dosyası (``data/arena_state.json``),
böylece Supabase olmadan da günlük seans çalışıp birikir (plan §yerel-önce).

Para alanları Decimal'dir (plan §Tasarım kuralları: float yerine numeric/Decimal);
JSON'a string olarak serileştirilir, geri yüklerken Decimal'e çevrilir.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

_DEFAULT_PATH = os.environ.get(
    "ARENA_STATE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "data", "arena_state.json"),
)


def D(x) -> Decimal:
    """Güvenli Decimal — float'tan string üzerinden (ikili kayan hata yok)."""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x if x is not None else 0))


@dataclass
class Position:
    ticker: str
    quantity: int = 0
    avg_cost: Decimal = field(default_factory=lambda: D(0))
    stop: Decimal = field(default_factory=lambda: D(0))
    target: Decimal = field(default_factory=lambda: D(0))
    opened_session: str = ""        # giriş seansı (max-hold için)

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "quantity": self.quantity,
                "avg_cost": str(self.avg_cost), "stop": str(self.stop),
                "target": str(self.target), "opened_session": self.opened_session}

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(ticker=d["ticker"], quantity=int(d["quantity"]),
                   avg_cost=D(d.get("avg_cost")), stop=D(d.get("stop")),
                   target=D(d.get("target")), opened_session=d.get("opened_session", ""))


@dataclass
class PendingOrder:
    ticker: str
    side: str                       # BUY | SELL
    quantity: int
    signal_session: str             # sinyalin üretildiği seans
    reason: str = ""
    stop: Decimal = field(default_factory=lambda: D(0))
    target: Decimal = field(default_factory=lambda: D(0))

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "side": self.side, "quantity": self.quantity,
                "signal_session": self.signal_session, "reason": self.reason,
                "stop": str(self.stop), "target": str(self.target)}

    @classmethod
    def from_dict(cls, d: dict) -> "PendingOrder":
        return cls(ticker=d["ticker"], side=d["side"], quantity=int(d["quantity"]),
                   signal_session=d.get("signal_session", ""), reason=d.get("reason", ""),
                   stop=D(d.get("stop")), target=D(d.get("target")))


@dataclass
class AccountState:
    profile_code: str
    status: str = "ACTIVE"          # ACTIVE | OBSERVER
    cash: Decimal = field(default_factory=lambda: D(100_000))
    realized_pnl: Decimal = field(default_factory=lambda: D(0))
    positions: dict = field(default_factory=dict)        # ticker -> Position
    pending_orders: list = field(default_factory=list)   # PendingOrder
    ledger: list = field(default_factory=list)           # nakit defteri kayıtları
    equity_history: list = field(default_factory=list)   # [{session, cash, pos_value, equity}]

    def to_dict(self) -> dict:
        return {
            "profile_code": self.profile_code, "status": self.status,
            "cash": str(self.cash), "realized_pnl": str(self.realized_pnl),
            "positions": {t: p.to_dict() for t, p in self.positions.items()},
            "pending_orders": [o.to_dict() for o in self.pending_orders],
            "ledger": self.ledger, "equity_history": self.equity_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AccountState":
        return cls(
            profile_code=d["profile_code"], status=d.get("status", "ACTIVE"),
            cash=D(d.get("cash", 100_000)), realized_pnl=D(d.get("realized_pnl", 0)),
            positions={t: Position.from_dict(p) for t, p in (d.get("positions") or {}).items()},
            pending_orders=[PendingOrder.from_dict(o) for o in (d.get("pending_orders") or [])],
            ledger=list(d.get("ledger") or []), equity_history=list(d.get("equity_history") or []),
        )

    def positions_value(self, prices: dict) -> Decimal:
        """Açık pozisyonların güncel değeri (fiyat yoksa maliyet)."""
        total = D(0)
        for t, p in self.positions.items():
            px = prices.get(t)
            total += D(px if px is not None else p.avg_cost) * p.quantity
        return total

    def equity(self, prices: dict) -> Decimal:
        return self.cash + self.positions_value(prices)


@dataclass
class ArenaState:
    season_id: str = "2026-H2-S1"
    created_at: str = ""
    last_session: str = ""                  # işlenen son seans tarihi (idempotency)
    accounts: dict = field(default_factory=dict)   # profile_code -> AccountState
    predictions: list = field(default_factory=list)  # observer karne kayıtları

    def to_dict(self) -> dict:
        return {"season_id": self.season_id, "created_at": self.created_at,
                "last_session": self.last_session,
                "accounts": {c: a.to_dict() for c, a in self.accounts.items()},
                "predictions": self.predictions}

    @classmethod
    def from_dict(cls, d: dict) -> "ArenaState":
        return cls(
            season_id=d.get("season_id", "2026-H2-S1"), created_at=d.get("created_at", ""),
            last_session=d.get("last_session", ""),
            accounts={c: AccountState.from_dict(a) for c, a in (d.get("accounts") or {}).items()},
            predictions=list(d.get("predictions") or []),
        )


def new_state(season_id: str = "2026-H2-S1", initial_capital: float = 100_000.0) -> ArenaState:
    """Tüm profilleri eşit kasayla seed eden taze sezon durumu."""
    from tradingagents.arena.profiles import PROFILES
    accounts = {}
    for code, prof in PROFILES.items():
        acc = AccountState(profile_code=code, status=prof.status, cash=D(initial_capital))
        acc.ledger.append({
            "ts": datetime.now(timezone.utc).isoformat(), "event_type": "INIT",
            "amount": str(D(initial_capital)), "balance_after": str(D(initial_capital)),
            "ticker": None, "metadata": {"season_id": season_id},
        })
        accounts[code] = acc
    return ArenaState(season_id=season_id,
                      created_at=datetime.now(timezone.utc).isoformat(), accounts=accounts)


def load_state(path: str = _DEFAULT_PATH) -> ArenaState | None:
    """Yerel durumu yükler; yoksa None (çağıran new_state ile başlatır)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return ArenaState.from_dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def save_state(state: ArenaState, path: str = _DEFAULT_PATH) -> None:
    """Durumu yerel JSON'a atomik yazar (önce .tmp, sonra rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

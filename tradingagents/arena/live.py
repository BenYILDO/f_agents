"""Canlı arena seansı — bugünün sinyallerinden emir üreten yapı.

Replay tarihseldir; bu modül **ileriye-dönük** çalışır: ``analyze_universe`` (güven +
rejim + makro şok + kalibre p_up tam yığını) bugünün kararlarını üretir, her hesap
kendi kurallarıyla emir verir, emirler **ertesi seans açılışında** (T+1) dolar,
nakit/pozisyon/equity yerelde birikir.

Sizing canlıda iki yolludur (plan §Profil kuralları):
  - Kaliteli kalibre ``p_up`` varsa → **çeyrek Kelly**.
  - Yoksa → sabit risk bütçesi (stop mesafesine göre).

Seans idempotenttir: aynı seans iki kez işlenmez (plan §cron tekrarı yeni emir açmaz).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

from tradingagents.arena.config import ExecutionConfig
from tradingagents.arena.profiles import PROFILES, Profile
from tradingagents.arena.state import (
    AccountState, ArenaState, PendingOrder, Position, D,
)

_BULLISH = {"AL", "GÜÇLÜ AL"}
_BEARISH = {"SAT", "KAÇIN"}


@dataclass
class SessionReport:
    session: str = ""
    skipped: bool = False
    filled: int = 0
    new_orders: int = 0
    predictions: int = 0
    per_account: dict = None  # code -> {equity, cash, n_positions, n_pending}

    def __post_init__(self):
        if self.per_account is None:
            self.per_account = {}


def _bps(x: float) -> Decimal:
    return D(x) / D(10_000)


def _signal_fields(outcome) -> dict:
    """AnalysisOutcome'dan canlı karar için gerekli alanları çıkarır."""
    sig = getattr(outcome, "signals", {}) or {}
    risk = sig.get("risk") or {}
    conf = sig.get("confidence") or {}
    return {
        "decision": outcome.gated_decision or outcome.decision,
        "confidence": outcome.confidence_score,
        "close": outcome.close,
        "stop": risk.get("stop"),
        "target": risk.get("target"),
        "p_up": conf.get("p_up"),
        "regime": outcome.regime_trend or "",
        "behavior": outcome.behavior or "",
    }


def _size_quantity(profile: Profile, f: dict, equity: Decimal, investable: Decimal) -> int:
    """Pozisyon adedi: kaliteli p_up varsa çeyrek Kelly, yoksa sabit risk bütçesi."""
    close = f.get("close")
    stop = f.get("stop")
    if not close or not stop or stop <= 0 or stop >= close:
        return 0
    close_d, stop_d = D(close), D(stop)
    stop_dist = close_d - stop_d
    equity_f = equity
    p_up = f.get("p_up")
    target = f.get("target")

    if p_up is not None and target and target > close:
        # Çeyrek Kelly: f* = p - (1-p)/b, b = ödül/risk
        b = (D(target) - close_d) / stop_dist
        p = D(p_up)
        f_star = p - (D(1) - p) / b if b > 0 else D(0)
        if f_star <= 0:
            return 0
        frac = min(D(profile.kelly_fraction) * f_star, D(profile.max_position_weight))
        qty_d = (frac * equity_f) / close_d
    else:
        risk_budget = D(profile.risk_per_trade) * equity_f
        qty_risk = risk_budget / stop_dist
        qty_weight = (D(profile.max_position_weight) * equity_f) / close_d
        qty_d = min(qty_risk, qty_weight)

    qty_cash = investable / close_d
    qty = int(min(qty_d, qty_cash).to_integral_value(rounding=ROUND_DOWN))
    return max(0, qty)


def decide_orders(profile: Profile, account: AccountState, outcomes: list,
                  equity: Decimal, cfg: ExecutionConfig, session: str) -> list:
    """Bugünün sinyallerinden bir hesabın ertesi seans emirlerini üretir (BUY+SELL).

    Çıkışlar (held pozisyon): ters sinyal (SAT/KAÇIN), close ≤ stop, ya da max-hold.
    Girişler: gated karar AL + güven eşiği + profil rejim/trend şartları; ranked by güven.
    """
    orders: list[PendingOrder] = []
    held = set(account.positions.keys())
    sig_map = {o.ticker: _signal_fields(o) for o in outcomes}

    # ── Çıkışlar ──────────────────────────────────────────────────────────
    for tk, pos in account.positions.items():
        f = sig_map.get(tk, {})
        decision = f.get("decision", "")
        close = f.get("close")
        reason = ""
        if decision in _BEARISH:
            reason = "ters sinyal"
        elif close is not None and pos.stop > 0 and D(close) <= pos.stop:
            reason = "stop"
        elif pos.opened_session and _days_between(pos.opened_session, session) >= profile.max_hold_days:
            reason = "max-hold"
        if reason:
            orders.append(PendingOrder(ticker=tk, side="SELL", quantity=pos.quantity,
                                       signal_session=session, reason=reason))

    # ── Girişler (ranked by güven) ────────────────────────────────────────
    pending_sells = {o.ticker for o in orders if o.side == "SELL"}
    slots = profile.max_positions - (len(held) - len(pending_sells))
    if slots > 0:
        reserve = equity * D(profile.min_cash_reserve)
        investable = account.cash * (D(1) - _bps(cfg.slippage_bps) * 0) * (D(1) - D(cfg.buying_power_buffer)) - reserve
        investable = max(D(0), investable)

        cands = []
        for o in outcomes:
            tk = o.ticker
            if tk in held:
                continue
            f = sig_map[tk]
            if f["decision"] not in _BULLISH:
                continue
            if f["confidence"] is None or f["confidence"] < profile.min_confidence:
                continue
            if profile.require_regime_bull and f["regime"] != "boğa":
                continue
            if profile.require_trend_behavior and f["behavior"] != "trend":
                continue
            cands.append((f["confidence"], tk, f))
        cands.sort(reverse=True, key=lambda x: x[0])

        for _c, tk, f in cands[:slots]:
            qty = _size_quantity(profile, f, equity, investable)
            if qty <= 0:
                continue
            est_cost = D(f["close"]) * qty * (D(1) + _bps(cfg.slippage_bps) + _bps(cfg.commission_bps))
            if est_cost > investable:
                continue
            investable -= est_cost
            orders.append(PendingOrder(
                ticker=tk, side="BUY", quantity=qty, signal_session=session,
                reason=f"AL güven {f['confidence']:.0f}",
                stop=D(f["stop"]), target=D(f.get("target") or f["close"]),
            ))
    return orders


def apply_fills(account: AccountState, prices: dict, session: str,
                cfg: ExecutionConfig) -> int:
    """Bekleyen emirleri bu seansın açılışında doldurur (T+1). Dolan sayısını döndürür.

    ``prices[ticker] = {"open": .., "close": ..}``. Açılış yoksa emir beklemede kalır.
    Tam-fill-ya-reject: kasa yetmezse alım reddedilir (plan §BIST açılış).
    """
    slip = _bps(cfg.slippage_bps)
    comm = _bps(cfg.commission_bps)
    still_pending: list[PendingOrder] = []
    filled = 0

    for o in account.pending_orders:
        px = (prices.get(o.ticker) or {}).get("open")
        if px is None:
            still_pending.append(o)          # açılış verisi yok → beklemede
            continue
        open_d = D(px)
        if o.side == "BUY":
            fill = open_d * (D(1) + slip)
            gross = fill * o.quantity
            commission = gross * comm
            cost = gross + commission
            if cost > account.cash:
                continue                      # reddedildi (kasa yetmedi)
            account.cash -= cost
            _ledger(account, session, "BUY", -cost, o.ticker, fill, commission)
            _add_position(account, o, fill, session)
            filled += 1
        else:  # SELL
            pos = account.positions.get(o.ticker)
            if pos is None or pos.quantity <= 0:
                continue
            qty = min(o.quantity, pos.quantity)
            fill = open_d * (D(1) - slip)
            gross = fill * qty
            commission = gross * comm
            proceeds = gross - commission
            account.cash += proceeds
            account.realized_pnl += proceeds - pos.avg_cost * qty
            _ledger(account, session, "SELL", proceeds, o.ticker, fill, commission)
            _reduce_position(account, o.ticker, qty)
            filled += 1

    account.pending_orders = still_pending
    return filled


def _add_position(account: AccountState, o: PendingOrder, fill: Decimal, session: str):
    pos = account.positions.get(o.ticker)
    if pos is None:
        account.positions[o.ticker] = Position(
            ticker=o.ticker, quantity=o.quantity, avg_cost=fill,
            stop=o.stop, target=o.target, opened_session=session)
    else:
        new_qty = pos.quantity + o.quantity
        pos.avg_cost = (pos.avg_cost * pos.quantity + fill * o.quantity) / new_qty
        pos.quantity = new_qty
        pos.stop = o.stop or pos.stop
        pos.target = o.target or pos.target


def _reduce_position(account: AccountState, ticker: str, qty: int):
    pos = account.positions.get(ticker)
    if pos is None:
        return
    pos.quantity -= qty
    if pos.quantity <= 0:
        del account.positions[ticker]


def _ledger(account: AccountState, session: str, event: str, amount: Decimal,
            ticker: str, fill: Decimal, commission: Decimal):
    account.ledger.append({
        "ts": datetime.now(timezone.utc).isoformat(), "session": session,
        "event_type": event, "amount": str(amount), "balance_after": str(account.cash),
        "ticker": ticker, "metadata": {"fill_price": str(fill), "commission": str(commission)},
    })


def record_equity(account: AccountState, prices: dict, session: str) -> Decimal:
    """Seans kapanış equity'sini geçmişe yazar (resmî EOD; intraday değil)."""
    close_prices = {t: (prices.get(t) or {}).get("close") for t in account.positions}
    pos_value = account.positions_value(close_prices)
    equity = account.cash + pos_value
    account.equity_history.append({
        "session": session, "cash": str(account.cash),
        "positions_value": str(pos_value), "equity": str(equity),
        "is_official": True,
    })
    return equity


def _days_between(d1: str, d2: str) -> int:
    try:
        a = datetime.fromisoformat(d1[:10]).date()
        b = datetime.fromisoformat(d2[:10]).date()
        return (b - a).days
    except Exception:  # noqa: BLE001
        return 0


def record_predictions(state: ArenaState, outcomes: list, session: str) -> int:
    """Observer karne sözleşmesi (plan F0.6): kaliteli p_up tahminlerini biriktirir.

    Para P&L'i değildir; horizon dolunca gerçekleşen fiyatla değerlendirilir (ileride).
    """
    n = 0
    for o in outcomes:
        f = _signal_fields(o)
        if f["p_up"] is None:
            continue
        state.predictions.append({
            "session": session, "ticker": o.ticker, "p_up": f["p_up"],
            "close_at_signal": f["close"], "decision": f["decision"],
            "horizon_days": 10, "realized_price": None, "realized_up": None,
        })
        n += 1
    return n


def run_session(state: ArenaState, outcomes: list, prices: dict, session: str,
                cfg: ExecutionConfig) -> SessionReport:
    """Bir günlük seansı tüm hesaplar için idempotent işler.

    Akış: (1) bekleyen emirleri bu açılışta doldur, (2) equity yaz, (3) bugünün
    sinyalleriyle ertesi seans emirlerini üret, (4) observer tahminlerini kaydet.
    ``outcomes`` tüm hesaplarca paylaşılan tek dondurulmuş analiz (plan §adil lig).
    """
    if state.last_session and session <= state.last_session:
        return SessionReport(session=session, skipped=True)

    report = SessionReport(session=session)
    for code, account in state.accounts.items():
        prof = PROFILES.get(code)
        if prof is None or account.status == "OBSERVER":
            # Observer para işlemez; yalnız equity'si sabit kasada kalır
            if account.status == "OBSERVER":
                record_equity(account, prices, session)
            continue
        report.filled += apply_fills(account, prices, session, cfg)
        equity = record_equity(account, prices, session)
        new_orders = decide_orders(prof, account, outcomes, equity, cfg, session)
        account.pending_orders = new_orders
        report.new_orders += len(new_orders)
        report.per_account[code] = {
            "equity": float(equity), "cash": float(account.cash),
            "n_positions": len(account.positions), "n_pending": len(new_orders),
        }

    report.predictions = record_predictions(state, outcomes, session)
    state.last_session = session
    return report


def build_session_inputs(tickers: list[str] | None = None):
    """Bugünün gerçek sinyallerini + fill fiyatlarını üretir (analyze_universe).

    Ağ gerektirir (yfinance). ``(outcomes, prices, session)`` döndürür:
      - outcomes: tüm hesaplarca paylaşılan tek dondurulmuş analiz (güven/rejim/
        makro şok/p_up tam yığını) — adil lig için bir kez hesaplanır.
      - prices: {ticker: {"open","close"}} son seans barından (T+1 fill + EOD MTM).
      - session: son bar tarihi (ISO) — seans kimliği/idempotency.
    """
    import pandas as pd
    from tradingagents.analysis.run import analyze_universe
    from tradingagents.analytics.composite import _fetch_daily
    from tradingagents.strategy.dip_signal import BIST30

    universe = tickers or list(BIST30)
    outcomes = analyze_universe(universe)

    prices: dict[str, dict] = {}
    session = ""
    for o in outcomes:
        try:
            df = _fetch_daily(o.ticker, period="1mo")
        except Exception:  # noqa: BLE001
            df = None
        if df is None or len(df) == 0:
            continue
        last = df.iloc[-1]
        prices[o.ticker] = {"open": float(last["Open"]), "close": float(last["Close"])}
        d = str(pd.Timestamp(df.index[-1]).date())
        session = max(session, d) if session else d

    if not session:
        session = datetime.now(timezone.utc).date().isoformat()
    return outcomes, prices, session

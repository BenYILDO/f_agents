"""Güncel fiyat yardımcıları — portföy P&L'i için son kapanışlar.

yfinance'ten son kapanış fiyatlarını çeker. Asla istisna fırlatmaz; ulaşılamayan
semboller sonuç sözlüğünde yer almaz (UI 'fiyat yok' gösterir). Portföy küçük
olduğundan (kullanıcının elindeki birkaç hisse) sembol başına thread'li çekim
hem basit hem dayanıklıdır.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import yfinance as yf


def _last_close(ticker: str) -> tuple[str, float | None]:
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist is None or hist.empty:
            return ticker, None
        closes = hist["Close"].dropna()
        if closes.empty:
            return ticker, None
        return ticker, float(closes.iloc[-1])
    except Exception:  # noqa: BLE001 — fiyat alınamadıysa sessizce atla
        return ticker, None


def latest_prices(tickers: list[str]) -> dict[str, float]:
    """Verilen sembollerin son kapanış fiyatlarını döndürür (ulaşılamayanlar yok)."""
    uniq = [t.strip().upper() for t in dict.fromkeys(tickers) if t and t.strip()]
    if not uniq:
        return {}
    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(uniq))) as ex:
        for ticker, price in ex.map(_last_close, uniq):
            if price is not None:
                out[ticker] = price
    return out

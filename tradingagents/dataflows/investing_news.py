"""Ticker-specific Turkish news for BIST (.IS) tickers via Investing.com.

The RSS feeds in :mod:`tradingagents.dataflows.turkish_news` are macro-heavy and
rarely carry company-specific items. Investing.com's Turkish desk, by contrast,
republishes BIST companies' KAP material disclosures (kâr dağıtımı, yönetim
değişikliği, pay alım/satım, yatırım onayları) as digestible Turkish news — so
this fetcher doubles as a practical KAP-disclosure signal without scraping the
WAF-gated KAP site itself.

Two best-effort steps, each degrading gracefully:

1. **Resolve** the ticker to its Investing equity slug via the public search API
   (``api.investing.com/api/search/v2/search``), confirming the Istanbul listing
   so we never attach a same-symbol foreign company's news. The endpoint blocks
   plain HTTP clients, so we call it through ``curl_cffi`` with Chrome TLS
   impersonation (already a yfinance dependency).

2. **Render** the equity's Turkish ``/news`` page with Playwright and read the
   headlines from the DOM (the article list is client-hydrated and absent from
   the server HTML). Playwright + chromium are optional runtime deps; if they are
   missing or the render fails/times out, the fetcher reports that and yields a
   placeholder rather than raising.

The Playwright call runs in a dedicated worker thread with a hard timeout: the
sync Playwright API refuses to start inside a running asyncio loop (LangGraph may
provide one), and a fresh thread has no such loop. The same wrapper bounds the
total time so a slow render can never stall the trading graph.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional

logger = logging.getLogger(__name__)

_SEARCH_API = "https://api.investing.com/api/search/v2/search?q={q}"
_BASE = "https://tr.investing.com"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _bist_base(ticker: str) -> str:
    """Strip the Yahoo ``.IS`` suffix to recover the bare BIST code (THYAO)."""
    return ticker.strip().upper().removesuffix(".IS")


def _resolve_equity_url(ticker: str, timeout: float) -> Optional[str]:
    """Resolve a BIST ticker to its Investing.com equity path, or None.

    Picks the search ``quotes`` entry whose symbol matches the BIST code and
    whose exchange is Istanbul, so e.g. a US-listed same-symbol name is never
    chosen. Returns a path like ``/equities/turk-hava-yollari``.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        logger.warning("curl_cffi not available; cannot resolve Investing equity for %s", ticker)
        return None

    base = _bist_base(ticker)
    # The search endpoint occasionally throttles a request; one quick retry
    # turns a transient miss into a hit without materially slowing the node.
    quotes: list = []
    for attempt in range(2):
        try:
            resp = cffi_requests.get(
                _SEARCH_API.format(q=base), impersonate="chrome", timeout=timeout
            )
            if resp.status_code == 200:
                quotes = resp.json().get("quotes", []) or []
                if quotes:
                    break
        except Exception as exc:  # noqa: BLE001 — network/parse, fail open
            logger.warning("Investing search failed for %s (attempt %d): %s", ticker, attempt + 1, exc)
        if attempt == 0:
            time.sleep(0.8)
    if not quotes:
        return None

    # Prefer an exact Istanbul-listed symbol match; fall back to the first
    # Turkey-flagged quote, then the first quote with an equity URL.
    def _is_equity(q: dict) -> bool:
        return isinstance(q, dict) and isinstance(q.get("url"), str) and "/equities/" in q["url"]

    for q in quotes:
        if _is_equity(q) and str(q.get("symbol", "")).upper() == base and \
                str(q.get("exchange", "")).lower() == "istanbul":
            return q["url"]
    for q in quotes:
        if _is_equity(q) and str(q.get("flag", "")).lower() == "turkey":
            return q["url"]
    for q in quotes:
        if _is_equity(q):
            return q["url"]
    return None


def _scrape_news_titles(news_url: str, limit: int, nav_timeout_ms: int) -> list[dict]:
    """Render an Investing equity /news page and return [{title, url}] from the DOM.

    Runs synchronously; callers invoke it through :func:`_run_in_thread` so it
    never touches a running asyncio loop. Returns [] if Playwright is missing or
    anything goes wrong.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("playwright not installed; skipping Investing /news render")
        return []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(user_agent=_UA, locale="tr-TR")
                page = ctx.new_page()
                page.goto(news_url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
                page.wait_for_timeout(3500)  # let the article list hydrate
                rows = page.eval_on_selector_all(
                    "article a[href*='/news/'], a[data-test='article-title-link']",
                    "els => els.map(e => ({title: (e.textContent||'').trim(), "
                    "url: e.getAttribute('href')})).filter(r => r.title.length > 12)",
                )
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 — render/timeout, fail open
        logger.warning("Investing /news render failed for %s: %s", news_url, exc)
        return []

    seen, out = set(), []
    for r in rows:
        key = r["title"].casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def _run_in_thread(fn, *args, timeout: float):
    """Run a blocking callable in a worker thread with a hard wall-clock timeout.

    Isolates sync Playwright from any running asyncio loop and guarantees the
    trading graph is never stalled by a slow page render.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            logger.warning("Investing /news scrape exceeded %.0fs; giving up", timeout)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Investing /news worker errored: %s", exc)
            return []


def fetch_investing_news(
    ticker: str,
    limit: int = 8,
    resolve_timeout: float = 15.0,
    render_timeout: float = 30.0,
) -> str:
    """Fetch ticker-specific Turkish news for a BIST ``ticker`` as a prompt block.

    Returns a formatted plaintext block, or a clear placeholder string when the
    equity can't be resolved or the page can't be rendered — never raises.
    """
    equity_url = _resolve_equity_url(ticker, resolve_timeout)
    if not equity_url:
        return (
            f"<{_bist_base(ticker)} için Investing.com'da eşleşen BIST hissesi "
            f"bulunamadı; ticker-özel Türkçe haber alınamadı>"
        )

    news_url = f"{_BASE}{equity_url}-news"
    # Total budget slightly above the nav timeout so the thread wrapper is the
    # outer bound; nav_timeout is in milliseconds.
    nav_timeout_ms = int(render_timeout * 1000)
    items = _run_in_thread(
        _scrape_news_titles, news_url, limit, nav_timeout_ms,
        timeout=render_timeout + 10.0,
    )

    if not items:
        return (
            f"<{_bist_base(ticker)} için Investing.com Türkçe haber sayfası "
            f"render edilemedi (Playwright yok ya da zaman aşımı)>"
        )

    base = _bist_base(ticker)
    lines = [f"{base} — Investing.com Türkçe haber başlıkları (KAP açıklamaları dahil):"]
    for it in items:
        url = it.get("url") or ""
        full = url if url.startswith("http") else f"{_BASE}{url}" if url else ""
        lines.append(f"  - {it['title']}" + (f"\n    {full}" if full else ""))
    return "\n".join(lines)

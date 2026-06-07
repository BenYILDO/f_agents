"""Turkish-language market news fetcher for Borsa İstanbul (BIST) tickers.

yfinance's news coverage for ``.IS`` tickers is mostly English-language and
international-desk framing, which misses the local-language retail/market
narrative that actually moves Turkish equities. This fetcher fills that gap
by pulling Turkish-language finance RSS feeds and surfacing the items that
mention the target company alongside recent market headlines for context.

Sources (public RSS, no API key, no registration):
  1. Investing.com Türkiye — borsa (stock-market) feed: BIST-focused, often
     references companies by their BIST code (e.g. "GESAN", "THYAO").
  2. Investing.com Türkiye — general finance feed.
  3. BloombergHT — broader macro / market headlines.

Like :mod:`tradingagents.dataflows.reddit`, this degrades gracefully: every
network/parse failure is swallowed per-feed, and the function always returns a
formatted plaintext string (real data or a clear placeholder) so the calling
agent never has to special-case missing data. KAP (the official disclosure
platform) is intentionally not scraped here — its rebuilt site is WAF-gated and
blocks keyless programmatic clients; disclosure-driven items still leak in via
the finance feeds above.
"""

from __future__ import annotations

import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# A descriptive, identified User-Agent. The feeds below serve this token; we
# do not spoof a browser because none of these endpoints require it.
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# (label, url) ordered by signal density for BIST-specific discussion. The
# borsa feed is the most on-topic; BloombergHT is broad macro context.
DEFAULT_FEEDS: tuple[tuple[str, str], ...] = (
    ("Investing.com TR — Borsa", "https://tr.investing.com/rss/stock.rss"),
    ("Investing.com TR — Finans", "https://tr.investing.com/rss/news.rss"),
    ("BloombergHT", "https://www.bloomberght.com/rss"),
)

# Corporate-form tokens dropped when deriving company keywords from a long
# name — they are too generic to signal a company-specific article.
_GENERIC_NAME_TOKENS = frozenset({
    "anonim", "ortakligi", "ortaklığı", "sirketi", "şirketi", "holding",
    "holdingi", "sanayi", "ticaret", "san", "tic", "ltd", "as", "a.s",
    "a.ş", "group", "grup", "yatirim", "yatırım", "gmyo", "gyo",
})


def _strip_html(text: str) -> str:
    """Reduce an RSS description (may contain HTML) to clean plain text."""
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(no_tags).split())


def _build_keywords(ticker: str, company_name: Optional[str]) -> list[str]:
    """Derive case-folded match keywords from the ticker base and company name.

    The BIST code (e.g. ``THYAO`` from ``THYAO.IS``) is the strongest signal —
    Turkish borsa feeds routinely headline companies by their code. Company-name
    tokens add recall for prose that spells the name out; generic corporate-form
    words are dropped to avoid matching unrelated firms.
    """
    keywords: list[str] = []
    base = ticker.strip().upper()
    if base.endswith(".IS"):
        base = base[:-3]
    if base:
        keywords.append(base.casefold())

    if company_name:
        for raw in re.split(r"[\s/.,]+", company_name):
            token = raw.strip()
            if len(token) >= 5 and token.casefold() not in _GENERIC_NAME_TOKENS:
                keywords.append(token.casefold())

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = []
    for kw in keywords:
        if kw and kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def _fetch_feed(label: str, url: str, limit: int, timeout: float) -> list[dict]:
    """Fetch and parse one RSS 2.0 feed. Returns [] on any failure."""
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/xml, text/xml, */*"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except (HTTPError, URLError, TimeoutError, ET.ParseError) as exc:
        logger.warning("Turkish news fetch failed for %s (%s): %s", label, url, exc)
        return []

    items = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "summary": _strip_html(item.findtext("description") or ""),
            "pub_date": (item.findtext("pubDate") or "").strip(),
            "source": label,
        })
    return items


def fetch_turkish_market_news(
    ticker: str,
    company_name: Optional[str] = None,
    feeds: Iterable[tuple[str, str]] = DEFAULT_FEEDS,
    limit_per_feed: int = 8,
    max_company_items: int = 8,
    max_market_items: int = 6,
    timeout: float = 10.0,
    inter_request_delay: float = 0.3,
) -> str:
    """Fetch Turkish-language market news for a BIST ``ticker`` as a prompt block.

    Items mentioning the company (by BIST code or name) are surfaced first as
    company-specific signal; the remaining recent headlines are included as
    market/macro context. Degrades gracefully — returns a placeholder string
    rather than raising when every feed is unreachable.
    """
    keywords = _build_keywords(ticker, company_name)

    collected: list[dict] = []
    seen_titles: set[str] = set()
    for i, (label, url) in enumerate(feeds):
        if i > 0:
            time.sleep(inter_request_delay)
        for item in _fetch_feed(label, url, limit_per_feed, timeout):
            key = item["title"].casefold()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            collected.append(item)

    if not collected:
        return (
            f"<Türkçe haber kaynaklarına şu an ulaşılamadı "
            f"({', '.join(label for label, _ in feeds)})>"
        )

    company_items, market_items = [], []
    for item in collected:
        haystack = f"{item['title']} {item['summary']}".casefold()
        if keywords and any(kw in haystack for kw in keywords):
            company_items.append(item)
        else:
            market_items.append(item)

    def _fmt(item: dict) -> str:
        line = f"  [{item['source']}"
        if item["pub_date"]:
            line += f" · {item['pub_date']}"
        line += f"] {item['title']}"
        if item["summary"]:
            excerpt = item["summary"]
            if len(excerpt) > 280:
                excerpt = excerpt[:280] + "…"
            line += f"\n    {excerpt}"
        return line

    blocks: list[str] = []
    base = ticker.strip().upper().removesuffix(".IS")
    if company_items:
        blocks.append(
            f"Şirkete özel haberler ({base}):\n"
            + "\n".join(_fmt(it) for it in company_items[:max_company_items])
        )
    else:
        blocks.append(
            f"<{base} için doğrudan şirket haberi bulunamadı; aşağıdaki genel "
            f"piyasa başlıkları bağlam olarak verilmiştir>"
        )

    if market_items:
        blocks.append(
            "Genel piyasa / makro başlıkları:\n"
            + "\n".join(_fmt(it) for it in market_items[:max_market_items])
        )

    return "\n\n".join(blocks)

"""Macro analyst (Türk usulü) — domestic macro-regime read for BIST tickers.

Why a separate analyst? The News analyst leans on global/English sources via
``get_global_news`` and the Sentiment analyst folds TCMB/rate/FX headlines in
only as company-level *backdrop* (see ``sentiment_analyst._build_bist_system_message``).
That buries the single biggest driver of Borsa İstanbul: the domestic macro
regime. A TCMB rate cut, an inflation surprise, or a lira move reprices the
*whole* index and tilts *sectors* (rate-sensitive banks/holdings, FX-exposed
importers vs. exporters) before it touches any single name. This analyst makes
that read explicit and hands it to the researchers and risk team as its own
``macro_report``.

Design mirrors the Sentiment analyst: it pre-fetches Turkish-language macro
headlines (no tool-call loop, no API key — deterministic and cheap) and injects
them into the prompt. BIST-only: for non-.IS tickers it emits a short note
instead of a report, since global macro is already covered by the News analyst.
"""

from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.dataflows.symbol_utils import is_bist_ticker
from tradingagents.dataflows.turkish_macro import fetch_turkish_macro_news
from tradingagents.dataflows.tcmb_calendar import resolve_ppk, get_macro_official_numbers
from tradingagents.dataflows.macro_event_study import (
    compute_event_study,
    format_event_study,
)
from tradingagents.dataflows.gold_fx import fetch_gold_fx_snapshot
from tradingagents.dataflows.data_health import (
    SourceHealth, OK, EMPTY, ERROR, any_ok, render_health,
)

# Rate-direction cues used to (a) condition the historical event study on the
# same kind of move and (b) label it. Cut/hike are checked before the generic
# rate theme so "faiz indirimi" routes to cut, not a direction-less decision.
_CUT_CUES = ("faiz indir", "indirim", "faizi düşür", "gevşeme", "faiz düş")
_HIKE_CUES = ("faiz artır", "artırım", "faizi yükselt", "sıkılaş", "faiz yüksel")
_RATE_CUES = ("faiz", "ppk", "tcmb", "politika faizi", "merkez banka")


def _detect_rate_direction(macro_text: str) -> tuple[str | None, str | None]:
    """Infer the current PPK move from macro headlines.

    Returns ``(direction, label)`` where direction is ``"cut"``/``"hike"``/None
    and label is the Turkish event name, or ``(None, None)`` when the headlines
    carry no rate signal (so the event study is skipped rather than forced).
    """
    t = macro_text.casefold()
    has_cut = any(c in t for c in _CUT_CUES)
    has_hike = any(c in t for c in _HIKE_CUES)
    if has_cut and not has_hike:
        return "cut", "TCMB faiz indirimi"
    if has_hike and not has_cut:
        return "hike", "TCMB faiz artırımı"
    if any(c in t for c in _RATE_CUES):
        return None, "TCMB faiz kararı"  # rate theme present, direction unclear
    return None, None


def _build_event_study(
    ticker: str, macro_text: str,
) -> tuple[str, SourceHealth, Optional[SourceHealth]]:
    """Return (event-study text, price-data health, PPK-source health).

    When no rate event is detected the study is skipped (both healths report
    ``empty`` with a reason rather than masquerading as data).
    """
    direction, label = _detect_rate_direction(macro_text)
    if label is None:
        return (
            "<Güncel başlıklarda olay-etüdünü tetikleyecek bir TCMB faiz "
            "kararı/sinyali yok; geçmiş benzer-olay analizi atlandı.>",
            SourceHealth("Olay-etüdü (fiyat geçmişi)", EMPTY, "tetikleyici faiz sinyali yok"),
            None,
        )
    decisions, ppk_health = resolve_ppk(direction=direction)
    result = compute_event_study(
        ticker, [d["date"] for d in decisions], direction_label=label,
    )
    if result.ok:
        ev_health = SourceHealth("Olay-etüdü (fiyat geçmişi)", OK,
                                 f"{result.n_events} olay", result.n_events)
    else:
        ev_health = SourceHealth("Olay-etüdü (fiyat geçmişi)", ERROR, result.reason)
    return format_event_study(result), ev_health, ppk_health

# Emitted for non-BIST instruments so the node is a graceful no-op there
# (global macro is the News analyst's job via get_global_news).
_NON_BIST_NOTE = (
    "Makro Analist yalnızca Borsa İstanbul (.IS) hisseleri için Türkiye makro "
    "görünümü üretir. Bu enstrüman BIST dışı olduğundan küresel makro "
    "değerlendirmesi Haber Analisti'ne (get_global_news) bırakılmıştır."
)


def create_macro_analyst(llm):
    """Create the Türkiye macro-analyst node for the trading graph.

    Pre-fetches theme-bucketed Turkish macro headlines and produces a
    BIST-wide / sector-impact macro report. Does not use tool-calling; the
    data is in the prompt from turn 0, so the node returns in a single pass.
    """

    def macro_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        if not is_bist_ticker(ticker):
            return {"messages": [AIMessage(content=_NON_BIST_NOTE)],
                    "macro_report": _NON_BIST_NOTE, "macro_data_health": ""}

        # Pre-fetch every source with health. Official EVDS numbers are the
        # backbone; news adds color; the event study is the ticker base rate.
        news = fetch_turkish_macro_news()
        official_text, official_health = get_macro_official_numbers()
        event_text, event_health, ppk_health = _build_event_study(ticker, news.text)
        gold_fx = fetch_gold_fx_snapshot()

        health = [*news.sources, *official_health, event_health, *gold_fx.sources]
        if ppk_health is not None:
            health.append(ppk_health)
        health_text = render_health(health)

        # Fail loud: if neither the news feeds NOR the official EVDS numbers
        # returned real data, the macro read rests on nothing — flag it at the
        # top of the report (per the desk's "warn and continue" policy) instead
        # of letting a placeholder pass as analysis.
        core_ok = news.ok or any_ok(official_health)
        warning = "" if core_ok else (
            "> ⚠️ **VERİ UYARISI:** Makro veri kaynaklarına (Türkçe haber RSS + "
            "TCMB EVDS resmi serileri) ulaşılamadı. Aşağıdaki değerlendirme "
            "sınırlı/eksik veriyle üretilmiştir — düşük güvenle ele alın.\n\n"
        )

        system_message = _build_system_message(
            current_date, official_text, news.text, event_text, health_text,
            gold_fx.text,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant collaborating with other analysts"
                    " on a Borsa İstanbul (BIST) trading desk."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # No bind_tools — the macro data is already in the prompt, so a single
        # invoke yields the report (the conditional edge then routes to clear).
        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages)
        body = result.content if isinstance(result.content, str) else str(result.content)
        report = warning + body

        return {"messages": [AIMessage(content=report)], "macro_report": report,
                "macro_data_health": health_text}

    return macro_analyst_node


def _build_system_message(
    current_date: str, official_block: str, macro_block: str,
    event_block: str, health_block: str, gold_fx_block: str = "",
) -> str:
    """Assemble the macro-analyst system message from all pre-fetched blocks."""
    return f"""You are the Türkiye Macro Analyst on a Borsa İstanbul (BIST) trading desk. Your job is NOT to analyze one company — it is to read the domestic macro regime as of {current_date} and translate it into a *market-wide and sector* signal that the rest of the desk weighs alongside the company-specific reports.

## Official TCMB data (EVDS) — the backbone, use first

Authoritative, numeric. When present, anchor your read on these hard numbers (latest policy rate, USD/TRY, CPI) rather than on news framing.

<start_of_official_data>
{official_block}
<end_of_official_data>

## Macro headlines (pre-fetched, Turkish-language, theme-bucketed)

These add color and timeliness on top of the official numbers, grouped by theme (faiz/para politikası, enflasyon, kur, büyüme, bütçe/ülke riski, jeopolitik).

<start_of_macro_news>
{macro_block}
<end_of_macro_news>

## Gold & FX snapshot (deterministic, computed) — the Turkish saver's benchmark

Gram altın ve dolar, Türk yatırımcının BIST'e karşı fiili alternatifleridir. Use this computed block to judge the *relative* attractiveness of equities: when gold/FX have sharply outperformed BIST in TL terms, domestic flows tend to rotate away from equities (and vice versa). The XU100/gram-gold ratio percentile tells you whether BIST is historically cheap or rich in real (gold) terms.

<start_of_gold_fx>
{gold_fx_block}
<end_of_gold_fx>

## Data health — be honest about what was actually available

This is the live status of each source. Any line that is not "✓" means that input is missing/degraded; lower your confidence accordingly and say so explicitly. NEVER present a regime read as solid when the sources behind it were empty.

<start_of_data_health>
{health_block}
<end_of_data_health>

## Historical event study — how THIS ticker reacted to similar past events

These are *computed* statistics (deterministic, from real price history), not estimates: the target ticker's forward returns after past TCMB decisions of the same kind as the current signal. Treat them as a base rate for the likely reaction and **relay them explicitly to the rest of the desk** — they are the answer to "when news like this happened before, how did this stock behave?".

<start_of_event_study>
{event_block}
<end_of_event_study>

## How to analyze (best practices)

1. **Lead with monetary policy.** The TCMB policy rate / PPK decision is the dominant BIST driver. A rate cut (faiz indirimi) is broadly risk-on for equities and especially helps rate-sensitive, highly-leveraged and long-duration names (banks' funding costs, holdings, real estate/GYO, high-capex industrials); a hike or hawkish hold is the opposite. State the current policy stance and its direction explicitly.

2. **Read inflation in context.** Falling inflation (dezenflasyon) supports the rate-cut path and real returns; an upside surprise threatens it. Connect the inflation read to the likely rate path, not just the headline number.

3. **Map the lira.** A weak/volatile lira hurts FX-indebted and import-dependent names but helps exporters (ihracatçılar) and FX-revenue names (e.g. aviation, refiners, exporters). Note the FX direction and which side of the market it favors.

4. **Country risk & flows.** CDS / risk premium, rating actions (Moody's/Fitch/S&P), and foreign-investor flows set the overall risk appetite (risk-on / risk-off) for the whole index.

5. **Translate to a regime + sector tilt.** Conclude with: (a) an overall regime read — **Risk-On / Nötr / Risk-Off** for BIST — and (b) which sectors the current macro favors vs. pressures. Be explicit that this is a top-down backdrop, not a company call.

6. **Ground the call in the event study.** When the event-study block has numbers, state the historical base rate plainly (e.g. "geçmiş N faiz indiriminden sonra bu hisse +5 günde medyan %X, pozitif oran %Y") and let it temper or reinforce the top-down read. If it is a placeholder (no matching event or no price data), say so — do not invent history.

7. **Be honest about data limits.** If the macro block is a "<...ulaşılamadı>" / "<...tespit edilemedi>" placeholder, say so and lower your confidence rather than inventing a regime.

## Output format

Write a concise macro report with:
- A one-line **headline regime**: Risk-On / Nötr / Risk-Off for BIST, with the single most important driver.
- Short paragraphs per relevant theme (rate path, inflation, lira, country risk) — only those the headlines actually support.
- A **historical base rate** line that relays the event-study numbers for this ticker (when available), so the trader and risk team see how the stock behaved after similar past events.
- A **sector tilt** note: macro tailwinds vs. headwinds by sector.
- A closing Markdown table summarizing: Theme | Signal (yön) | BIST/Sector impact.

{get_language_instruction()}"""

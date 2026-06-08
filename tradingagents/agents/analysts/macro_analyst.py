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

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.dataflows.symbol_utils import is_bist_ticker
from tradingagents.dataflows.turkish_macro import fetch_turkish_macro_news
from tradingagents.dataflows.tcmb_calendar import get_ppk_decisions
from tradingagents.dataflows.macro_event_study import (
    compute_event_study,
    format_event_study,
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


def _build_event_study_block(ticker: str, macro_block: str) -> str:
    """Build the 'how did this ticker behave after similar past events' block."""
    direction, label = _detect_rate_direction(macro_block)
    if label is None:
        return ("<Güncel başlıklarda olay-etüdünü tetikleyecek bir TCMB faiz "
                "kararı/sinyali yok; geçmiş benzer-olay analizi atlandı.>")
    decisions = get_ppk_decisions(direction=direction)
    result = compute_event_study(
        ticker, [d["date"] for d in decisions], direction_label=label,
    )
    return format_event_study(result)

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
                    "macro_report": _NON_BIST_NOTE}

        macro_block = fetch_turkish_macro_news()
        event_block = _build_event_study_block(ticker, macro_block)
        system_message = _build_system_message(current_date, macro_block, event_block)

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
        report = result.content if isinstance(result.content, str) else str(result.content)

        return {"messages": [AIMessage(content=report)], "macro_report": report}

    return macro_analyst_node


def _build_system_message(current_date: str, macro_block: str, event_block: str) -> str:
    """Assemble the macro-analyst system message with macro + event-study blocks."""
    return f"""You are the Türkiye Macro Analyst on a Borsa İstanbul (BIST) trading desk. Your job is NOT to analyze one company — it is to read the domestic macro regime as of {current_date} and translate it into a *market-wide and sector* signal that the rest of the desk weighs alongside the company-specific reports.

## Macro headlines (pre-fetched, Turkish-language, theme-bucketed)

These are the freshest Turkish macro headlines, grouped by theme (faiz/para politikası, enflasyon, kur, büyüme, bütçe/ülke riski, jeopolitik).

<start_of_macro_news>
{macro_block}
<end_of_macro_news>

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

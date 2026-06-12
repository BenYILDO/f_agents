"""Geopolitics analyst (Türk usulü) — iç siyaset + jeopolitik risk okuması.

Why a separate analyst? Türkiye piyasası siyasete dünya ortalamasının çok
üzerinde duyarlıdır: bir gece kararnamesiyle TCMB başkanı değişir ve endeks
ertesi gün taban olur (Mart 2021); bir tutuklama haberi tek günde %8 düşürür
(Mart 2025); bir seçim sonucu rejim değiştirir (Haziran 2023 → Şimşek
rallisi). Macro analyst bu olayları ancak FAİZ/KUR'a yansıdıktan SONRA görür
— bu analist ise siyasi şoku kaynağında okur ve fiyat etkisini şok GELMEDEN
tartar. Sentiment/News analistlerinden farkı: şirket haberi değil, ülke
çapında siyasi risk primi analiz eder.

Design mirrors the Macro analyst: pre-fetched data (siyasi başlık filtresi +
küratörlü şok takvimi + bu hissenin geçmiş siyasi şoklardaki davranışının
olay etüdü), no tool-call loop, BIST-only. The event study answers the core
question deterministically: "geçmiş siyasi şoklarda BU hisse ne yaptı?" —
the LLM interprets, never invents the numbers.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.dataflows.data_health import (
    SourceHealth, OK, ERROR, render_health,
)
from tradingagents.dataflows.macro_event_study import (
    compute_event_study,
    format_event_study,
)
from tradingagents.dataflows.political_events import (
    fetch_political_news,
    format_shock_calendar,
    political_event_dates,
)
from tradingagents.dataflows.symbol_utils import is_bist_ticker

# BIST dışı enstrümanlarda zarif no-op (küresel jeopolitik zaten News
# analistinin get_global_news kapsamında).
_NON_BIST_NOTE = (
    "Jeopolitik Analist yalnızca Borsa İstanbul (.IS) hisseleri için Türkiye "
    "siyasi/jeopolitik risk okuması üretir. Bu enstrüman BIST dışı olduğundan "
    "küresel jeopolitik değerlendirmesi Haber Analisti'ne bırakılmıştır."
)


def _build_political_event_study(ticker: str) -> tuple[str, SourceHealth]:
    """Hissenin geçmiş siyasi şoklardaki forward getirilerini hesaplar."""
    dates = political_event_dates()
    result = compute_event_study(
        ticker, dates, direction_label="Türkiye siyasi/jeopolitik şok",
    )
    if result.ok:
        health = SourceHealth("Siyasi olay-etüdü (fiyat geçmişi)", OK,
                              f"{result.n_events} olay", result.n_events)
    else:
        health = SourceHealth("Siyasi olay-etüdü (fiyat geçmişi)", ERROR, result.reason)
    return format_event_study(result), health


def create_geopolitics_analyst(llm):
    """Create the Türkiye geopolitics-analyst node for the trading graph.

    Pre-fetches filtered political headlines, the curated shock calendar and
    the ticker's political-shock event study; single LLM pass, no tools.
    """

    def geopolitics_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        if not is_bist_ticker(ticker):
            return {"messages": [AIMessage(content=_NON_BIST_NOTE)],
                    "geopolitics_report": _NON_BIST_NOTE}

        news = fetch_political_news()
        event_text, event_health = _build_political_event_study(ticker)
        health_text = render_health([*news.sources, event_health])

        # Fail loud (masa politikası: uyar ve devam et): başlıklar da olay
        # etüdü de boşsa rapor spekülasyona döner — bunu en üstte söyle.
        core_ok = news.ok or event_health.status == OK
        warning = "" if core_ok else (
            "> ⚠️ **VERİ UYARISI:** Siyasi haber akışına ve olay-etüdü fiyat "
            "verisine ulaşılamadı. Aşağıdaki değerlendirme sınırlı/eksik "
            "veriyle üretilmiştir — düşük güvenle ele alın.\n\n"
        )

        system_message = _build_system_message(
            current_date, news.text, format_shock_calendar(), event_text, health_text,
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

        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages)
        body = result.content if isinstance(result.content, str) else str(result.content)
        report = warning + body

        return {"messages": [AIMessage(content=report)],
                "geopolitics_report": report}

    return geopolitics_analyst_node


def _build_system_message(
    current_date: str, news_block: str, calendar_block: str,
    event_block: str, health_block: str,
) -> str:
    """Assemble the geopolitics-analyst system message from pre-fetched blocks."""
    return f"""You are the Türkiye Geopolitics & Domestic Politics Analyst on a Borsa İstanbul (BIST) trading desk. Your job is to read the CURRENT political risk picture as of {current_date} — domestic politics, foreign policy, geopolitical tensions — and translate it into a market risk signal. You are NOT a political commentator: every political observation must end in a market implication (risk primi, kur kanalı, yabancı akımı, sektör etkisi) or be dropped.

## Why this desk needs you (internalize this)

Borsa İstanbul is exceptionally politics-sensitive. The historical record is unambiguous:
- A midnight decree replacing the central bank governor (Mar 2021) gapped the index limit-down and crashed the lira within hours.
- A single politically-charged detention (Mar 2025) dropped BIST ~8% in a day and forced a trading halt.
- An election outcome that restored orthodox policy (Jun 2023, Şimşek appointment) started a multi-month re-rating rally.
- Geopolitical flashpoints (2015 Russian jet, 2018 Brunson crisis, 2019 Syria operations) hit through the lira and CDS first, equities second.
Your value is reading these BEFORE they fully price in, and refusing to manufacture a crisis when the political backdrop is genuinely calm.

## Current political headlines (pre-fetched, Turkish-language, filtered)

<start_of_political_news>
{news_block}
<end_of_political_news>

## Curated political shock calendar — the desk's institutional memory

These are the dated political/geopolitical shocks that actually moved this market since 2013, with the first market reaction. Use them as your reference library for "what kind of event is this most similar to?".

<start_of_shock_calendar>
{calendar_block}
<end_of_shock_calendar>

## Historical event study — how THIS ticker behaved after past political shocks

Computed statistics (deterministic, real price history), not estimates: the target ticker's forward returns after the shock dates above. This is the base rate for "when politics hits, how does this stock react?" — relay it explicitly to the desk.

<start_of_event_study>
{event_block}
<end_of_event_study>

## Data health — be honest about what was actually available

<start_of_data_health>
{health_block}
<end_of_data_health>

## How to analyze (best practices)

1. **Classify the current picture.** Sakin / Gerilim birikiyor / Akut şok. Most days are "sakin" — say so plainly and assign LOW political risk; do not invent tension to sound useful. An analyst who cries wolf daily is worthless to the desk.

2. **Match against the shock calendar.** If something is brewing (seçim takvimi, yargı süreci, sınır ötesi gerilim, kredi notu kararı), find the closest historical analog(s) in the calendar and anchor your impact estimate on what actually happened then — direction, magnitude, duration.

3. **Trace the transmission channel.** Political risk hits BIST through specific channels — name the active one(s): (a) kur/CDS (risk primi), (b) yabancı yatırımcı akımı, (c) para politikasının bağımsızlığı algısı, (d) sektörel düzenleme riski (bankalar, enerji, savunma, müteahhitlik), (e) ihracat pazarları / ticaret koridorları.

4. **Separate company exposure from index exposure.** Savunma hisseleri (ASELS) jeopolitik gerilimden POZİTİF etkilenebilir; turizm/havacılık (THYAO, PGSUS, TAVHL) güvenlik algısına duyarlıdır; bankalar politika belirsizliğinin ilk hedefidir; ihracatçılar kur şokunda göreli kazanır. State which side of the political trade THIS ticker sits on.

5. **Use the event-study base rate.** When the event study has numbers, relay them plainly (örn. "geçmiş N siyasi şokta bu hisse +5 günde medyan %X, pozitif oran %Y") and let them calibrate your magnitude estimate. If it is a placeholder, say so — do not invent history.

6. **Election & calendar awareness.** Seçim öncesi dönemler tipik olarak: artan harcama/popülist adımlar, kur baskısı, yabancı çıkışı, volatilite artışı. Seçim sonrası belirsizliğin çözülmesi tek başına ralli tetikleyebilir (yön sonuçtan bağımsız olarak "belirsizlik primi" iade edilir). Note where we are in the political calendar.

7. **Be honest about data limits.** If the headlines block is a placeholder, lower confidence and say so. NEVER present a risk read as solid when the sources behind it were empty.

## Output format

Write a concise political-risk report with:
- A one-line **headline risk level**: Düşük / Orta / Yüksek / Akut for political risk, with the single most important driver.
- **Aktif kanallar**: which transmission channels are live (kur/CDS, akım, MB bağımsızlığı, sektörel, ticaret) — only those the evidence supports.
- A **historical base rate** line relaying the event-study numbers for this ticker (when available).
- **Bu hisse için net etki**: hangi tarafta (pozitif/nötr/negatif) ve neden — şirketin siyasi duyarlılık profiliyle.
- A closing Markdown table: Risk Faktörü | Durum | Olasılık/Vade | BIST/Hisse Etkisi.

{get_language_instruction()}"""

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


@lru_cache(maxsize=32)
def _cached_fundamental_brief(ticker: str, trade_date: str) -> str:
    """Deterministik temel skor (Piotroski-tarzı) — (ticker, tarih) başına bir kez.

    Düğüm tool-çağrı döngüsünde tekrar çalışır; brif cache'lenir. Hata
    durumunda fail-open placeholder döner (ajan tool'larıyla devam eder).
    """
    try:
        from tradingagents.analytics.fundamental_score import format_fundamental_brief
        return format_fundamental_brief(ticker)
    except Exception as exc:  # noqa: BLE001
        return f"<Deterministik temel skor üretilemedi: {type(exc).__name__}: {exc}>"


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        fundamental_brief = _cached_fundamental_brief(ticker, current_date)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + "\n\n## Pre-computed deterministic fundamental score\n\n"
            "The block below was COMPUTED from real financial statements by the desk's deterministic engine "
            "(Piotroski-style soundness criteria + valuation ratios). Treat the computed criteria as ground "
            "truth and weave the score into your report; your added value is the interpretation (sector "
            "context, inflation adjustment, valuation judgement), not re-deriving the arithmetic. In "
            "high-inflation Türkiye, nominal revenue/profit growth is misleading — emphasize margins, "
            "leverage direction and cash generation instead. If the block is a placeholder, proceed with "
            "tools only.\n\n<start_of_fundamental_score>\n"
            + fundamental_brief
            + "\n<end_of_fundamental_score>"
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node

"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
    resolve_instrument_identity,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.investing_news import fetch_investing_news
from tradingagents.dataflows.symbol_utils import is_bist_ticker
from tradingagents.dataflows.turkish_news import fetch_turkish_market_news


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch sources. Each fetcher degrades gracefully and returns a
        # string (no exceptions surface from here), so the LLM always sees
        # something — either real data or a clear placeholder.
        #
        # BIST (.IS) tickers route to Turkish-language news instead of
        # StockTwits/Reddit, which have no Turkish retail coverage and return
        # empty placeholders for these names (see turkish_news.py).
        if is_bist_ticker(ticker):
            company_name = resolve_instrument_identity(ticker).get("company_name")
            news_block = get_news.func(ticker, start_date, end_date)
            investing_block = fetch_investing_news(ticker)
            turkish_block = fetch_turkish_market_news(ticker, company_name)
            system_message = _build_bist_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                investing_block=investing_block,
                turkish_block=turkish_block,
            )
        else:
            news_block = get_news.func(ticker, start_date, end_date)
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
            reddit_block = fetch_reddit_posts(ticker)
            system_message = _build_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                stocktwits_block=stocktwits_block,
                reddit_block=reddit_block,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


def _build_bist_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    investing_block: str,
    turkish_block: str,
) -> str:
    """Assemble the sentiment-analyst system message for a Borsa İstanbul ticker.

    BIST names have no meaningful StockTwits/Reddit footprint, so this variant
    pairs yfinance's (mostly English, international-desk) news with two Turkish-
    language sources: company-specific Investing.com news that republishes KAP
    material disclosures, and broader Turkish market/macro headlines. Output
    fields match the standard variant so the :class:`SentimentReport` schema is
    unchanged."""
    return f"""You are a financial market sentiment analyst covering Borsa İstanbul (BIST). Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary news sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### International news — Yahoo Finance, past 7 days
Mostly English-language, international-desk framing (global wires, earnings coverage). Fact-driven, slower-moving signal; reflects how foreign investors see the name.

<start_of_news>
{news_block}
<end_of_news>

### Company-specific Turkish news & KAP disclosures — Investing.com TR
Company-level Turkish headlines for this exact ticker. Turkish outlets republish the company's KAP material disclosures (özel durum açıklamaları) here — dividend decisions (kâr/temettü dağıtımı), board/management changes (yönetim kurulu değişikliği), share buybacks (pay alım/satım), capital raises, and major investments. Treat these as the closest available proxy for official disclosure flow; they are events, not opinion, and are the highest-signal company-specific input in this prompt.

<start_of_company_turkish_news>
{investing_block}
<end_of_company_turkish_news>

### Turkish market/macro news — Investing.com TR (borsa) + BloombergHT
Broader local-language market and macro framing for context (BIST direction, TCMB/rate and FX/lira moves, sector themes). Use as backdrop, not as a company-specific catalyst. Note: Turkish retail social platforms (the StockTwits/Reddit equivalents) are not available for BIST names, so this local news flow plus the company block above are your primary local-sentiment proxies.

<start_of_turkish_news>
{turkish_block}
<end_of_turkish_news>

## How to analyze this data (best practices)

1. **Lead with the company-specific Turkish block / KAP disclosures.** A dividend cancellation, management shake-up, or major investment is a concrete, market-moving event. Weight these above both macro headlines and opinion-driven commentary.

2. **Weigh local vs. international framing.** When Turkish-language coverage is bullish but international wires are quiet or bearish (or vice versa), that divergence is itself a signal — local flow frequently leads on Turkish names.

3. **Separate company-specific signal from macro.** The third block is market/macro context. Macro themes (TCMB/interest-rate decisions, inflation prints, FX/lira moves, geopolitics) move the whole market; weight them as backdrop, not as a company-specific catalyst.

4. **Distinguish opinion from event.** A KAP-driven disclosure or earnings headline is an event; an analyst's market commentary is opinion. Both are inputs but should be weighted differently.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That is the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If the company-specific block returned a "<...bulunamadı>" / "<...render edilemedi>" placeholder, or a source is otherwise "<unavailable>", the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative.

7. **Identify catalysts and risks** emerging across sources — upcoming earnings, capacity/expansion news, regulatory or FX exposure, sector-wide moves.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call. Remember prices are quoted in Turkish lira (TRY).

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when both sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size (lower it when only general market headlines were available).
- **narrative**: Full source-by-source breakdown, local-vs-international divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)

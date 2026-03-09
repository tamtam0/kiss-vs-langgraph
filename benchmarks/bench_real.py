"""Real-framework benchmark: actual KISS KISSAgent vs actual LangGraph ODR graph.

Both frameworks use IDENTICAL tool functions (yfinance + web search).
KISS integrates them as Python callables; LangGraph as LangChain tools.
SSL disabled for Zscaler environments.

Requires Python 3.13+ (for KISS framework syntax).
Run with: .venv313/bin/python benchmarks/bench_real.py
"""

import json
import os
import re
import ssl
import sys
import time
import traceback
import urllib3
from dataclasses import asdict, dataclass
from pathlib import Path

# ─── SSL: Disable for Zscaler ───────────────────────────────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
from requests.adapters import HTTPAdapter

_orig_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, request, **kwargs)
requests.Session.send = _patched_send

import httpx
import yfinance as yf

# ─── Config ──────────────────────────────────────────────────────────────────
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "")
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")

# For KISS: if using Databricks, we need a model name starting with "claude-"
# so it routes to AnthropicModel. We'll monkey-patch the client creation.
KISS_MODEL = os.environ.get("KISS_MODEL", "claude-sonnet-4-20250514")  # fallback
if MODEL.startswith("databricks-claude") or MODEL.startswith("us-anthropic-"):
    # Map to a real claude model name for KISS routing
    KISS_MODEL = "claude-sonnet-4-20250514"

# ─── Finance Benchmark Questions ─────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    {
        "id": "q1",
        "topic": (
            "Compare NVIDIA (NVDA) and AMD (AMD) as AI chip investments. "
            "Fetch their current stock prices, 52-week performance, P/E ratios, "
            "revenue growth, and market positioning. Which is the better buy today?"
        ),
    },
    {
        "id": "q2",
        "topic": (
            "Analyze Tesla (TSLA) stock: fetch current price, 52-week high/low, "
            "recent earnings data, and analyst sentiment. Search for recent news "
            "about Tesla's robotaxi plans and their impact on valuation."
        ),
    },
    {
        "id": "q3",
        "topic": (
            "Build a risk assessment of the Magnificent 7 tech stocks (AAPL, MSFT, "
            "GOOGL, AMZN, NVDA, META, TSLA). Fetch their current P/E ratios, "
            "year-to-date returns, and market caps. Search for recent macro risks "
            "affecting big tech. Which stocks are most overvalued?"
        ),
    },
]


# ─── Data Classes ────────────────────────────────────────────────────────────
@dataclass
class BenchResult:
    framework: str
    question_id: str
    question: str
    wall_time_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    output_length: int = 0
    output_preview: str = ""
    tool_calls_made: int = 0
    tools_used: str = ""
    success: bool = False
    error: str = ""
    quality_score: float = 0.0
    quality_reasoning: str = ""
    _full_output: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED TOOL FUNCTIONS — identical logic for both frameworks
# ═══════════════════════════════════════════════════════════════════════════════

def get_stock_info(ticker: str) -> str:
    """Get comprehensive stock information for a ticker symbol using Yahoo Finance.
    Returns current price, market cap, P/E ratio, 52-week high/low, volume,
    earnings data, sector, analyst targets, and more.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA', 'TSLA')
    """
    t = yf.Ticker(ticker)
    info = t.info
    fields = [
        "shortName", "symbol", "currentPrice", "previousClose",
        "marketCap", "trailingPE", "forwardPE", "priceToBook",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "fiftyDayAverage",
        "twoHundredDayAverage", "volume", "averageVolume",
        "dividendYield", "beta", "trailingEps", "forwardEps",
        "revenueGrowth", "earningsGrowth", "profitMargins",
        "returnOnEquity", "totalRevenue", "totalDebt", "totalCash",
        "sector", "industry", "fullTimeEmployees",
        "recommendationKey", "targetMeanPrice", "numberOfAnalystOpinions",
    ]
    result = {}
    for f in fields:
        v = info.get(f)
        if v is not None:
            result[f] = v
    return json.dumps(result, indent=2, default=str)


def get_stock_history(ticker: str, period: str = "6mo") -> str:
    """Get historical price data for a stock. Returns OHLCV summary with
    start/end prices, period high/low, percent change, and recent closes.

    Args:
        ticker: Stock ticker symbol
        period: Time period — one of '1d','5d','1mo','3mo','6mo','1y','2y','5y','ytd','max'
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    if hist.empty:
        return json.dumps({"error": f"No history data for {ticker}"})
    first, last = hist.iloc[0], hist.iloc[-1]
    summary = {
        "ticker": ticker, "period": period, "data_points": len(hist),
        "start_date": str(hist.index[0].date()),
        "end_date": str(hist.index[-1].date()),
        "start_close": round(float(first["Close"]), 2),
        "end_close": round(float(last["Close"]), 2),
        "period_high": round(float(hist["High"].max()), 2),
        "period_low": round(float(hist["Low"].min()), 2),
        "pct_change": round(float((last["Close"] - first["Close"]) / first["Close"] * 100), 2),
        "avg_daily_volume": int(hist["Volume"].mean()),
        "recent_closes": [
            {"date": str(d.date()), "close": round(float(c), 2)}
            for d, c in zip(hist.index[-5:], hist["Close"].iloc[-5:])
        ],
    }
    return json.dumps(summary, indent=2)


def web_search(query: str) -> str:
    """Search the web for recent news, analysis, and information using DuckDuckGo.
    Returns top 5 results with titles, snippets, and URLs.
    Use for current events, analyst opinions, breaking news, and market sentiment.

    Args:
        query: Search query string
    """
    try:
        from duckduckgo_search import DDGS
        raw = list(DDGS().text(query, max_results=5))
        results = []
        for r in raw:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", "")[:300],
                "url": r.get("href", ""),
            })
        if not results:
            return json.dumps({"query": query, "results": [], "note": "No results found."})
        return json.dumps({"query": query, "results": results}, indent=2)
    except Exception as e:
        return json.dumps({"query": query, "error": str(e)})


# The tool list — same functions for both frameworks
SHARED_TOOLS = [get_stock_info, get_stock_history, web_search]


# ─── Cost estimation fallback (for Databricks model names not in KISS pricing table)
# Claude Opus pricing: $15/M input, $75/M output
# Claude Sonnet pricing: $3/M input, $15/M output
def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost using Claude Opus pricing (conservative) when framework returns $0."""
    return (input_tokens * 15.0 / 1_000_000) + (output_tokens * 75.0 / 1_000_000)


# ═══════════════════════════════════════════════════════════════════════════════
# KISS: Real KISSAgent with actual framework code
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_kiss_for_databricks():
    """Monkey-patch KISS's AnthropicModel to work with Databricks endpoint."""
    if not (ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN):
        return  # Using direct Anthropic API, no patch needed

    from kiss.core.models.anthropic_model import AnthropicModel
    from anthropic import Anthropic

    _original_init_method = AnthropicModel.initialize

    def _patched_initialize(self, prompt, attachments=None):
        """Patched to inject Databricks base_url and auth headers."""
        self.client = Anthropic(
            api_key="unused",
            base_url=ANTHROPIC_BASE_URL,
            default_headers={"Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}"},
            http_client=httpx.Client(verify=False),
        )
        # Continue with original logic minus client creation
        content = prompt
        if attachments:
            from kiss.core.models.model import Attachment
            blocks = []
            for att in attachments:
                source = {"type": "base64", "media_type": att.mime_type, "data": att.to_base64()}
                if att.mime_type.startswith("image/"):
                    blocks.append({"type": "image", "source": source})
                elif att.mime_type == "application/pdf":
                    blocks.append({"type": "document", "source": source})
            blocks.append({"type": "text", "text": prompt})
            content = blocks

        self.conversation = [{"role": "user", "content": content}]

    AnthropicModel.initialize = _patched_initialize

    # Also patch model name routing: make KISS accept our Databricks model name
    from kiss.core.models import model_info
    _original_model_fn = model_info.model

    def _patched_model(model_name, model_config=None, token_callback=None):
        """Route custom/Databricks model name through AnthropicModel."""
        # Catch Databricks model names or any non-standard claude model names
        is_custom = (
            model_name.startswith("databricks-claude")
            or model_name.startswith("us-anthropic-")
            or (model_name.startswith("claude") and model_name not in _known_models())
        )
        if is_custom:
            return AnthropicModel(
                model_name=model_name,
                api_key="unused",
                model_config=model_config,
                token_callback=token_callback,
            )
        return _original_model_fn(model_name, model_config, token_callback)

    def _known_models():
        """Get set of known model names from KISS model registry."""
        try:
            from kiss.core.models.model_info import MODEL_INFO
            return set(MODEL_INFO.keys())
        except Exception:
            return set()

    model_info.model = _patched_model


# Tool call counter (shared by both frameworks, reset per question)
_kiss_tool_counter = {"count": 0, "tools": []}
# Token counter for LangGraph (updated via monkey-patched messages.create)
_lg_token_counter = {"input": 0, "output": 0}

def _wrap_tool_counting(fn):
    """Wrap a tool function to count calls."""
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _kiss_tool_counter["count"] += 1
        _kiss_tool_counter["tools"].append(fn.__name__)
        print(f"      [{_kiss_tool_counter['count']}] Tool: {fn.__name__}({json.dumps(kwargs)[:80]})")
        return fn(*args, **kwargs)
    return wrapper


def run_kiss_real(question: str) -> BenchResult:
    """Run actual KISS KISSAgent with real tool-use ReAct loop."""
    from kiss.core.kiss_agent import KISSAgent

    result = BenchResult(framework="KISS", question_id="", question=question)
    _kiss_tool_counter["count"] = 0
    _kiss_tool_counter["tools"] = []
    start = time.time()

    try:
        agent = KISSAgent("bench-kiss")
        wrapped_tools = [_wrap_tool_counting(t) for t in SHARED_TOOLS]

        system_prompt = """You are an expert financial research analyst with access to real-time market data tools.

Your approach:
1. Use get_stock_info to fetch current stock data, prices, and fundamentals
2. Use get_stock_history to get price trends and returns
3. Use web_search to find recent news, analyst opinions, and market context
4. Analyze all gathered data and produce a comprehensive research report

Be thorough: fetch data for ALL tickers mentioned. Cross-reference fundamentals with news.
Write a detailed report with specific numbers, comparisons, and actionable conclusions.
Use markdown formatting with clear sections."""

        output = agent.run(
            model_name=MODEL,
            prompt_template=question,
            system_prompt=system_prompt,
            tools=wrapped_tools,
            is_agentic=True,
            max_steps=15,
            max_budget=5.0,
        )

        result.wall_time_seconds = time.time() - start
        result.total_tokens = agent.total_tokens_used
        result.estimated_cost = agent.budget_used
        # Fallback cost estimation if framework returned $0 (Databricks model names)
        if result.estimated_cost < 0.001 and result.total_tokens > 0:
            result.estimated_cost = _estimate_cost(
                result.total_tokens * 2 // 3,  # rough input/output split
                result.total_tokens // 3,
            )
        result.output_length = len(output)
        result.output_preview = output[:500]
        result._full_output = output
        result.tool_calls_made = _kiss_tool_counter["count"]
        result.tools_used = ",".join(sorted(set(_kiss_tool_counter["tools"])))
        result.success = True

    except Exception as e:
        result.wall_time_seconds = time.time() - start
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH ODR: Real compiled StateGraph with custom tools
# ═══════════════════════════════════════════════════════════════════════════════

def _build_langgraph_odr():
    """Build a LangGraph ODR-style graph using the same shared tools.

    We build the graph from scratch (not importing deep_researcher.py) because
    the ODR code expects specific search APIs (Tavily/OpenAI/Anthropic web search).
    Instead, we create an equivalent 4-stage StateGraph that uses our shared tools.
    """
    import asyncio
    import operator
    from typing import Annotated, Literal, Optional

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool as langchain_tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict

    # ── Wrap shared functions as LangChain tools ──
    @langchain_tool
    def lc_get_stock_info(ticker: str) -> str:
        """Get comprehensive stock information for a ticker symbol using Yahoo Finance.
        Returns current price, market cap, P/E ratio, 52-week high/low, volume,
        earnings data, sector, analyst targets, and more."""
        return get_stock_info(ticker)

    @langchain_tool
    def lc_get_stock_history(ticker: str, period: str = "6mo") -> str:
        """Get historical price data for a stock. Returns OHLCV summary with
        start/end prices, period high/low, percent change, and recent closes."""
        return get_stock_history(ticker, period)

    @langchain_tool
    def lc_web_search(query: str) -> str:
        """Search the web for recent news, analysis, and information.
        Returns top 5 results with titles, snippets, and URLs."""
        return web_search(query)

    lc_tools = [lc_get_stock_info, lc_get_stock_history, lc_web_search]

    # ── State definitions (mirroring ODR) ──
    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]
        research_brief: str
        sub_topics: list[str]
        research_findings: Annotated[list[str], operator.add]
        final_report: str

    # ── Token tracking ──
    # Track tokens via shared counter; reset by runner before each question
    global _lg_token_counter

    # ── LLM setup ──
    from langchain_anthropic import ChatAnthropic

    llm_kwargs = {"model": MODEL, "max_tokens": 4096, "temperature": 0}
    if ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN:
        import anthropic as anthropic_sdk
        llm_kwargs["anthropic_api_url"] = ANTHROPIC_BASE_URL
        llm_kwargs["anthropic_api_key"] = "unused"
        llm_kwargs["default_headers"] = {
            "Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}",
        }
    elif ANTHROPIC_API_KEY:
        llm_kwargs["api_key"] = ANTHROPIC_API_KEY

    base_llm = ChatAnthropic(**llm_kwargs)

    # For Databricks/Zscaler: replace the internal client with one using verify=False
    if ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN:
        import anthropic as anthropic_sdk
        _raw_client = anthropic_sdk.Anthropic(
            api_key="unused",
            base_url=ANTHROPIC_BASE_URL,
            default_headers={"Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}"},
            http_client=httpx.Client(verify=False),
        )
        # Wrap messages.create to capture token usage
        _orig_create = _raw_client.messages.create
        def _tracking_create(*args, **kwargs):
            resp = _orig_create(*args, **kwargs)
            if hasattr(resp, 'usage'):
                _lg_token_counter["input"] += getattr(resp.usage, 'input_tokens', 0)
                _lg_token_counter["output"] += getattr(resp.usage, 'output_tokens', 0)
            return resp
        _raw_client.messages.create = _tracking_create
        base_llm._client = _raw_client

    llm_with_tools = base_llm.bind_tools(lc_tools)

    # ── Stage 1: Write Research Brief ──
    def write_brief(state: AgentState):
        messages = state.get("messages", [])
        question = messages[0].content if messages else ""
        resp = base_llm.invoke([
            SystemMessage(content="You are a research planner. Analyze the question and produce a brief."),
            HumanMessage(content=f"""Analyze this financial research question and create a research brief.
Break it into 3 specific sub-topics that each need data gathering and analysis.

Question: {question}

Return a JSON object with:
- "brief": overall research brief (1-2 paragraphs)
- "sub_topics": list of exactly 3 specific research sub-topics as strings"""),
        ])
        text = resp.content
        sub_topics = [question]
        try:
            jm = re.search(r'\{.*\}', text, re.DOTALL)
            if jm:
                data = json.loads(jm.group())
                parsed = data.get("sub_topics", [])
                if parsed and len(parsed) >= 2:
                    sub_topics = parsed[:3]
        except Exception:
            pass
        return {"research_brief": text, "sub_topics": sub_topics}

    # ── Stage 2: Research each sub-topic (with tools) ──
    def research_subtopic(state: AgentState):
        sub_topics = state.get("sub_topics", [])
        findings = []

        for i, topic in enumerate(sub_topics[:3]):
            print(f"    Stage 2.{i+1}: Researching sub-topic...")
            msgs = [
                SystemMessage(content=f"""You are a financial data researcher. Investigate this specific sub-topic
using the available tools. Fetch real stock data with get_stock_info and get_stock_history.
Search for news with web_search. Be thorough. Report specific numbers.
After gathering data, provide a structured summary (300-600 words)."""),
                HumanMessage(content=f"Research this sub-topic thoroughly: {topic}"),
            ]

            # ReAct loop for this researcher
            for _round in range(6):
                resp = llm_with_tools.invoke(msgs)
                msgs.append(resp)

                if not resp.tool_calls:
                    break

                # Execute tool calls
                for tc in resp.tool_calls:
                    tool_map = {t.name: t for t in lc_tools}
                    tool_fn = tool_map.get(tc["name"])
                    if tool_fn:
                        _kiss_tool_counter["count"] += 1  # reuse counter for LG
                        _kiss_tool_counter["tools"].append(tc["name"].replace("lc_", ""))
                        print(f"      [{_kiss_tool_counter['count']}] Tool: {tc['name']}({json.dumps(tc['args'])[:80]})")
                        try:
                            result = tool_fn.invoke(tc["args"])
                        except Exception as e:
                            result = f"Error: {e}"
                    else:
                        result = f"Unknown tool: {tc['name']}"
                    msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

            # Extract final text from last AI message
            finding = ""
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                    finding = m.content
                    break
            findings.append(f"## Sub-topic {i+1}: {topic}\n\n{finding}")

        return {"research_findings": findings}

    # ── Stage 3: Compress research ──
    def compress_research(state: AgentState):
        print("    Stage 3: Compressing research...")
        findings = state.get("research_findings", [])
        combined = "\n\n---\n\n".join(findings)
        resp = base_llm.invoke([
            HumanMessage(content=f"""Synthesize these research findings into a compressed summary.
Preserve ALL specific data points, numbers, prices, ratios, and key insights.

{combined}

Create a structured synthesis that captures all important quantitative data and qualitative insights."""),
        ])
        return {"research_brief": resp.content}

    # ── Stage 4: Final report ──
    def final_report(state: AgentState):
        print("    Stage 4: Writing final report...")
        messages = state.get("messages", [])
        question = messages[0].content if messages else ""
        compressed = state.get("research_brief", "")
        resp = base_llm.invoke([
            HumanMessage(content=f"""You are a senior financial analyst writing a client-facing research report.

Original question: {question}

Research findings:
{compressed}

Write a comprehensive research report with:
- Executive summary with key numbers
- Detailed analysis with specific data points
- Comparative analysis where applicable
- Risk assessment
- Actionable recommendations
Use markdown formatting. Include specific numbers from the research."""),
        ])
        return {"final_report": resp.content}

    # ── Build the StateGraph ──
    builder = StateGraph(AgentState)
    builder.add_node("write_brief", write_brief)
    builder.add_node("research", research_subtopic)
    builder.add_node("compress", compress_research)
    builder.add_node("final_report", final_report)

    builder.add_edge(START, "write_brief")
    builder.add_edge("write_brief", "research")
    builder.add_edge("research", "compress")
    builder.add_edge("compress", "final_report")
    builder.add_edge("final_report", END)

    return builder.compile()


def run_langgraph_real(question: str) -> BenchResult:
    """Run actual LangGraph compiled StateGraph with real tools."""
    from langchain_core.messages import HumanMessage

    result = BenchResult(framework="LangGraph-ODR", question_id="", question=question)
    _kiss_tool_counter["count"] = 0
    _kiss_tool_counter["tools"] = []
    _lg_token_counter["input"] = 0
    _lg_token_counter["output"] = 0
    start = time.time()

    try:
        graph = _build_langgraph_odr()
        state = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "research_brief": "",
            "sub_topics": [],
            "research_findings": [],
            "final_report": "",
        })

        report = state.get("final_report", "")
        result.wall_time_seconds = time.time() - start
        result.output_length = len(report)
        result.output_preview = report[:500]
        result._full_output = report
        result.tool_calls_made = _kiss_tool_counter["count"]
        result.tools_used = ",".join(sorted(set(_kiss_tool_counter["tools"])))
        result.total_tokens = _lg_token_counter["input"] + _lg_token_counter["output"]
        result.input_tokens = _lg_token_counter["input"]
        result.output_tokens = _lg_token_counter["output"]
        result.estimated_cost = _estimate_cost(_lg_token_counter["input"], _lg_token_counter["output"])
        result.success = True

    except Exception as e:
        result.wall_time_seconds = time.time() - start
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY JUDGE
# ═══════════════════════════════════════════════════════════════════════════════

def judge_quality(question: str, report: str) -> tuple[float, str]:
    """LLM judge scoring data grounding, depth, recency, actionability."""
    import anthropic

    if ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN:
        client = anthropic.Anthropic(
            api_key="unused", base_url=ANTHROPIC_BASE_URL,
            default_headers={"Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}"},
            http_client=httpx.Client(verify=False),
        )
    else:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    resp = client.messages.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": f"""You are an expert financial research evaluator. Rate this report 1-10.

QUESTION: {question}

REPORT:
{report[:4000]}

Rate: data_grounding, analysis_depth, recency, actionability, overall (1-10 each).
Return ONLY JSON: {{"data_grounding":N,"analysis_depth":N,"recency":N,"actionability":N,"overall":N,"reasoning":"brief"}}"""}],
    )
    text = resp.content[0].text
    try:
        jm = re.search(r'\{.*\}', text, re.DOTALL)
        if jm:
            scores = json.loads(jm.group())
            return float(scores.get("overall", 5)), json.dumps(scores)
    except Exception:
        pass
    return 5.0, '{"reasoning":"parse error"}'


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    print("=" * 70)
    print("KISS vs LangGraph-ODR — REAL Framework Benchmark (Tool-Augmented)")
    print("=" * 70)
    print(f"Python:  {sys.version.split()[0]}")
    print(f"Model:   {MODEL}")
    print(f"KISS:    KISSAgent (actual framework)")
    print(f"LangGraph: Compiled StateGraph (actual framework)")
    print(f"Tools:   get_stock_info, get_stock_history, web_search (shared)")
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}")

    # Patch KISS for Databricks if needed
    _patch_kiss_for_databricks()

    # Preflight
    print("\nPreflight: Testing yfinance...")
    try:
        p = yf.Ticker("AAPL").info.get("currentPrice")
        print(f"  AAPL: ${p}")
    except Exception as e:
        print(f"  WARNING: {e}")

    all_results: list[BenchResult] = []

    for q in BENCHMARK_QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"Question {q['id']}: {q['topic'][:70]}...")
        print(f"{'─'*70}")

        # ── KISS ──
        print(f"\n  ▶ KISS (KISSAgent — real framework):")
        kr = run_kiss_real(q["topic"])
        kr.question_id = q["id"]
        if kr.success:
            print(f"    ✓ {kr.wall_time_seconds:.1f}s | "
                  f"{kr.total_tokens} tokens | ${kr.estimated_cost:.4f} | "
                  f"{kr.tool_calls_made} tool calls")
            sc, rs = judge_quality(q["topic"], kr._full_output)
            kr.quality_score = sc
            kr.quality_reasoning = rs
            print(f"    Quality: {sc}/10")
        else:
            print(f"    ✗ FAILED: {kr.error[:120]}")
        all_results.append(kr)

        # ── LangGraph ──
        print(f"\n  ▶ LangGraph-ODR (StateGraph — real framework):")
        lr = run_langgraph_real(q["topic"])
        lr.question_id = q["id"]
        if lr.success:
            print(f"    ✓ {lr.wall_time_seconds:.1f}s | "
                  f"{lr.total_tokens} tokens | ${lr.estimated_cost:.4f} | "
                  f"{lr.tool_calls_made} tool calls")
            sc, rs = judge_quality(q["topic"], lr._full_output)
            lr.quality_score = sc
            lr.quality_reasoning = rs
            print(f"    Quality: {sc}/10")
        else:
            print(f"    ✗ FAILED: {lr.error[:120]}")
        all_results.append(lr)

    # ── Save ──
    def _clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    kiss_ok = [r for r in all_results if r.framework == "KISS" and r.success]
    lg_ok = [r for r in all_results if r.framework == "LangGraph-ODR" and r.success]

    def avg(lst, attr):
        vals = [getattr(r, attr) for r in lst]
        return sum(vals) / len(vals) if vals else 0

    summary = {
        "kiss": {
            "avg_time_s": round(avg(kiss_ok, "wall_time_seconds"), 2),
            "avg_tokens": round(avg(kiss_ok, "total_tokens")),
            "avg_cost": round(avg(kiss_ok, "estimated_cost"), 4),
            "avg_quality": round(avg(kiss_ok, "quality_score"), 1),
            "avg_output_len": round(avg(kiss_ok, "output_length")),
            "avg_tool_calls": round(avg(kiss_ok, "tool_calls_made"), 1),
            "questions_passed": len(kiss_ok),
        },
        "langgraph_odr": {
            "avg_time_s": round(avg(lg_ok, "wall_time_seconds"), 2),
            "avg_tokens": round(avg(lg_ok, "total_tokens")),
            "avg_cost": round(avg(lg_ok, "estimated_cost"), 4),
            "avg_quality": round(avg(lg_ok, "quality_score"), 1),
            "avg_output_len": round(avg(lg_ok, "output_length")),
            "avg_tool_calls": round(avg(lg_ok, "tool_calls_made"), 1),
            "questions_passed": len(lg_ok),
        },
    }

    output = {
        "benchmark_type": "real-framework-tool-augmented",
        "python_version": sys.version.split()[0],
        "model": MODEL,
        "kiss_version": "KISSAgent (kiss-agent-framework)",
        "langgraph_version": "StateGraph (langgraph)",
        "tools": ["get_stock_info", "get_stock_history", "web_search"],
        "benchmark_results": [_clean(asdict(r)) for r in all_results],
        "summary": summary,
    }

    output_path = Path(__file__).parent / "benchmark_real_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — Real Framework Benchmark")
    print("=" * 70)

    if kiss_ok and lg_ok:
        k, l = summary["kiss"], summary["langgraph_odr"]
        print(f"\n{'Metric':<20} {'KISS':>12} {'LangGraph':>12} {'Ratio':>12}")
        print(f"{'─'*56}")
        if k['avg_time_s'] > 0:
            print(f"{'Avg Time':<20} {k['avg_time_s']:>10.1f}s {l['avg_time_s']:>10.1f}s {l['avg_time_s']/max(k['avg_time_s'],0.1):>10.1f}x")
        if k['avg_tokens'] > 0:
            print(f"{'Avg Tokens':<20} {k['avg_tokens']:>12,} {l['avg_tokens']:>12,}")
        if k['avg_cost'] > 0:
            print(f"{'Avg Cost':<20} ${k['avg_cost']:>10.4f} ${l['avg_cost']:>10.4f}")
        print(f"{'Avg Tool Calls':<20} {k['avg_tool_calls']:>12.1f} {l['avg_tool_calls']:>12.1f} {l['avg_tool_calls']/max(k['avg_tool_calls'],0.1):>10.1f}x")
        print(f"{'Avg Quality':<20} {k['avg_quality']:>11.1f} {l['avg_quality']:>11.1f}")
        print(f"{'Avg Output':<20} {k['avg_output_len']:>10,}c {l['avg_output_len']:>10,}c")
        print(f"{'Passed':<20} {k['questions_passed']:>12} {l['questions_passed']:>12}")

    print(f"\n{'='*70}")
    return output


if __name__ == "__main__":
    run_benchmark()

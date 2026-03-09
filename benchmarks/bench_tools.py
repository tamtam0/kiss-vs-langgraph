"""Tool-augmented benchmark: KISS vs LangGraph-ODR with real tools.

Both frameworks get access to yfinance and web search tools.
KISS uses a single-agent ReAct loop; LangGraph-ODR uses a multi-stage pipeline
where each researcher sub-agent has tool access.

Designed for Zscaler/corporate proxy environments (SSL verification disabled).
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

# ─── SSL: Disable verification for Zscaler environments ─────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests to skip SSL globally
import requests
from requests.adapters import HTTPAdapter
_original_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs["verify"] = False
    return _original_send(self, request, **kwargs)
requests.Session.send = _patched_send

import anthropic
import yfinance as yf

# ─── Config ──────────────────────────────────────────────────────────────────
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "")
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
MAX_TOOL_ROUNDS = 10  # max ReAct iterations per agent

# ─── Finance Benchmark Questions ─────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    {
        "id": "q1",
        "topic": (
            "Compare NVIDIA (NVDA) and AMD (AMD) as AI chip investments. "
            "Analyze their current stock prices, 52-week performance, P/E ratios, "
            "revenue growth, and market positioning. Which is the better buy today?"
        ),
        "complexity": "high",
    },
    {
        "id": "q2",
        "topic": (
            "Analyze Tesla (TSLA) stock: fetch current price, 52-week high/low, "
            "recent earnings data, and analyst sentiment. Then search for recent news "
            "about Tesla's robotaxi plans and their impact on valuation."
        ),
        "complexity": "high",
    },
    {
        "id": "q3",
        "topic": (
            "Build a risk assessment of the Magnificent 7 tech stocks (AAPL, MSFT, "
            "GOOGL, AMZN, NVDA, META, TSLA). Fetch their current P/E ratios, "
            "year-to-date returns, and market caps. Search for recent macro risks "
            "affecting big tech. Which stocks are most overvalued?"
        ),
        "complexity": "high",
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
    tools_used: str = ""  # comma-separated list
    success: bool = False
    error: str = ""
    quality_score: float = 0.0
    quality_reasoning: str = ""
    _full_output: str = ""


# ─── Anthropic Client ────────────────────────────────────────────────────────
def get_client():
    """Create Anthropic client with SSL disabled for Zscaler."""
    import httpx
    http_client = httpx.Client(verify=False)

    if ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN:
        return anthropic.Anthropic(
            api_key="unused",
            base_url=ANTHROPIC_BASE_URL,
            default_headers={"Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}"},
            http_client=http_client,
        )
    elif ANTHROPIC_API_KEY:
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, http_client=http_client)
    else:
        print("ERROR: Set ANTHROPIC_API_KEY or both ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN")
        sys.exit(1)


# ─── Tool Definitions (Anthropic format) ─────────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "name": "yfinance_get_stock_info",
        "description": (
            "Get comprehensive stock information for a ticker symbol using Yahoo Finance. "
            "Returns current price, market cap, P/E ratio, 52-week high/low, volume, "
            "earnings data, sector, and more."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'AAPL', 'NVDA', 'TSLA')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "yfinance_get_history",
        "description": (
            "Get historical price data for a stock over a specified period. "
            "Returns OHLCV data (Open, High, Low, Close, Volume). "
            "Use for calculating returns, trends, and technical analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "period": {
                    "type": "string",
                    "description": "Time period: '1d','5d','1mo','3mo','6mo','1y','2y','5y','ytd','max'",
                    "default": "6mo",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for recent news, analysis, and information. "
            "Returns a summary of top search results with titles, snippets, and URLs. "
            "Use for current events, analyst opinions, breaking news, and market sentiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
            },
            "required": ["query"],
        },
    },
]


# ─── Tool Implementations ────────────────────────────────────────────────────
def execute_tool(name: str, inputs: dict) -> str:
    """Execute a tool by name and return the result as a string."""
    try:
        if name == "yfinance_get_stock_info":
            return _yf_get_info(inputs["ticker"])
        elif name == "yfinance_get_history":
            return _yf_get_history(inputs["ticker"], inputs.get("period", "6mo"))
        elif name == "web_search":
            return _web_search(inputs["query"])
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _yf_get_info(ticker: str) -> str:
    """Fetch stock info via yfinance."""
    t = yf.Ticker(ticker)
    info = t.info
    # Extract the most useful fields
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


def _yf_get_history(ticker: str, period: str) -> str:
    """Fetch price history via yfinance."""
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    if hist.empty:
        return json.dumps({"error": f"No history data for {ticker}"})

    # Summarize instead of dumping everything
    first = hist.iloc[0]
    last = hist.iloc[-1]
    high = hist["High"].max()
    low = hist["Low"].min()
    avg_vol = hist["Volume"].mean()
    pct_change = ((last["Close"] - first["Close"]) / first["Close"]) * 100

    summary = {
        "ticker": ticker,
        "period": period,
        "data_points": len(hist),
        "start_date": str(hist.index[0].date()),
        "end_date": str(hist.index[-1].date()),
        "start_close": round(float(first["Close"]), 2),
        "end_close": round(float(last["Close"]), 2),
        "period_high": round(float(high), 2),
        "period_low": round(float(low), 2),
        "pct_change": round(float(pct_change), 2),
        "avg_daily_volume": int(avg_vol),
        # Last 5 trading days
        "recent_closes": [
            {"date": str(d.date()), "close": round(float(c), 2)}
            for d, c in zip(hist.index[-5:], hist["Close"].iloc[-5:])
        ],
    }
    return json.dumps(summary, indent=2)


def _web_search(query: str) -> str:
    """Web search using DuckDuckGo HTML (no API key needed, works behind Zscaler)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=15,
            verify=False,
        )
        resp.raise_for_status()

        # Parse results from HTML
        results = []
        # Extract result snippets using regex (avoid BeautifulSoup dependency)
        # DuckDuckGo HTML results have class="result__snippet"
        snippets = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet">(.*?)</(?:span|div)',
            resp.text, re.DOTALL
        )
        for url, title, snippet in snippets[:5]:
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            # DuckDuckGo wraps URLs in redirect; extract actual URL
            actual_url = url
            if "uddg=" in url:
                match = re.search(r'uddg=([^&]+)', url)
                if match:
                    from urllib.parse import unquote
                    actual_url = unquote(match.group(1))
            results.append({"title": title, "snippet": snippet, "url": actual_url})

        if not results:
            return json.dumps({
                "query": query,
                "results": [],
                "note": "No results parsed. The search may have been blocked or returned no matches.",
            })

        return json.dumps({"query": query, "results": results}, indent=2)

    except Exception as e:
        return json.dumps({"query": query, "error": str(e)})


# ─── ReAct Tool-Use Loop ─────────────────────────────────────────────────────
def call_llm_with_tools(
    messages: list[dict],
    system: str,
    tools: list[dict],
    max_tokens: int = 4096,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> tuple[str, dict, int, list[str]]:
    """
    Run a ReAct-style tool-use loop.

    Returns: (final_text, cumulative_usage, tool_call_count, tools_used_list)
    """
    client = get_client()
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    tool_call_count = 0
    tools_used = []

    for round_num in range(max_rounds):
        kwargs = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "messages": messages,
            "system": system,
        }
        if tools:
            kwargs["tools"] = tools

        resp = client.messages.create(**kwargs)
        total_usage["input_tokens"] += resp.usage.input_tokens
        total_usage["output_tokens"] += resp.usage.output_tokens

        # Check if the model wants to use tools
        if resp.stop_reason == "tool_use":
            # Process all tool calls in this response
            assistant_content = resp.content
            tool_results = []

            for block in assistant_content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    tool_name = block.name
                    tool_input = block.input
                    tools_used.append(tool_name)

                    print(f"      [{round_num+1}] Tool: {tool_name}({json.dumps(tool_input)[:80]})")
                    result = execute_tool(tool_name, tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Add assistant response and tool results to messages
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        else:
            # Model is done — extract final text
            final_text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text, total_usage, tool_call_count, tools_used

    # Hit max rounds — extract whatever text we have
    final_text = ""
    for block in resp.content:
        if hasattr(block, "text"):
            final_text += block.text
    return final_text, total_usage, tool_call_count, tools_used


# ─── KISS: Single-Agent ReAct with Tools ─────────────────────────────────────
def run_kiss_with_tools(question: str) -> BenchResult:
    """KISS approach: one agent, full tool access, ReAct loop."""
    result = BenchResult(framework="KISS", question_id="", question=question)
    start = time.time()

    try:
        system = """You are an expert financial research analyst with access to real-time market data tools.

Your approach:
1. Use yfinance tools to fetch current stock data, prices, fundamentals, and history
2. Use web_search to find recent news, analyst opinions, and market context
3. Analyze all gathered data and produce a comprehensive research report

Be thorough: fetch data for ALL tickers mentioned. Cross-reference fundamentals with news.
Write a detailed report with specific numbers, comparisons, and actionable conclusions.
Use markdown formatting with clear sections."""

        messages = [{"role": "user", "content": question}]

        text, usage, tool_count, tools_list = call_llm_with_tools(
            messages=messages,
            system=system,
            tools=TOOL_DEFINITIONS,
            max_tokens=4096,
        )

        result.wall_time_seconds = time.time() - start
        result.input_tokens = usage["input_tokens"]
        result.output_tokens = usage["output_tokens"]
        result.total_tokens = usage["input_tokens"] + usage["output_tokens"]
        result.estimated_cost = (usage["input_tokens"] * 15 + usage["output_tokens"] * 75) / 1_000_000
        result.output_length = len(text)
        result.output_preview = text[:500]
        result._full_output = text
        result.tool_calls_made = tool_count
        result.tools_used = ",".join(sorted(set(tools_list)))
        result.success = True

    except Exception as e:
        result.wall_time_seconds = time.time() - start
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ─── LangGraph-ODR: Multi-Stage Pipeline with Tools ─────────────────────────
def run_langgraph_with_tools(question: str) -> BenchResult:
    """LangGraph ODR approach: multi-stage pipeline, each researcher has tools."""
    result = BenchResult(framework="LangGraph-ODR", question_id="", question=question)
    total_input = 0
    total_output = 0
    total_tool_calls = 0
    all_tools_used = []
    start = time.time()

    try:
        client = get_client()

        # ── Stage 1: Write Research Brief (no tools) ──
        print("    Stage 1: Research brief...")
        brief_resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""Analyze this financial research question and create a research brief.
Break it into 3 specific sub-topics that each need data gathering and analysis.

Research question: {question}

Return a JSON object with:
- "brief": overall research brief (1-2 paragraphs)
- "sub_topics": list of 3 specific sub-topics to research, each as a detailed string"""}],
        )
        brief_text = brief_resp.content[0].text
        total_input += brief_resp.usage.input_tokens
        total_output += brief_resp.usage.output_tokens

        # Parse sub-topics
        sub_topics = [question]
        try:
            json_match = re.search(r'\{.*\}', brief_text, re.DOTALL)
            if json_match:
                brief_data = json.loads(json_match.group())
                parsed = brief_data.get("sub_topics", [])
                if parsed and len(parsed) >= 2:
                    sub_topics = parsed[:3]
        except Exception:
            pass

        # ── Stage 2: Research each sub-topic (with tools) ──
        research_findings = []
        for i, topic in enumerate(sub_topics[:3]):
            print(f"    Stage 2.{i+1}: Researching sub-topic...")
            researcher_system = f"""You are a financial data researcher. Your task is to investigate ONE specific aspect
of a larger research question using the available tools.

Use yfinance tools to get real stock data. Use web_search for news and context.
Be thorough — fetch data for every relevant ticker. Report specific numbers.

Sub-topic to research: {topic}

After gathering data, provide a structured summary of your findings (300-600 words)."""

            messages = [{"role": "user", "content": f"Research this sub-topic thoroughly using the available tools: {topic}"}]

            finding, usage_r, tc, tl = call_llm_with_tools(
                messages=messages,
                system=researcher_system,
                tools=TOOL_DEFINITIONS,
                max_tokens=2048,
                max_rounds=6,  # fewer rounds per researcher
            )
            total_input += usage_r["input_tokens"]
            total_output += usage_r["output_tokens"]
            total_tool_calls += tc
            all_tools_used.extend(tl)
            research_findings.append(f"## Sub-topic {i+1}: {topic}\n\n{finding}")

        # ── Stage 3: Compress & synthesize (no tools) ──
        print("    Stage 3: Compressing research...")
        combined = "\n\n---\n\n".join(research_findings)
        compress_resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": f"""Synthesize these research findings into a compressed summary.
Preserve ALL specific data points, numbers, prices, ratios, and key insights.

{combined}

Create a structured synthesis that captures all important quantitative data and qualitative insights."""}],
        )
        compressed = compress_resp.content[0].text
        total_input += compress_resp.usage.input_tokens
        total_output += compress_resp.usage.output_tokens

        # ── Stage 4: Final report (no tools) ──
        print("    Stage 4: Writing final report...")
        report_resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": f"""You are a senior financial analyst writing a client-facing research report.

Original question: {question}

Research findings:
{compressed}

Write a comprehensive research report with:
- Executive summary with key numbers
- Detailed analysis with specific data points (prices, P/E, revenue, etc.)
- Comparative analysis where applicable
- Risk assessment
- Actionable recommendations
Use markdown formatting. Include specific numbers from the research."""}],
        )
        final_report = report_resp.content[0].text
        total_input += report_resp.usage.input_tokens
        total_output += report_resp.usage.output_tokens

        result.wall_time_seconds = time.time() - start
        result.input_tokens = total_input
        result.output_tokens = total_output
        result.total_tokens = total_input + total_output
        result.estimated_cost = (total_input * 15 + total_output * 75) / 1_000_000
        result.output_length = len(final_report)
        result.output_preview = final_report[:500]
        result._full_output = final_report
        result.tool_calls_made = total_tool_calls
        result.tools_used = ",".join(sorted(set(all_tools_used)))
        result.success = True

    except Exception as e:
        result.wall_time_seconds = time.time() - start
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ─── Quality Judge ───────────────────────────────────────────────────────────
def judge_quality(question: str, report: str) -> tuple[float, str]:
    """LLM judge for financial research quality. Evaluates data grounding."""
    client = get_client()
    judge_prompt = f"""You are an expert financial research evaluator. Rate this report on a 1-10 scale.

RESEARCH QUESTION: {question}

REPORT:
{report[:4000]}

Rate on these dimensions:
1. Data grounding — Does it cite specific prices, ratios, market caps? (1-10)
2. Analysis depth — Beyond just listing numbers, does it interpret them? (1-10)
3. Recency — Does it reference current/recent data, not just general knowledge? (1-10)
4. Actionability — Are the conclusions specific and investment-relevant? (1-10)
5. Overall quality (1-10)

Return ONLY a JSON object:
{{"data_grounding": N, "analysis_depth": N, "recency": N, "actionability": N, "overall": N, "reasoning": "brief explanation"}}"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = resp.content[0].text
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            return float(scores.get("overall", 5)), json.dumps(scores)
    except Exception:
        pass
    return 5.0, '{"reasoning": "Could not parse quality score"}'


# ─── Main Runner ─────────────────────────────────────────────────────────────
def run_benchmark():
    """Run the tool-augmented benchmark suite."""
    print("=" * 70)
    print("KISS vs LangGraph-ODR — Tool-Augmented Finance Benchmark")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Tools: yfinance_get_stock_info, yfinance_get_history, web_search")
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}")
    print(f"Max tool rounds per agent: {MAX_TOOL_ROUNDS}")

    # Quick tool sanity check
    print("\nPreflight: Testing yfinance...")
    try:
        test = yf.Ticker("AAPL").info.get("currentPrice")
        print(f"  AAPL current price: ${test} ✓")
    except Exception as e:
        print(f"  WARNING: yfinance test failed: {e}")

    all_results: list[BenchResult] = []

    for q in BENCHMARK_QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"Question {q['id']}: {q['topic'][:70]}...")
        print(f"{'─'*70}")

        # Run KISS
        print(f"\n  ▶ KISS (single-agent ReAct):")
        kiss_result = run_kiss_with_tools(q["topic"])
        kiss_result.question_id = q["id"]
        if kiss_result.success:
            print(f"    ✓ {kiss_result.wall_time_seconds:.1f}s | "
                  f"{kiss_result.total_tokens} tokens | "
                  f"${kiss_result.estimated_cost:.4f} | "
                  f"{kiss_result.tool_calls_made} tool calls")
            score, reasoning = judge_quality(q["topic"], kiss_result._full_output)
            kiss_result.quality_score = score
            kiss_result.quality_reasoning = reasoning
            print(f"    Quality: {score}/10")
        else:
            print(f"    ✗ FAILED: {kiss_result.error[:120]}")
        all_results.append(kiss_result)

        # Run LangGraph-ODR
        print(f"\n  ▶ LangGraph-ODR (multi-stage pipeline):")
        lg_result = run_langgraph_with_tools(q["topic"])
        lg_result.question_id = q["id"]
        if lg_result.success:
            print(f"    ✓ {lg_result.wall_time_seconds:.1f}s | "
                  f"{lg_result.total_tokens} tokens | "
                  f"${lg_result.estimated_cost:.4f} | "
                  f"{lg_result.tool_calls_made} tool calls")
            score, reasoning = judge_quality(q["topic"], lg_result._full_output)
            lg_result.quality_score = score
            lg_result.quality_reasoning = reasoning
            print(f"    Quality: {score}/10")
        else:
            print(f"    ✗ FAILED: {lg_result.error[:120]}")
        all_results.append(lg_result)

    # ── Results ──
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
        "benchmark_type": "tool-augmented-finance",
        "model": MODEL,
        "tools": ["yfinance_get_stock_info", "yfinance_get_history", "web_search"],
        "benchmark_results": [_clean(asdict(r)) for r in all_results],
        "summary": summary,
    }

    output_path = Path(__file__).parent / "benchmark_tools_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY — Tool-Augmented Finance Benchmark")
    print("=" * 70)

    if kiss_ok and lg_ok:
        k, l = summary["kiss"], summary["langgraph_odr"]
        print(f"\n{'Metric':<20} {'KISS':>12} {'LangGraph':>12} {'Ratio':>12}")
        print(f"{'─'*56}")
        print(f"{'Avg Time':<20} {k['avg_time_s']:>10.1f}s {l['avg_time_s']:>10.1f}s {l['avg_time_s']/k['avg_time_s']:>10.1f}x")
        print(f"{'Avg Tokens':<20} {k['avg_tokens']:>12,} {l['avg_tokens']:>12,} {l['avg_tokens']/k['avg_tokens']:>10.1f}x")
        print(f"{'Avg Cost':<20} ${k['avg_cost']:>10.4f} ${l['avg_cost']:>10.4f} {l['avg_cost']/k['avg_cost']:>10.1f}x")
        print(f"{'Avg Tool Calls':<20} {k['avg_tool_calls']:>12.1f} {l['avg_tool_calls']:>12.1f} {l['avg_tool_calls']/max(k['avg_tool_calls'],0.1):>10.1f}x")
        print(f"{'Avg Quality':<20} {k['avg_quality']:>11.1f} {l['avg_quality']:>11.1f} {'':>12}")
        print(f"{'Avg Output':<20} {k['avg_output_len']:>10,}c {l['avg_output_len']:>10,}c {'':>12}")
        print(f"{'Passed':<20} {k['questions_passed']:>12} {l['questions_passed']:>12} {'':>12}")

    print(f"\n{'='*70}")
    return output


if __name__ == "__main__":
    run_benchmark()

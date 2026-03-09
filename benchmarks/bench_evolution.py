"""KISS prompt evolution benchmark: measures quality improvement across runs.

Uses KISS's built-in prompt_refiner_agent to evolve the system prompt between runs
based on the agent's trajectory. This demonstrates GEPA-style reflective prompt
evolution — the same mechanism KISS uses for iterative self-improvement.

Run 1: Baseline (static system prompt)
Run 2: Evolved prompt (refined using Run 1's trajectory)
Run 3: Further evolved (refined using Run 2's trajectory)

Requires Python 3.13+ (for KISS framework syntax).
Run with: .venv313/bin/python benchmarks/bench_evolution.py
"""

import json
import os
import re
import ssl
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

# ─── SSL: Disable for Zscaler ───────────────────────────────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
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

NUM_EVOLUTION_RUNS = 3  # Baseline + 2 evolved runs

# ─── Benchmark Questions ─────────────────────────────────────────────────────
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


@dataclass
class EvolutionResult:
    question_id: str
    question: str
    run_number: int  # 1=baseline, 2=1st evolution, 3=2nd evolution
    system_prompt_used: str = ""
    wall_time_seconds: float = 0.0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    tool_calls_made: int = 0
    tools_used: str = ""
    output_length: int = 0
    output_preview: str = ""
    quality_score: float = 0.0
    quality_reasoning: str = ""
    success: bool = False
    error: str = ""
    _full_output: str = ""
    _trajectory: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED TOOL FUNCTIONS (same as bench_real.py)
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
        "avg_volume": int(hist["Volume"].mean()),
        "last_5_closes": [round(float(x), 2) for x in hist["Close"].tail(5).tolist()],
    }
    return json.dumps(summary, indent=2)


def web_search(query: str) -> str:
    """Search the web for recent news, analysis, and information.
    Returns top 5 results with titles, snippets, and URLs.

    Args:
        query: Search query string
    """
    try:
        from duckduckgo_search import DDGS
        raw = list(DDGS().text(query, max_results=5))
        results = [
            {"title": r.get("title", ""), "snippet": r.get("body", "")[:300], "url": r.get("href", "")}
            for r in raw
        ]
        return json.dumps({"query": query, "results": results}, indent=2)
    except Exception as e:
        return json.dumps({"query": query, "error": str(e), "results": []})


SHARED_TOOLS = [get_stock_info, get_stock_history, web_search]

# ─── Cost estimation ─────────────────────────────────────────────────────────
def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * 15.0 / 1_000_000) + (output_tokens * 75.0 / 1_000_000)


# ═══════════════════════════════════════════════════════════════════════════════
# KISS PATCHES (same as bench_real.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_kiss_for_databricks():
    if not (ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN):
        return

    from kiss.core.models.anthropic_model import AnthropicModel
    from anthropic import Anthropic

    _original_init_method = AnthropicModel.initialize

    def _patched_initialize(self, prompt, attachments=None):
        self.client = Anthropic(
            api_key="unused",
            base_url=ANTHROPIC_BASE_URL,
            default_headers={"Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}"},
            http_client=httpx.Client(verify=False),
        )
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

    from kiss.core.models import model_info
    _original_model_fn = model_info.model

    def _patched_model(model_name, model_config=None, token_callback=None):
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
        try:
            from kiss.core.models.model_info import MODEL_INFO
            return set(MODEL_INFO.keys())
        except Exception:
            return set()

    model_info.model = _patched_model


# ─── Tool counting ───────────────────────────────────────────────────────────
_tool_counter = {"count": 0, "tools": []}

def _wrap_tool_counting(fn):
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _tool_counter["count"] += 1
        _tool_counter["tools"].append(fn.__name__)
        print(f"      [{_tool_counter['count']}] Tool: {fn.__name__}({json.dumps(kwargs)[:80]})")
        return fn(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_SYSTEM_PROMPT = """You are an expert financial research analyst with access to real-time market data tools.

Your approach:
1. Use get_stock_info to fetch current stock data, prices, and fundamentals
2. Use get_stock_history to get price trends and returns
3. Use web_search to find recent news, analyst opinions, and market context
4. Analyze all gathered data and produce a comprehensive research report

Be thorough: fetch data for ALL tickers mentioned. Cross-reference fundamentals with news.
Write a detailed report with specific numbers, comparisons, and actionable conclusions.
Use markdown formatting with clear sections."""


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE KISS RUN — returns result + trajectory for prompt evolution
# ═══════════════════════════════════════════════════════════════════════════════

def run_kiss_once(question: str, system_prompt: str, run_number: int) -> EvolutionResult:
    """Run KISS agent once with the given system prompt. Returns result + trajectory."""
    from kiss.core.kiss_agent import KISSAgent

    result = EvolutionResult(
        question_id="", question=question, run_number=run_number,
        system_prompt_used=system_prompt,
    )
    _tool_counter["count"] = 0
    _tool_counter["tools"] = []
    start = time.time()

    try:
        agent = KISSAgent(f"bench-kiss-run{run_number}")
        wrapped_tools = [_wrap_tool_counting(t) for t in SHARED_TOOLS]

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
        if result.estimated_cost < 0.001 and result.total_tokens > 0:
            result.estimated_cost = _estimate_cost(
                result.total_tokens * 2 // 3,
                result.total_tokens // 3,
            )
        result.output_length = len(output)
        result.output_preview = output[:500]
        result._full_output = output
        result._trajectory = agent.get_trajectory()
        result.tool_calls_made = _tool_counter["count"]
        result.tools_used = ",".join(sorted(set(_tool_counter["tools"])))
        result.success = True

    except Exception as e:
        result.wall_time_seconds = time.time() - start
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT EVOLUTION — uses KISS's built-in prompt_refiner_agent
# ═══════════════════════════════════════════════════════════════════════════════

def evolve_prompt(original_prompt: str, previous_prompt: str, trajectory: str) -> str:
    """Use KISS's prompt_refiner_agent to evolve the system prompt.

    This is the same mechanism KISS uses internally for prompt evolution —
    a non-agentic LLM call that analyzes the trajectory and produces a
    refined prompt that addresses weaknesses while preserving strengths.
    """
    from kiss.agents.kiss import prompt_refiner_agent

    print("    Evolving prompt via prompt_refiner_agent...")
    # Truncate trajectory to avoid exceeding context
    max_traj = 8000
    if len(trajectory) > max_traj:
        trajectory = trajectory[:max_traj] + "\n... [truncated]"

    try:
        refined = prompt_refiner_agent(
            original_prompt_template=original_prompt,
            previous_prompt_template=previous_prompt,
            agent_trajectory_summary=trajectory,
            model_name=MODEL,
        )
        # The refiner returns the full refined prompt
        if refined and len(refined.strip()) > 50:
            print(f"    Prompt evolved ({len(previous_prompt)} -> {len(refined.strip())} chars)")
            return refined.strip()
        else:
            print("    Warning: prompt_refiner returned short result, keeping previous")
            return previous_prompt
    except Exception as e:
        print(f"    Warning: prompt evolution failed ({e}), keeping previous prompt")
        return previous_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY JUDGE (same as bench_real.py)
# ═══════════════════════════════════════════════════════════════════════════════

def judge_quality(question: str, report: str) -> tuple[float, str]:
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

def run_evolution_benchmark():
    print("=" * 70)
    print("KISS Prompt Evolution Benchmark")
    print("=" * 70)
    print(f"Python:      {sys.version.split()[0]}")
    print(f"Model:       {MODEL}")
    print(f"Framework:   KISS KISSAgent (actual framework)")
    print(f"Evolution:   prompt_refiner_agent (built-in KISS prompt evolution)")
    print(f"Runs/Q:      {NUM_EVOLUTION_RUNS} (1 baseline + {NUM_EVOLUTION_RUNS-1} evolved)")
    print(f"Tools:       get_stock_info, get_stock_history, web_search")
    print(f"Questions:   {len(BENCHMARK_QUESTIONS)}")

    _patch_kiss_for_databricks()

    # Preflight
    print("\nPreflight: Testing yfinance...")
    try:
        p = yf.Ticker("AAPL").info.get("currentPrice")
        print(f"  AAPL: ${p}")
    except Exception as e:
        print(f"  WARNING: {e}")

    all_results: list[EvolutionResult] = []

    for q in BENCHMARK_QUESTIONS:
        print(f"\n{'='*70}")
        print(f"Question {q['id']}: {q['topic'][:70]}...")
        print(f"{'='*70}")

        current_prompt = BASELINE_SYSTEM_PROMPT

        for run_num in range(1, NUM_EVOLUTION_RUNS + 1):
            run_label = "Baseline" if run_num == 1 else f"Evolution #{run_num - 1}"
            print(f"\n  ▶ Run {run_num}/{NUM_EVOLUTION_RUNS} ({run_label}):")
            if run_num > 1:
                print(f"    System prompt length: {len(current_prompt)} chars")

            # Run KISS
            er = run_kiss_once(q["topic"], current_prompt, run_num)
            er.question_id = q["id"]

            if er.success:
                print(f"    ✓ {er.wall_time_seconds:.1f}s | "
                      f"{er.total_tokens} tokens | ${er.estimated_cost:.4f} | "
                      f"{er.tool_calls_made} tool calls")
                # Judge quality
                sc, rs = judge_quality(q["topic"], er._full_output)
                er.quality_score = sc
                er.quality_reasoning = rs
                print(f"    Quality: {sc}/10")

                # Evolve prompt for next run (if not the last run)
                if run_num < NUM_EVOLUTION_RUNS and er._trajectory:
                    current_prompt = evolve_prompt(
                        original_prompt=BASELINE_SYSTEM_PROMPT,
                        previous_prompt=current_prompt,
                        trajectory=er._trajectory,
                    )
            else:
                print(f"    ✗ FAILED: {er.error[:120]}")

            all_results.append(er)

        # Reset prompt for next question
        current_prompt = BASELINE_SYSTEM_PROMPT

    # ── Build summary per run number ──
    def _clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    def avg(lst, attr):
        vals = [getattr(r, attr) for r in lst if r.success]
        return sum(vals) / len(vals) if vals else 0

    run_summaries = {}
    for run_num in range(1, NUM_EVOLUTION_RUNS + 1):
        run_results = [r for r in all_results if r.run_number == run_num and r.success]
        label = "baseline" if run_num == 1 else f"evolution_{run_num - 1}"
        run_summaries[label] = {
            "run_number": run_num,
            "avg_time_s": round(avg(run_results, "wall_time_seconds"), 2),
            "avg_tokens": round(avg(run_results, "total_tokens")),
            "avg_cost": round(avg(run_results, "estimated_cost"), 4),
            "avg_quality": round(avg(run_results, "quality_score"), 1),
            "avg_tool_calls": round(avg(run_results, "tool_calls_made"), 1),
            "avg_output_len": round(avg(run_results, "output_length")),
            "questions_passed": len(run_results),
        }

    # Also build per-question evolution tracking
    evolution_tracking = {}
    for q in BENCHMARK_QUESTIONS:
        q_results = [r for r in all_results if r.question_id == q["id"] and r.success]
        evolution_tracking[q["id"]] = {
            "topic": q["topic"][:80],
            "runs": [
                {
                    "run": r.run_number,
                    "quality": r.quality_score,
                    "time_s": round(r.wall_time_seconds, 1),
                    "tokens": r.total_tokens,
                    "cost": round(r.estimated_cost, 4),
                    "tool_calls": r.tool_calls_made,
                }
                for r in q_results
            ],
        }

    output = {
        "benchmark_type": "kiss-prompt-evolution",
        "python_version": sys.version.split()[0],
        "model": MODEL,
        "framework": "KISS KISSAgent (actual framework)",
        "evolution_method": "prompt_refiner_agent (built-in KISS prompt evolution)",
        "num_runs_per_question": NUM_EVOLUTION_RUNS,
        "tools": ["get_stock_info", "get_stock_history", "web_search"],
        "benchmark_results": [_clean(asdict(r)) for r in all_results],
        "run_summaries": run_summaries,
        "evolution_tracking": evolution_tracking,
    }

    output_path = Path(__file__).parent / "benchmark_evolution_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("EVOLUTION SUMMARY — Quality Improvement Across Runs")
    print("=" * 70)

    print(f"\n{'Metric':<20}", end="")
    for label in run_summaries:
        print(f" {label:>16}", end="")
    print()
    print("─" * (20 + 17 * len(run_summaries)))

    for metric, fmt in [
        ("avg_quality", "{:>15.1f}"),
        ("avg_time_s", "{:>14.1f}s"),
        ("avg_tokens", "{:>15,}"),
        ("avg_cost", "${:>14.4f}"),
        ("avg_tool_calls", "{:>15.1f}"),
    ]:
        name = metric.replace("avg_", "Avg ").replace("_", " ").title()
        print(f"{name:<20}", end="")
        for s in run_summaries.values():
            val = s[metric]
            print(f" {fmt.format(val)}", end="")
        print()

    # Per-question quality evolution
    print(f"\n{'─'*70}")
    print("Quality Evolution Per Question:")
    for qid, track in evolution_tracking.items():
        scores = [r["quality"] for r in track["runs"]]
        trend = " -> ".join(f"{s:.0f}" for s in scores)
        delta = scores[-1] - scores[0] if len(scores) > 1 else 0
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {qid}: {trend}  ({arrow}{abs(delta):+.1f})")

    print(f"\n{'='*70}")
    return output


if __name__ == "__main__":
    run_evolution_benchmark()

# KISS vs LangGraph Open Deep Research — Benchmark Suite

Reproducible benchmarks comparing KISS Agent Framework's single-agent approach against LangGraph's multi-stage Open Deep Research pipeline.

## Three Benchmarks

### 1. Real Framework Benchmark (`bench_real.py`) — Recommended

Uses **actual framework code**: KISS `KISSAgent.run()` and LangGraph `StateGraph.compile()`. Requires Python 3.13+.

| Metric | KISS | LangGraph-ODR | Ratio |
|--------|------|---------------|-------|
| Avg Time | 88s | 274s | **3.1x faster** |
| Avg Tokens | 20,793 | 85,449 | **4.1x fewer** |
| Avg Cost | $0.73 | $2.04 | **2.8x cheaper** |
| Avg Tool Calls | 15.0 | 55.7 | **3.7x fewer** |
| Avg Quality | 6.3/10 | **7.3/10** | LG +1.0 pts |
| Questions Passed | 3/3 | 3/3 | Equal |

### 2. API-Simulated Tool Benchmark (`bench_tools.py`)

Simulates both approaches with raw Anthropic API calls. Useful for environments without Python 3.13.

| Metric | KISS | LangGraph-ODR | Ratio |
|--------|------|---------------|-------|
| Avg Time | 97s | 445s | **4.6x faster** |
| Avg Tokens | 14,058 | 88,711 | **6.3x fewer** |
| Avg Cost | $0.45 | $2.11 | **4.7x cheaper** |
| Avg Tool Calls | 12.3 | 60.0 | **4.9x fewer** |
| Avg Quality | 7.0/10 | **8.3/10** | LG +1.3 pts |

### 3. No-Tool Research Benchmark (`bench_harness.py`)

Pure LLM generation without tools. Measures orchestration overhead only.

| Metric | KISS | LangGraph-ODR | Ratio |
|--------|------|---------------|-------|
| Avg Time | 90s | 248s | **2.8x faster** |
| Avg Tokens | 3,486 | 15,421 | **4.4x fewer** |
| Avg Cost | $0.25 | $0.79 | **3.1x cheaper** |

## Quick Start

```bash
# Clone
git clone https://github.com/ksenxx/kiss_ai.git
cd kiss_ai

# Install dependencies (Python 3.13+ required for bench_real.py)
pip install anthropic yfinance langgraph langchain-anthropic duckduckgo-search

# Configure API (choose one):

# Option A: Direct Anthropic
export ANTHROPIC_API_KEY="your-key"

# Option B: Databricks-hosted
export ANTHROPIC_BASE_URL="https://your-workspace.cloud.databricks.com/serving-endpoints/anthropic"
export ANTHROPIC_AUTH_TOKEN="your-databricks-token"
export ANTHROPIC_MODEL="databricks-claude-opus-4-6"

# Run real-framework benchmark (recommended, requires Python 3.13+)
python3 benchmarks/bench_real.py

# Or run API-simulated tool benchmark (any Python 3.10+)
python3 benchmarks/bench_tools.py

# Or run the no-tool research benchmark
python3 benchmarks/bench_harness.py
```

## Tools Available (bench_tools.py)

| Tool | Description |
|------|-------------|
| `yfinance_get_stock_info` | Current price, P/E, market cap, 52-week range, analyst targets |
| `yfinance_get_history` | Historical OHLCV data with summary statistics |
| `web_search` | DuckDuckGo web search (no API key required) |

## Benchmark Design

### KISS Approach (Single-Agent ReAct)
One agent with all tools. Decides what to fetch, calls tools, writes the report. Mirrors how KISS's `RelentlessAgent` works.

### LangGraph-ODR Approach (Multi-Stage Pipeline)
Reproduces the 4-stage pipeline from [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research):
1. **Research Brief** — Decompose question into 3 sub-topics
2. **3 Researchers** — Each sub-agent has independent tool access (up to 6 rounds)
3. **Compress** — Synthesize all findings
4. **Final Report** — Generate comprehensive report

### Quality Scoring
LLM-judged on 4 dimensions: data grounding, analysis depth, recency, actionability. Overall score 1-10.

## Customization

Edit `BENCHMARK_QUESTIONS` in either file:

```python
BENCHMARK_QUESTIONS = [
    {
        "id": "q1",
        "topic": "Your research question here",
        "complexity": "high",
    },
]
```

## Zscaler / Corporate Proxy

`bench_tools.py` disables SSL verification globally for corporate proxy environments. No additional configuration needed.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Option A | Direct Anthropic API key |
| `ANTHROPIC_BASE_URL` | Option B | Proxy/Databricks endpoint URL |
| `ANTHROPIC_AUTH_TOKEN` | Option B | Bearer token for proxy endpoint |
| `ANTHROPIC_MODEL` | No | Model name (default: `claude-opus-4-6`) |

## License

Apache 2.0 — same as KISS Agent Framework.

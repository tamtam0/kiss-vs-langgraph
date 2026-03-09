# Comparing KISS vs LangGraph Deep Research

Empirical benchmarks comparing [KISS Agent Framework](https://github.com/ksenxx/kiss_ai)'s single-agent approach against [LangGraph Open Deep Research](https://github.com/langchain-ai/open_deep_research)'s multi-stage pipeline on real finance research tasks.

## What's Here

### Benchmarks (`benchmarks/`)

| File | Description | Requires |
|------|-------------|----------|
| `bench_real.py` | **Recommended.** Uses actual `KISSAgent.run()` and `StateGraph.compile()` with identical tool functions | Python 3.13+ |
| `bench_evolution.py` | Runs KISS 3x per question, evolving the system prompt between runs via `prompt_refiner_agent` | Python 3.13+ |
| `bench_tools.py` | API-simulated tool benchmark (raw Anthropic API, no framework imports) | Python 3.10+ |
| `bench_harness.py` | No-tool research benchmark (pure LLM generation) | Python 3.10+ |

### Results (`benchmarks/*.json`)

Raw benchmark outputs with per-question metrics.

### Article & Visuals (`docs/`)

LinkedIn article and hero images presenting the findings. Open `docs/article.html` in a browser.

## Quick Start

```bash
# Python 3.13+ required for bench_real.py and bench_evolution.py
pip install kiss-ai anthropic yfinance langgraph langchain-anthropic duckduckgo-search

# Set API key
export ANTHROPIC_API_KEY="your-key"

# Run framework comparison
python benchmarks/bench_real.py

# Run prompt evolution experiment
python benchmarks/bench_evolution.py
```

Supports Databricks-hosted Claude endpoints via `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN`.

## Results Summary

### Experiment 1: KISS vs LangGraph (bench_real.py)

| Metric | KISS | LangGraph-ODR | Ratio |
|--------|------|---------------|-------|
| Avg Time | 88s | 274s | 3.1x faster |
| Avg Tokens | 20,793 | 85,449 | 4.1x fewer |
| Avg Cost | $0.73 | $2.04 | 2.8x cheaper |
| Avg Tool Calls | 15.0 | 55.7 | 3.7x fewer |
| Avg Quality | 6.3/10 | 7.3/10 | LG +1.0 pts |

### Experiment 2: KISS Prompt Evolution (bench_evolution.py)

| Metric | Baseline | Evolution #1 | Evolution #2 |
|--------|----------|-------------|-------------|
| Avg Quality | 6.7 | 6.7 | 6.3 |
| Avg Tool Calls | 13.3 | 20.7 | 21.0 |
| Avg Tokens | 15,634 | 31,559 | 32,114 |
| Avg Cost | $0.55 | $1.10 | $1.12 |

Per-question quality: Q1 7→7→5, Q2 6→6→7, Q3 7→7→7.

## Tools

All benchmarks use the same three tool functions:

| Tool | Source | Description |
|------|--------|-------------|
| `get_stock_info` | yfinance | Current price, P/E, market cap, 52-week range, analyst targets |
| `get_stock_history` | yfinance | Historical OHLCV with summary statistics |
| `web_search` | duckduckgo-search | Web search (no API key required) |

## License

Apache 2.0

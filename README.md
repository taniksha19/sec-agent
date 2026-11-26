# SEC Filings Research Agent

A production-style RAG research agent over SEC filings (10-K, 10-Q, 8-K earnings releases) that answers financial questions with citations, defends against prompt injection in retrieved documents, and runs on Kubernetes with measured latency, quality, and cost.

> Status: Phases 1 (data and baseline RAG), 2 (better retrieval and evals) 3 (agent, API, Go gateway, MCP servers, Qdrant vs. Pinecone) 4 (Langfuse tracing, Prometheus + Grafana, prompt-injection defenses) and 5 (LLM routing between free hosted models, provider fallback) done. **Phase 6 (Kubernetes, load test, CI)** next. See [plan.md](plan.md) for the full build plan.

## Results

66 golden questions (58 answerable, 8 with no answer in the filings), one `make eval` run:

| Retrieval | Recall@1 / @5 / @10 | Accuracy | Faithfulness | Refused when answerable |
| --- | --- | --- | --- | --- |
| Phase 1 baseline: vector top-5 | 0.190 / 0.478 / 0.616 | 0.545 | | 0.431 |
| Vector, contextual chunk headers | 0.362 / 0.543 / 0.703 | 0.621 | 0.919 | 0.362 |
| Hybrid (dense + BM25, RRF) | 0.388 / 0.586 / 0.716 | 0.621 | 0.946 | 0.362 |
| Hybrid + NVIDIA rerank of 50 | 0.728 / 0.841 / 0.858 | 0.833 | 1.000 | 0.138 |
| Same, new table captions | 0.694 / 0.849 / 0.875 | 0.864 | 1.000 | 0.121 |
| Phase 3 agent (plan, corrective retrieval, tools) | 0.703 / 0.853 / 0.957 | 0.939 | 0.930 | 0.017 |
| **Phase 4: same agent with prompt-injection defenses** | **0.703 / 0.853 / 0.948** | **0.970** | **0.931** | **0.000** |

- Biggest gains are on table questions (accuracy 0.444 in Phase 1, 0.833 now) and multi-hop (0.000 to 0.333).
- No invented answers: all 8 unanswerable questions are refused in every setup.
- The Phase 3 agent splits two-company questions and retries weak retrievals: multi-hop accuracy 0.444 to 0.889. It costs about 4 LLM calls per question and faithfulness drops to 0.930 from citation slips on long answers.
- Latency (Phase 4, from Langfuse traces): p50 13.8 s, p95 48.9 s on the NVIDIA free tier; 92.8% of agent time is hosted-LLM calls, 6.5% rerank, 0.5% vector search.
- Prompt injection: 31 attacks planted in real retrieved filings. With no defense 5 succeed; with delimiters, chunk cleaning and tool guardrails 3 do (tool misuse 2/6 to 0/6), at no cost to clean accuracy. The model alone already resisted all wrong-figure, prompt-leak and table attacks; the remaining gaps are in [docs/decisions.md](docs/decisions.md).
- Model routing (Phase 5, free models): sending the planner and grader steps to a small fast model halves agent latency (p50 5.5 s to 2.8 s, p95 31.8 s to 14.6 s) but costs accuracy (0.985 to 0.924 in the same interleaved run): the fast grader stops searching too early. Strong-only stays the default; `LLM_ROUTE=routed` turns routing on. A Gemini fallback covers NVIDIA timeouts and rate limits.
- Second backend: Pinecone serverless gives the same recall as Qdrant on the same chunks and fusion (R@5 0.849 with rerank on both); retrieval p95 80-83 ms vs. 46-50 ms for local Qdrant, mostly the network trip to us-east-1.

All numbers, with how each was measured, are in [docs/results.md](docs/results.md).

## Data

20 companies across 9 sectors ([data/companies.csv](data/companies.csv)): last 3 10-Ks, last 4 10-Qs, and the 8-K earnings releases for the same quarters. 251 filings (60 10-Ks, 80 10-Qs, 111 8-Ks): 538 MB of HTML, about 15M tokens of text, 48,534 chunks (18,471 of them tables). Golden questions are in [data/golden/questions.jsonl](data/golden/questions.jsonl), each with the source filing, page and an exact quote.

## Run locally

Needs Docker, [uv](https://docs.astral.sh/uv/), and API keys in `.env` (copy `.env.example`): an SEC user agent, an NVIDIA API key (answers and reranking), and a Gemini key (eval judge).

```
uv sync
make up                                  # Qdrant, API, Go gateway on :8080, Prometheus :9090, Grafana :3000
make download                            # filings from EDGAR into data/raw/
uv run python -m ingest.parse            # sections and chunks into data/processed/
uv run python -m retrieval.index         # embed and load into Qdrant (~1.5-2 h on a laptop CPU)
uv run python -m agent.baseline "How much did Apple spend on R&D in fiscal 2026?"
make eval                                # vector, hybrid, hybrid+rerank: recall, accuracy, faithfulness
make test
```

### Tools over MCP

The agent's tools are also two MCP servers (stdio), so any MCP client can use them:

- `sec-math`: `calculate`, `compare`. Pure functions, no keys or data.
- `sec-filings`: `search_filings`, `lookup_table` over the indexed filings. Needs Qdrant up and `NVIDIA_API_KEY` in `.env`.

Example client config (Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "sec-math": {"command": "uv", "args": ["--directory", "/path/to/sec-agent", "run", "python", "-m", "mcp_servers.math_server"]},
    "sec-filings": {"command": "uv", "args": ["--directory", "/path/to/sec-agent", "run", "python", "-m", "mcp_servers.filings_server"]}
  }
}
```

## Docs

- [Architecture](docs/architecture.md)
- [Decisions](docs/decisions.md)
- [Results](docs/results.md)

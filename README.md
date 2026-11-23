# SEC Filings Research Agent

A production-style RAG research agent over SEC filings (10-K, 10-Q, 8-K earnings releases) that answers financial questions with citations, defends against prompt injection in retrieved documents, and runs on Kubernetes with measured latency, quality, and cost.

> Status: Phases 1 (data and baseline RAG) and 2 (better retrieval and evals) done. **Phase 3 (agent and API layer)** next. See [plan.md](plan.md) for the full build plan.

## Results

66 golden questions (58 answerable, 8 with no answer in the filings), one `make eval` run:

| Retrieval | Recall@1 / @5 / @10 | Accuracy | Faithfulness | Refused when answerable |
| --- | --- | --- | --- | --- |
| Phase 1 baseline: vector top-5 | 0.190 / 0.478 / 0.616 | 0.545 | | 0.431 |
| Vector, contextual chunk headers | 0.362 / 0.543 / 0.703 | 0.621 | 0.919 | 0.362 |
| Hybrid (dense + BM25, RRF) | 0.388 / 0.586 / 0.716 | 0.621 | 0.946 | 0.362 |
| **Hybrid + NVIDIA rerank of 50** | **0.728 / 0.841 / 0.858** | **0.833** | **1.000** | **0.138** |

- Biggest gains are on table questions (accuracy 0.444 in Phase 1, 0.833 now) and multi-hop (0.000 to 0.333).
- No invented answers: all 8 unanswerable questions are refused in every setup.
- 8 of the 11 remaining failures are retrieval misses that end in a refusal, mostly two-company questions. Phase 3 targets these with a corrective retrieval loop.

All numbers, with how each was measured, are in [docs/results.md](docs/results.md).

## Data

20 companies across 7 sectors ([data/companies.csv](data/companies.csv)): last 3 10-Ks, last 4 10-Qs, and the 8-K earnings releases for the same quarters. 243 filings, 48,534 chunks. Golden questions are in [data/golden/questions.jsonl](data/golden/questions.jsonl), each with the source filing, page and an exact quote.

## Run locally

Needs Docker, [uv](https://docs.astral.sh/uv/), and API keys in `.env` (copy `.env.example`): an SEC user agent, an NVIDIA API key (answers and reranking), and a Gemini key (eval judge).

```
uv sync
make up                                  # Qdrant, API, and Go gateway on 127.0.0.1:8080
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

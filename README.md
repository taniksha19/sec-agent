# SEC Filings Research Agent

A production-style RAG research agent over SEC filings (10-K, 10-Q, 8-K earnings releases) that answers financial questions with citations, defends against prompt injection in retrieved documents, and runs on Kubernetes with measured latency, quality, and cost.

> Status: **Phase 1 (Data and baseline RAG)** in progress. See [plan.md](plan.md) for the full build plan.

## Results

Phase 1 baseline (vector search top-5, no reranking), on 66 golden questions:

| Metric | Value |
| --- | --- |
| Recall@1 / @5 / @10 | 0.190 / 0.478 / 0.616 |
| Answer accuracy (all) | 0.545 |
| Accuracy on unanswerable questions (correct refusal) | 1.000 |
| Multi-hop accuracy | 0.000 |

All numbers, with how each was measured, are in [docs/results.md](docs/results.md).

## Data

20 companies across 7 sectors ([data/companies.csv](data/companies.csv)): last 3 10-Ks, last 4 10-Qs, and the 8-K earnings releases for the same quarters. 243 filings, 47,768 chunks. Golden questions are in [data/golden/questions.jsonl](data/golden/questions.jsonl), each with the source filing, page and an exact quote.

## Run locally

Needs Docker, [uv](https://docs.astral.sh/uv/), and API keys in `.env` (copy `.env.example`): an SEC user agent, an NVIDIA API key (answers), and a Gemini key (eval judge).

```
uv sync
make up                                  # Qdrant
make download                            # filings from EDGAR into data/raw/
uv run python -m ingest.parse            # sections and chunks into data/processed/
uv run python -m retrieval.index         # embed and load into Qdrant (~1.5-2 h on a laptop CPU)
uv run python -m agent.baseline "How much did Apple spend on R&D in fiscal 2026?"
make eval                                # recall and accuracy on the golden set
make test
```

## Docs

- [Architecture](docs/architecture.md)
- [Decisions](docs/decisions.md)
- [Results](docs/results.md)

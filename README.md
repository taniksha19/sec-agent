# SEC Filings Research Agent

A production-style RAG research agent over SEC filings (10-K, 10-Q, earnings transcripts) that answers financial questions with citations, defends against prompt injection in retrieved documents, and runs on Kubernetes with measured latency, quality, and cost.

> Status: **Phase 1 (Data and baseline RAG)** in progress. See [plan.md](plan.md) for the full build plan.

## Results

See [docs/results.md](docs/results.md). Numbers are added as each phase is measured.

## Run locally

Coming in Phase 1 (`docker compose up`, `make eval`).

## Docs

- [Architecture](docs/architecture.md)
- [Decisions](docs/decisions.md)
- [Results](docs/results.md)

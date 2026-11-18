# Architecture

Request flow:

1. Client -> **Go gateway** (API-key auth, rate limiting, request logging, routing)
2. Gateway -> **FastAPI** -> **LangGraph agent**
3. Agent calls **tools via MCP servers** (calculator, table lookup, cross-company comparison)
4. Agent calls **retrieval**: BM25 + vector search (`VectorStore` interface) -> fusion -> cross-encoder rerank
5. Retrieved chunks pass through the **security sanitizer** (untrusted input) before reaching the agent
6. Agent calls the **LLM gateway**, which routes each question to a cheap or a strong hosted model, with cost accounting and fallback

Offline: **ingestion** (EDGAR download -> parse -> section chunking -> embed) writes to the vector store.

Cross-cutting: **observability** (Langfuse traces, Prometheus metrics, Grafana dashboards) wraps every call.

## VectorStore interface

`upsert(chunks)`, `search(query_vector, k, filters)`, `delete(ids)`. Qdrant or pgvector is primary; Pinecone is the optional second backend. Same embedding model and golden set across backends.

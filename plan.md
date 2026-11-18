# SEC Filings Research Agent: Build Plan for Claude Code

Nov 13, 2026 · @Taniksha Dasharath Datar

## Project brief

Build a production-style RAG research agent over SEC filings (10-K, 10-Q, earnings transcripts) that answers financial questions with citations, defends against prompt injection in retrieved documents, and runs on Kubernetes with measured latency, quality, and cost.

**Why this project exists:** it is a portfolio piece for AI Engineer roles. Every layer must produce a measured number that can go on a resume, and the repo must read like a real system, not a tutorial.

**Goals**

- Hybrid retrieval (BM25 + vectors) with a reranker, benchmarked against a vector-only baseline.
- A LangGraph agent with tools exposed through MCP servers, behind a Go or TypeScript API gateway.
- A prompt-injection sanitizer with before/after attack success rates.
- A LoRA-fine-tuned small model served with vLLM, routed through an LLM gateway alongside a hosted model.
- Langfuse tracing, Grafana dashboards, and evals that gate merges in CI.
- Kubernetes deployment with Terraform, autoscaling, and a load test.

**Non-goals**

- No custom frontend beyond a minimal chat UI. Spend the time on the backend and infra.
- No multi-tenant billing, user accounts beyond a simple API key, or fancy UI polish.
- No scraping beyond SEC EDGAR.

**How Claude Code should work on this**

1. Read this whole document first, then work one phase at a time in order. Do not start a phase until the previous phase's acceptance criteria pass.
2. At the end of each phase, commit, update the README, and write measured results into `docs/results.md` (table of metric, baseline, final, how it was measured). Never invent numbers; if something was not measured, leave it blank.
3. Prefer small, boring, readable code. No speculative abstractions, no unused config options. Add a short comment only where the reason for a choice is not obvious.
4. Write tests for parsing, retrieval, and tool logic. Keep every eval reproducible from one command (`make eval`).
5. Keep secrets in `.env` (git-ignored) with a committed `.env.example`. Pin dependency versions.
6. Local first: everything must run on a laptop with `docker compose up` before any cloud step. Cloud resources are created only in Phase 6 and Phase 5's GPU session, and torn down after.
7. When a decision has real tradeoffs (e.g., a chunking strategy), try the options, record the results in `docs/decisions.md`, and keep the winner. This log feeds the README's "what broke and how I fixed it" section.

## Architecture

A request enters through the Go gateway, and the LangGraph agent then calls tools, retrieval, and the LLM gateway; retrieved text is sanitized before the agent uses it.

&#91;embedded content: architecture · 9 components\]

The shaded box is the security differentiator: retrieved filings are treated as untrusted input. Ingestion runs offline and writes to the vector store; observability wraps every call.

**VectorStore interface.** Retrieval talks to one interface, `VectorStore`, with `upsert(chunks)`, `search(query_vector, k, filters)`, and `delete(ids)`. Qdrant or pgvector is the primary backend and Pinecone is the optional second one. Use the same embedding model and the same golden set across backends so latency, recall, and cost comparisons are fair.

## Tech stack and repo structure

Python handles the AI logic, Go handles the gateway, and everything runs in containers.

| Layer | Choice | Notes |
| --- | --- | --- |
| Language (core) | Python 3.11+ | uv or pip-tools for pinned deps |
| API (core) | FastAPI + Pydantic | Typed request/response models |
| Gateway | Go (or TypeScript) | Auth, rate limiting, request logging, routing |
| Agent | LangGraph | Tools exposed via MCP servers |
| Vector store | Qdrant or pgvector (primary), Pinecone (optional second backend) | Behind one `VectorStore` interface |
| Keyword search | BM25 (rank-bm25 or Postgres full-text) | Hybrid with vectors |
| Reranker | Cross-encoder (e.g., bge-reranker) | Compare against no reranker |
| Embeddings | Open model (e.g., bge or e5) | Same model across all backends for fair comparison |
| Fine-tuning | HuggingFace + PEFT (LoRA) | Small model (Llama or Qwen, 7B or smaller) |
| Serving | vLLM | Quantized small model, GPU only for benchmarks |
| Observability | Langfuse, Prometheus, Grafana | Tracing on every LLM call |
| Infra | Docker Compose (local), kind (local K8s), GKE or EKS (final) | Terraform for cloud |
| CI/CD | GitHub Actions | Lint, tests, evals gate merges |
| Load testing | k6 or Locust | Report p50/p95 and req/s |
| Data | SEC EDGAR | Free; respect their rate limits and set a User-Agent |

**Suggested repo layout**

```text
sec-agent/
  README.md
  Makefile                # make up, make eval, make loadtest
  docker-compose.yml
  .env.example
  docs/
    architecture.md
    decisions.md          # tradeoffs tried and what won
    results.md            # every measured number, with method
  data/
    raw/                  # downloaded filings (git-ignored)
    processed/
    golden/               # golden question set (committed)
  src/
    ingest/               # EDGAR download, parsing, chunking
    retrieval/            # VectorStore interface, backends, hybrid, rerank
    agent/                # LangGraph graph, prompts
    tools/                # calculator, table lookup, comparison
    mcp_servers/          # MCP server wrappers around tools
    security/             # sanitizer, attack corpus
    llm_gateway/          # model routing by cost and latency
    api/                  # FastAPI app
  gateway/                # Go service
  evals/                  # eval harness, metrics, CI entrypoint
  finetune/               # LoRA training scripts and configs
  infra/
    k8s/                  # manifests or Helm chart
    terraform/
  loadtest/
  tests/
```

## Phases, tasks, and acceptance criteria

Six phases, about one week each. Do them in order; each ends with a commit, a README update, and measured numbers in `docs/results.md`.

### Phase 1: Data and baseline RAG

- Download 10-K and 10-Q filings and earnings transcripts for about 20 companies from EDGAR (set a User-Agent, respect rate limits).
- Parse text and tables; chunk by filing section (Item 1A, Item 7, and so on) and keep metadata: company, ticker, form type, period, section, page.
- Embed chunks and load them into the primary vector store behind a `VectorStore` interface.
- Build a baseline: vector search, top-k, answer generation with citations to filing and section.
- Write the first 50+ golden questions with checkable answers (see Evals).

*Acceptance:* `docker compose up` starts the stack; a question returns a cited answer; baseline recall@5 and answer accuracy are recorded.

### Phase 2: Better retrieval and evals

- Add BM25 and hybrid fusion (reciprocal rank fusion is fine), then a cross-encoder reranker.
- Add table-aware extraction so numeric questions hit the right table rows.
- Build the eval harness: retrieval metrics plus an LLM-as-judge for answer correctness and citation faithfulness.
- Compare vector-only, hybrid, and hybrid + rerank. Record each result and note which question types improved.

*Acceptance:* `make eval` prints recall@k, answer accuracy, and faithfulness for all three configurations; hybrid + rerank beats baseline, or `docs/decisions.md` explains why not.

### Phase 3: Agent and API layer

- Build the LangGraph agent with tools: calculator, table lookup, and cross-company comparison.
- Wrap the tools as MCP servers so the same tools work from other MCP clients.
- Expose the agent through FastAPI, then put the Go (or TypeScript) gateway in front with API-key auth, rate limiting, and request logging.
- Add a second `VectorStore` backend (Pinecone, or whichever of Qdrant/pgvector is not primary) and run the golden set on both.

*Acceptance:* multi-step questions (for example, "compare R&D spend growth for A and B over 3 years") are answered via tool calls; gateway rejects bad keys and enforces limits; both backends pass the golden set with recall and latency recorded.

### Phase 4: Security layer and observability

- Build the prompt-injection corpus and sanitizer (details in the Security section).
- Add Langfuse tracing to every retrieval, tool, and LLM call.
- Add Prometheus metrics and a Grafana dashboard: p50/p95 latency, token cost, error rate, retrieval hit rate.

*Acceptance:* attack success rate measured with and without the sanitizer; every request visible as a Langfuse trace; the dashboard shows live metrics. This is the first demo-ready checkpoint.

### Phase 5: Fine-tuning and serving

- Build a financial Q&A training set from the filings (generate with a strong model, then filter and spot-check by hand). Hold out a test split that never touches training.
- LoRA fine-tune a small model (Llama or Qwen, 7B or smaller). Rent a GPU only for this session and tear it down after.
- Serve the base and tuned models with vLLM; compare accuracy, latency, and throughput.
- Build the LLM gateway that routes between the local tuned model and a hosted model based on cost and latency.

*Acceptance:* before/after accuracy on the held-out set; vLLM throughput and latency recorded; routing demonstrably lowers cost per query on the golden set at similar accuracy.

### Phase 6: Deploy and stress test

- Write Terraform and Kubernetes manifests (or a Helm chart) for the gateway, API, agent, vector store, and observability. Develop on kind first.
- Add autoscaling (HPA) on CPU or request rate.
- Add the GitHub Actions pipeline: lint, unit tests, and the eval suite, which blocks merges when quality drops below a threshold.
- Deploy to GKE or EKS only for the final run, load test with k6 or Locust, record results, then tear the cluster down.
- Finish the README: architecture diagram, results table, "what broke and how I fixed it," cost notes, and how to run everything.

*Acceptance:* the load test reports p50/p95 latency and requests per second at a stated pod count; CI fails on a deliberate quality regression; the README lets a stranger run the project locally.

## Evals

Evals are the spine of the project: every change to retrieval, prompts, or models must show up as a number, and CI blocks merges that make the numbers worse.

**Golden set** (`data/golden/`, committed as JSONL, 50+ questions, grow to 100)

Each record: `id`, `question`, `type`, `expected_answer`, `source_company`, `source_filing`, `source_section`. Mix question types so you can see where each retrieval change helps:

- Fact lookup ("What was Company X's total revenue in fiscal 2024?")
- Table and numeric ("What was R&D spend as a percent of revenue?")
- Multi-hop and comparison ("Which of A or B grew operating margin faster over 3 years?")
- Risk-factor and narrative questions ("What supply-chain risks does X disclose?")
- Unanswerable questions, where the correct behavior is to say the filings do not contain the answer

Write answers by reading the actual filings, not by asking a model. Spot-check any model-assisted entries by hand.

**Metrics**

| Metric | What it measures | How |
| --- | --- | --- |
| Recall@k (k = 1, 5, 10) | Did retrieval surface the source chunk? | Match retrieved chunks to the golden source section |
| Answer accuracy | Is the final answer correct? | Exact or tolerance match for numbers; LLM-as-judge for text |
| Citation faithfulness | Is every claim supported by a cited chunk? | LLM-as-judge on answer vs. cited text |
| Refusal rate on unanswerable | Does it avoid making things up? | Share of unanswerable questions correctly declined |
| Latency (p50, p95) | Speed | Wall-clock per question |
| Cost per query | Tokens times price | From Langfuse traces |

**Rules for the judge:** use a different model from the one answering when possible, fix the judge prompt in version control, and hand-check a sample of judge decisions so you know its error rate.

**CI gating:** `make eval` must run on a small fixed subset (about 20 questions) in GitHub Actions and fail if recall@5 or answer accuracy falls more than an agreed margin below the committed baseline in `evals/baseline.json`. The full set runs on demand and before each phase is marked done.

## Security layer

The differentiator: most RAG demos ignore the fact that retrieved text is untrusted input. This layer measures how often injected instructions inside documents hijack the agent, then shows how much a sanitizer reduces that.

**Threat model.** The user is trusted. The filings and any other retrieved text are not. An attacker who can influence a document (a poisoned filing excerpt, a transcript, an uploaded file) tries to make the agent ignore its instructions, leak its system prompt or tool configuration, call tools it should not, or give a false answer.

**Attack corpus** (`src/security/attacks/`, committed as JSONL)

- Build 50+ test cases by inserting benign-looking injected text into real filing chunks. Cover categories: instruction override, system-prompt extraction, tool misuse (for example, forcing a tool call with attacker-chosen arguments), answer manipulation (state a wrong figure), and instructions hidden in tables or footnotes.
- Keep the corpus inside a controlled test harness against your own agent. Do not target any third-party system.
- Each case records the category, the clean chunk it modifies, the injected payload, and a pass/fail check (did the agent follow the injection?).

**Defenses to build and compare**

1. **Baseline:** no defense.
2. **Prompt hardening:** clear separation of instructions and retrieved content, with the retrieved text explicitly marked as data.
3. **Sanitizer:** a classifier or small model that flags and strips instruction-like content from chunks before they reach the agent. This can reuse ideas from the sanitizer-model work in the Agentic AI course project.
4. **Tool guardrails:** allow-list tool arguments and require that tool calls trace back to the user's request, not retrieved text.

**Measurement**

| Configuration | Attack success rate | Answer accuracy on clean golden set | Added latency |
| --- | --- | --- | --- |
| No defense | measure | measure | none |
| Prompt hardening | measure | measure | measure |
| Sanitizer | measure | measure | measure |
| Sanitizer + tool guardrails | measure | measure | measure |

The tradeoff matters: report false positives (clean chunks wrongly stripped) and the accuracy cost of each defense, not only the attack success rate drop. Write the results and failure cases into `docs/results.md` and `docs/decisions.md`.

## Infra, budget, and cost controls

Target total spend is roughly $30-60 (under $20 if the managed-cluster step is skipped); verify current cloud and API pricing before each paid step, since rates and free tiers change.

| Item | Free option | Paid only when |
| --- | --- | --- |
| Data | SEC EDGAR | never |
| Vector DB | Qdrant or pgvector in Docker | Pinecone backend test (check free tier first) |
| Observability | Self-hosted Langfuse, Prometheus, Grafana | never |
| CI | GitHub Actions free tier | never |
| Kubernetes | kind or minikube locally | Final GKE or EKS deploy and load test |
| LLM calls | Groq and Gemini free tiers where limits allow | Judge runs and training-data generation at scale |
| GPU | Colab free tier if it fits | LoRA fine-tune and vLLM benchmarks (a few hours, rented) |

**Cost controls Claude Code must follow**

- Develop and test everything locally on Docker Compose and kind. Touch the cloud only in Phase 5 (GPU session) and Phase 6 (final deploy).
- Before creating any paid resource, state what it is, the hourly or daily cost, and the planned teardown, then wait for the user to confirm.
- Tag all cloud resources with `project=sec-agent` so they are easy to find and delete.
- Provide a `make teardown` target (Terraform destroy plus GPU instance shutdown) and run it at the end of every paid session.
- Remind the user to set a billing budget alert (suggested $20-30) before the first paid step.
- Cache embeddings, LLM judge outputs, and generated training data on disk so reruns do not repeat paid calls.
- Never commit API keys, cloud credentials, or `.env` files.

## Resume metrics and README requirements

Every resume bullet will be built from a number in `docs/results.md`, so Claude Code should record these as each phase finishes. Leave a cell empty if it was not measured.

| Phase | Metric to record | Baseline | Final |
| --- | --- | --- | --- |
| 2 | Recall@5, answer accuracy (vector-only vs. hybrid + rerank) |  |  |
| 3 | Recall and p95 latency, backend A vs. backend B |  |  |
| 4 | Attack success rate with and without sanitizer; false-positive rate |  |  |
| 4 | Added latency from sanitizer |  |  |
| 5 | Held-out accuracy, base vs. LoRA-tuned model |  |  |
| 5 | vLLM throughput (tokens/s) and p95 latency |  |  |
| 5 | Cost per query with and without model routing |  |  |
| 6 | Load test: requests/s, p50/p95 latency, pod count |  |  |
| 6 | CI: merge blocked on a deliberate regression (yes/no, evidence) |  |  |

**README must include**

1. One-paragraph summary and an architecture diagram.
2. A results table pulled from `docs/results.md`.
3. A "What broke and how I fixed it" section drawn from `docs/decisions.md`, with at least three real examples (these become interview stories).
4. How to run locally in under 10 minutes (`docker compose up`, `make eval`).
5. Cost notes and how to tear down cloud resources.
6. Known limitations and what to build next.

**Definition of done:** a stranger can clone the repo, run the stack locally, reproduce the eval numbers, and see the load-test results; every claim in the README has a number and a method behind it.

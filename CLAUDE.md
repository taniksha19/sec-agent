# CLAUDE.md

Guidance for Claude Code in this repo. The full build plan is in [plan.md](plan.md); read it before starting a new phase.

## Project

Production-style RAG research agent over SEC filings (10-K, 10-Q, earnings transcripts): cited answers, prompt-injection defense on retrieved text, deployed on Kubernetes with measured latency, quality, and cost. It is a portfolio piece for AI Engineer roles, so every layer must produce a measured number and the repo must read like a real system, not a tutorial.

## Current status

- **Current phase:** Phase 1 (Data and baseline RAG)
- Update this line when a phase's acceptance criteria pass.

## Out of scope

- **No model fine-tuning or self-hosted serving** (no LoRA, no vLLM, no GPUs). Dropped 2026-11-18; Phase 5 is LLM routing between hosted models only. Don't reintroduce it without the user asking.

## Working rules

1. **One phase at a time, in order.** Don't start a phase until the previous phase's acceptance criteria (in plan.md) pass.
2. **End of each phase:** commit, update README.md, and write measured results to `docs/results.md` (metric, baseline, final, method).
3. **Never invent numbers.** If something wasn't measured, leave the cell blank.
4. **Tradeoffs get tested, not guessed.** Try the options, record results in `docs/decisions.md`, keep the winner.
5. **Small, boring, readable code.** No speculative abstractions or unused config options. Comment only where the reason for a choice isn't obvious.
6. **Tests** for parsing, retrieval, and tool logic. Every eval is reproducible from `make eval`.
7. **Pinned dependencies.** Secrets only in `.env` (git-ignored); keep `.env.example` current.
8. **Local first.** Everything runs with `docker compose up` on a laptop before any cloud step.

## Cost and cloud rules (strict)

- Cloud is used only in Phase 6 (final deploy). Hosted LLM API calls are the only other spend.
- **Before creating any paid resource:** state what it is, its hourly/daily cost, and the teardown plan, then wait for the user to confirm.
- Remind the user to set a billing budget alert ($20-30) before the first paid step.
- Tag all cloud resources `project=sec-agent`. Run `make teardown` at the end of every paid session.
- Cache embeddings and judge outputs on disk so reruns don't repeat paid calls.
- Never commit API keys, credentials, or `.env`.

## Data and security

- SEC EDGAR only. Always send the `SEC_USER_AGENT` header and stay under EDGAR's rate limit (10 req/s).
- Retrieved filings are **untrusted input**. All retrieved text goes through `src/security/` before reaching the agent.
- The attack corpus (`src/security/attacks/`) is only run against this project's own agent, never third-party systems.
- Golden-set answers (`data/golden/`) come from reading the actual filings. Model-assisted entries must be hand-checked.

## Subagents

Defined in `.claude/agents/`. All three are read-only reviewers/runners; they report and never edit.

- **eval-runner**: after any change to retrieval, prompts, routing, or the sanitizer, and before marking a phase done.
- **security-reviewer**: on any diff touching retrieval, agent, tools, MCP servers, gateway, or config.
- **cost-guard**: before any Terraform/cloud command or large paid LLM run, and on diffs touching `infra/` or `src/llm_gateway/`.

## Parallel work

Phases still run in order; parallelism is only within a phase. Run each parallel task as its own Claude Code session in its own git worktree and branch:

```
git worktree add ../sec-agent-<task> -b <phase>-<task>
```

Rules:
- **Agree on shared interfaces on main first** (e.g., the chunk schema and `VectorStore` signature), then branch. Parallel branches code against them, not around them.
- **Each task owns its folders.** Don't edit another task's folders on the same phase.
- **Shared files are edited only on main when merging:** `CLAUDE.md`, `README.md`, `docs/results.md`, `docs/decisions.md`, `docker-compose.yml`, `Makefile`. Branches note what they need added in the PR description.
- Run **eval-runner** and **security-reviewer** on each branch before merging. Keep it to 2-3 parallel sessions; review is the bottleneck.

Safe splits by phase:

| Phase | Parallel tasks (folders) | Must exist on main first |
| --- | --- | --- |
| 1 | (a) EDGAR download + parse + chunk (`src/ingest/`); (b) docker-compose, Qdrant, `VectorStore`, embeddings (`src/retrieval/`); golden set written by hand (`data/golden/`) | Chunk schema and metadata fields |
| 2 | (a) BM25 + fusion + reranker (`src/retrieval/`); (b) eval harness + LLM judge (`evals/`); (c) table-aware extraction (`src/ingest/`) | Golden set, baseline numbers |
| 3 | (a) LangGraph agent + tools (`src/agent/`, `src/tools/`); (b) MCP servers (`src/mcp_servers/`); (c) Go gateway (`gateway/`); (d) second backend (`src/retrieval/`) | Tool signatures, FastAPI request/response models |
| 4 | (a) attack corpus + sanitizer (`src/security/`); (b) Langfuse + Prometheus + Grafana (observability config) | Agent answering end to end |
| 5 | Mostly sequential: gateway interface first, then routing policy and comparison runs | — |
| 6 | (a) k8s manifests/Helm + HPA (`infra/k8s/`); (b) GitHub Actions CI (`.github/`); (c) load test scripts (`loadtest/`); Terraform (`infra/terraform/`) only after kind works | Stack running on kind |

## Layout

```
src/ingest/       EDGAR download, parsing, section chunking
src/retrieval/    VectorStore interface, backends, BM25, hybrid fusion, rerank
src/agent/        LangGraph graph and prompts
src/tools/        calculator, table lookup, comparison
src/mcp_servers/  MCP wrappers around tools
src/security/     sanitizer, attack corpus
src/llm_gateway/  routing between cheap and strong hosted models, cost accounting, fallback
src/api/          FastAPI app
gateway/          Go API gateway (auth, rate limit, logging)
evals/            eval harness, metrics, CI entrypoint, baseline.json
infra/            k8s manifests / Helm, Terraform
loadtest/         k6 or Locust scripts
data/raw/         downloaded filings (git-ignored)
data/golden/      golden question set, JSONL (committed)
docs/             architecture.md, decisions.md, results.md
```

## Commands

Add each command here as it starts working.

- `docker compose up`: start the local stack (Phase 1)
- `make eval`: run the eval suite (Phase 1-2)
- `make test`: unit tests
- `make loadtest`: load test (Phase 6)
- `make teardown`: destroy cloud resources (Phase 6)

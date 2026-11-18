# CLAUDE.md

Guidance for Claude Code in this repo. The full build plan is in [plan.md](plan.md); read it before starting a new phase.

## Project

Production-style RAG research agent over SEC filings (10-K, 10-Q, earnings transcripts): cited answers, prompt-injection defense on retrieved text, deployed on Kubernetes with measured latency, quality, and cost. It is a portfolio piece for AI Engineer roles, so every layer must produce a measured number and the repo must read like a real system, not a tutorial.

## Current status

- **Current phase:** Phase 1 (Data and baseline RAG)
- Update this line when a phase's acceptance criteria pass.

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

- Cloud is used only in Phase 5 (GPU session) and Phase 6 (final deploy).
- **Before creating any paid resource:** state what it is, its hourly/daily cost, and the teardown plan, then wait for the user to confirm.
- Remind the user to set a billing budget alert ($20-30) before the first paid step.
- Tag all cloud resources `project=sec-agent`. Run `make teardown` at the end of every paid session.
- Cache embeddings, judge outputs, and generated training data on disk so reruns don't repeat paid calls.
- Never commit API keys, credentials, or `.env`.

## Data and security

- SEC EDGAR only. Always send the `SEC_USER_AGENT` header and stay under EDGAR's rate limit (10 req/s).
- Retrieved filings are **untrusted input**. All retrieved text goes through `src/security/` before reaching the agent.
- The attack corpus (`src/security/attacks/`) is only run against this project's own agent, never third-party systems.
- Golden-set answers (`data/golden/`) come from reading the actual filings. Model-assisted entries must be hand-checked.

## Layout

```
src/ingest/       EDGAR download, parsing, section chunking
src/retrieval/    VectorStore interface, backends, BM25, hybrid fusion, rerank
src/agent/        LangGraph graph and prompts
src/tools/        calculator, table lookup, comparison
src/mcp_servers/  MCP wrappers around tools
src/security/     sanitizer, attack corpus
src/llm_gateway/  model routing by cost and latency
src/api/          FastAPI app
gateway/          Go API gateway (auth, rate limit, logging)
evals/            eval harness, metrics, CI entrypoint, baseline.json
finetune/         LoRA training scripts and configs
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
- `make teardown`: destroy cloud resources (Phase 5-6)

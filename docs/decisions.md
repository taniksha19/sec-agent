# Decisions

Tradeoffs tried, what was measured, and what won. This log feeds the README's "What broke and how I fixed it" section.

Template:

## YYYY-MM-DD: <decision>
- **Options tried:**
- **How measured:**
- **Result:**
- **Chosen:** and why

## 2026-11-18: Drop LoRA fine-tuning and vLLM serving
- **Options considered:** (a) full Phase 5 as planned: LoRA-tune a 7B-or-smaller model, serve with vLLM, route between it and a hosted model; (b) drop fine-tuning and serving, keep the LLM gateway routing between a cheap and a strong hosted model.
- **How measured:** not measured; a scope decision made before Phase 1.
- **Result:** (a) needs rented GPUs, a generated and hand-checked training set, and about a week of work. Those are the costliest, least distinctive parts of the project.
- **Chosen:** (b). It keeps the cost-per-query routing result, removes all GPU spend, and frees time for the security layer, evals, and infra.

## 2026-11-18: Company set and filing scope
- **Options considered:** 5-10 companies (easy to hand-label, but retrieval is too easy to show differences between configurations); about 20 in sector peer groups; 50+ (too many golden answers to write by hand).
- **How measured:** not measured; a scope decision made before Phase 1.
- **Chosen:** 20 companies in 8 sector groups (`data/companies.csv`). Peer groups enable comparison questions; there is a mix of fiscal-year ends (AAPL Sep, MSFT Jun, NVDA/WMT/TGT Jan) and hard cases for parsing (bank statements, BA's negative equity). CIKs and fiscal-year ends are read from EDGAR at download time, not hardcoded.
- **Filing scope:** the last 3 annual 10-Ks and last 4 10-Qs per company, plus 8-K earnings press releases (Exhibit 99.1) for the same quarters.
- **Transcripts replaced:** earnings-call transcripts are not filed on EDGAR, and the plan excludes other sources, so 8-K earnings releases stand in for them.

## 2026-11-18: Embedding model and Qdrant version
- **Options considered:** bge-small-en-v1.5 via fastembed (ONNX, CPU) vs. sentence-transformers (needs torch, a much larger install).
- **How measured:** not measured yet; picked for laptop-friendliness. Revisit in Phase 2 if recall is the bottleneck.
- **Chosen:** bge-small-en-v1.5 (384 dims) through fastembed. fastembed 0.9.1's `query_embed` does not add bge's query instruction for this model, so `Embedder.embed_query` adds it itself. Model files and document vectors are cached under `data/cache/` (on D:, git-ignored).
- **Qdrant:** server pinned to v1.20.0 to match qdrant-client 1.20.0.

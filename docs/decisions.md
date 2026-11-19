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

## 2026-11-19: Qdrant primary, Pinecone for comparison only
- **Chosen:** Qdrant for local dev, CI, and the Phase 6 Kubernetes deploy. Pinecone added in Phase 3 behind `VectorStore`, used only to compare recall and latency on the golden set.
- **Why:** Qdrant is free, runs locally, keeps the deploy self-contained (one teardown), and needs no account for people who clone the repo.
- **Optional later:** point the deployed API at Pinecone for a short load test to compare self-hosted vs. managed under load.

## 2026-11-19: Section parsing for non-standard filing layouts
- **Problem:** Several large filers don't use clean "Item N." headings. Intel uses plain titles; BofA's 10-Q lists "Part I/Part II" right before MD&A; JPMorgan's 10-K "Item 7/8" are one-line stubs pointing to an annual-report section after Item 15; Exxon's history spans two CIKs.
- **Tried:** (1) Item headings only, keeping each heading's last occurrence to skip the table of contents: 3 companies broken. (2) Added a whole-line title fallback, stub detection (<1500 chars, then use the title found later), and title-based Part I/II for 10-Q items: Intel, BofA, JPM 10-Ks fixed.
- **Measured:** `python -m ingest.check_sections` flags 23 of 140 10-K/10-Q filings (117 look clean). Flagged: MSFT (7), F (6), JPM 10-Q MD&A (4), CVX Item 7 (3), PEP Item 8 (3). Flags are suspicions, not each confirmed. All text is still chunked; only section labels are affected. 8-Ks are unaffected.
- **Chosen:** accept for Phase 1. Avoid flagged sections when writing golden questions. Revisit in Phase 2 alongside table-aware extraction (option: compare against an open-source parser such as edgartools).

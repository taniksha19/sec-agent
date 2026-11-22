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

## 2026-11-20: Embedding throughput on CPU
- **Measured:** embedding and loading all 47,768 chunks took 362 min (~2.2 chunks/s) with bge-small via fastembed on a Ryzen 5 5500U laptop CPU. A local Qwen benchmark ran concurrently for part of it, so this is a pessimistic number.
- **Impact:** vectors are cached on disk (`data/cache/embeddings/`), so re-indexing unchanged chunks is fast. But any chunking change in Phase 2 re-embeds everything. Run chunk-size experiments on a subset of companies first, and only re-embed the full corpus for the winner.

## 2026-11-20: Answer LLM for the baseline
- **Options tested:** NVIDIA-hosted Nemotron 3.5 Lightning (30B-A3B), Nemotron 3.5 Super (120B-A12B), Nemotron 3 Ultra (550B-A55B), on the same top-5 RAG prompts. Local Qwen 3.5 4B (Ollama) was ruled out earlier at ~140 s per answer on CPU.
- **Measured (spot check, 2 questions, not an eval):** Lightning ~1 s but invented an NVIDIA R&D figure ("$10.8 billion") that was in none of the 5 sources. Super ~2 s and correctly refused. Ultra ~25 s per answer.
- **Thinking:** Nemotron 3.5 thinks by default; on one RAG prompt it used the whole 1500-token budget with no reply left. Turned off via `chat_template_kwargs.enable_thinking=false`.
- **Chosen:** Super, thinking off. Lightning is the cheap-model candidate for Phase 5 routing.

## 2026-11-22: Eval judge and recall matching
- **Judge:** Gemini 3.5 Flash-Lite through Google's OpenAI-compatible endpoint. A different provider from the answer model (Nemotron), so the judge is not grading its own style. Flash-Lite answered a test prompt in 0.7 s vs 7.7 s for Gemini 3.5 Flash. The free tier returned 429s after a burst of ~20 calls, so judge calls are spaced 4.5 s apart. Judge replies are cached on disk.
- **Judge prompt fix (measured):** the first full run marked a correct Coca-Cola 2025 answer wrong because "2025 has not occurred yet". The prompt now says the reference is ground truth and not to use the judge's own sense of dates.
- **Recall matching:** matching retrieved chunks to the single golden chunk (or its exact quote) undercounted. The same figure is in the 10-K table ("94,827"), its MD&A ("$94.83 billion"), the 10-Q, and the earnings release ("$215.9 billion"), and the model answered correctly from them. A chunk now counts if it is from the same company and contains the figures the reference uses, compared by value with rounding tolerance. Rounded figures under $10B ("2.2 billion") are not matched, being too common. Narrative sources match on key terms. Recall@5 on the same run went from 0.315 (exact quote) to 0.478 (value match).
- **Unanswerable questions** are scored without the judge: correct only if the answer contains the refusal string.
- **Injection exposure in the eval (security review, for Phase 4):** model answers are written from untrusted filing text and then graded by the judge. The answer is wrapped in `<answer_to_grade>` tags and the judge is told to ignore instructions inside it. Unanswerable questions now count as correct only on a bare refusal, so a made-up claim plus the refusal sentence fails. The eval calls `store.search` and `BaselineRAG.generate` directly (it needs top-10 for recall), so the Phase 4 sanitizer must sit in `generate`/`build_prompt` or the store, not only in `retrieve()`, or the eval would measure the unsanitized path.

## 2026-11-22: Golden set checked by machine, not yet by hand
- **Why:** CLAUDE.md asks for model-assisted golden entries to be hand-checked. The user asked for a machine recheck instead, so all 66 entries are marked `"verified": "machine"`. Entries become `"human"` when hand-checked; a hand spot-check of 5-10 is still recommended.
- **Checks, all independent of the project's parser:**
  1. Every evidence quote found in the original EDGAR HTML (24 text quotes; every table row located in a raw `<tr>`).
  2. Column check on every table figure: the figure's column matched to the nearest header row with years. All 34 table sources sit under the year the question asks about (25 needed a manual read because the row also held %-change columns).
  3. Independent re-answer of all 58 answerable questions from the source text by Mistral Large 4 (a different vendor from both the answer model and the judge), without seeing the reference. 57 agreed on every figure and percentage; Mistral returned an empty reply for the NVIDIA H20 question every time, so Nemotron 3 Ultra answered it instead, and it agreed.
  4. Arithmetic for all derived answers recomputed.
  5. The 8 unanswerable questions: the facts are absent from the corpus (searched; NVIDIA mentions OpenAI but never revenue from it).
- **Found:** Gemini 3.5 Flash's free tier allows only 20 requests/day (hit during this check). Gemini 3.5 Flash-Lite, the eval judge, has a separate quota.

## 2026-11-22: Hybrid retrieval and reranker
- **BM25 in Qdrant, not in Python memory:** chosen so one database serves dense, keyword and hybrid search (Phase 6 deploys one service; every API replica would otherwise hold its own index). Each point has a named dense vector and a BM25 sparse vector (fastembed `Qdrant/bm25`, IDF applied by Qdrant). Rank fusion (RRF) over 50 candidates from each.
- **Measured (66 golden questions, same answer model and judge):**

  | Retrieval | R@1 | R@5 | R@10 | Accuracy |
  | --- | --- | --- | --- | --- |
  | Vector | 0.190 | 0.478 | 0.599 | 0.530 |
  | Hybrid | 0.198 | 0.513 | 0.612 | 0.545 |
  | Hybrid + MiniLM-L6 (CPU) | 0.233 | 0.431 | 0.556 | 0.470 |
  | Hybrid + bge-reranker-base (CPU) | 0.371 | 0.539 | 0.694 | 0.606 |
  | **Hybrid + Nemotron 3.5 Rerank 8B (NVIDIA hosted)** | **0.565** | **0.737** | **0.741** | **0.727** |

- **Hybrid alone** fixed 3 questions (exact terms: BoA net interest income, Walmart associates, Merck R&D) and broke 3 (keyword hits pushed the right chunk below top 5). The reranker is what turns the wider candidate list into a gain.
- **MiniLM made results worse** (trained on web search; mis-ranks financial tables). **bge** helped but took 13.5 s per question on CPU. **NVIDIA's reranker** beat bge on 8 questions and lost none, at ~0.4 s per question.
- **Chosen:** hybrid + NVIDIA rerank of 30 candidates is now the default (`from_env`, `make eval`). Tradeoff: a hosted dependency, and questions plus chunk text go to NVIDIA (already the case for the answer model). Local fallback if the endpoint is unavailable: bge (`--rerank bge`).
- **vLLM was considered for speed and rejected:** it needs an NVIDIA GPU and self-hosted serving is out of scope.
- **Retired endpoints:** NVIDIA's older rerankers (llama-3.2-nv-rerankqa-1b-v2, llama-nemotron-rerank-1b-v2) returned 410 Gone; check the catalog before relying on a model name.
- **Noted:** R@10 (0.741) is barely above R@5 (0.737), so the 30-candidate pool now caps recall; a larger pool is cheap with a hosted reranker and worth testing. Qdrant keeps optimizing its index for a few minutes after a bulk rebuild, and evals in that window shift slightly; measure after it settles.
- **Security review of the reranker:** no key leaks. Fixed before merge: the hosted response is validated (every passage scored exactly once, finite scores) and nothing is cached otherwise, since a defaulted score could float an unscored chunk into the prompt; retries also cover network errors, honor `Retry-After`, and don't sleep after the last attempt; cache writes are atomic. Checked: no cached NVIDIA score had been defaulted. **For Phase 4:** hybrid + rerank widens how an attacker-written chunk can reach the top 5 (keyword stuffing for BM25, "answer-shaped" text for the reranker), so the attack corpus should include both and attack success should be measured with rerank on and off. **For Phase 3:** the retry loop can hold a request ~1-2 min; give the API path a short time budget and fall back to no rerank.

## 2026-11-22: Rerank candidate pool and faithfulness metric
- **Candidates (NVIDIA reranker, hybrid):** 20 -> R@5 0.720, accuracy 0.742; 30 -> 0.737, 0.727; 50 -> 0.780, 0.758. Differences are 1-3 questions of 66, but 50 is best on both and costs little with a hosted reranker. **Chosen: 50** (the hybrid prefetch ceiling). Local rerankers scale linearly with it.
- **Faithfulness:** the judge sees the answer and only the chunks it cites, and checks every claim is stated in (or computed from) the cited chunk; uncited claims count as unfaithful; refusals are excluded. Final config: 0.978 (44/45). The one miss added a Tesla detail that its cited chunk does not contain.
- **Judge parsing:** Flash-Lite once returned `{"faithful": true, "reason": "<one sentence>}` (unclosed string), which counted as a failure. The parser now falls back to reading just the verdict field from malformed JSON.
- **Corrective retrieval loop** (grade chunks, rewrite query, retry up to 3 rounds) was proposed and moved to Phase 3, where it is built as LangGraph nodes with per-company question splitting; plan.md updated.

## 2026-11-22: Contextual chunk headers and multi-row table headers
- **Problem (measured):** most remaining table misses were retrieval misses. The AMD income statement chunk ranked below 200 in dense search for "How much did AMD spend on R&D in 2025?": the chunk was labels and numbers, with no company name, table title or year in words. 37% of chunks (17,705) are tables. Split tables also repeated only their first header row ("Year Ended"), losing the row with the dates.
- **Change:** every chunk starts with a context line (company, ticker, form, fiscal period, section, and for tables the short text block above the table as its title). Split tables repeat all leading header rows up to the first body row.
- **Measured on a 5-company subset** (AMD, NVDA, TGT, COST, MRK; the 18 golden questions sourced only from them; separate collections, same retrieval code; `make eval --recall-only`, so no LLM calls):

  | Chunks | Retrieval | R@1 | R@5 |
  | --- | --- | --- | --- |
  | Old | hybrid | 0.139 | 0.417 |
  | New | hybrid | 0.417 | 0.583 |
  | Old | hybrid + NVIDIA rerank | 0.361 | 0.528 |
  | New | hybrid + NVIDIA rerank | 0.639 | 0.750 |

- **Full corpus (all 20 companies, 58 answerable golden questions, `--recall-only`; 48,534 chunks):**

  | Retrieval | Old R@1 / @5 / @10 | New R@1 / @5 / @10 |
  | --- | --- | --- |
  | vector | 0.190 / 0.478 / 0.599 | 0.362 / 0.543 / 0.703 |
  | hybrid | 0.198 / 0.513 / 0.612 | 0.397 / 0.586 / 0.716 |
  | hybrid + NVIDIA rerank (50) | 0.582 / 0.780 / 0.780 | 0.728 / 0.841 / 0.858 |

- **Chosen:** new chunking. Accuracy on the full set waits for the judge quota to reset.
- **Judge quota:** Gemini 3.5 Flash-Lite's free tier is 500 requests/day (hit during this experiment; each question now costs 2 judge calls with faithfulness). Added `--recall-only` to `make eval` for retrieval experiments that need no LLM.

## 2026-11-22: Chunks flagged by NVIDIA's content-safety screen sink to the bottom
- **Problem:** after the re-chunk, the hosted reranker returned 422 "input rejected by content safety screening" for one golden question. Two ordinary Chevron 10-Q income-tax paragraphs trip it, with or without the context header, every time. One flagged passage rejects the whole 50-passage request.
- **Options:** sink flagged chunks to the bottom; drop them; skip reranking for that question.
- **Chosen:** sink. On that 422 the reranker halves the batch until each flagged passage is alone (about 12 extra calls for one flag in 50, instead of 50). Flagged passages get a fixed score of -1e9 (cached) and a log line with the text's hash. A chunk the reranker refused to score can never outrank one it scored, and the question keeps the rerank gain for the other 48 chunks.
- **Guards (from security review):** the error is matched on the API's exact message, not a substring, because 422 bodies can echo untrusted chunk text. If the question itself is flagged, the reranker raises instead of caching -1e9 for every passage. Flagged chunks can still fill the top-k when few chunks were scored, so the Phase 4 sanitizer must cover them like any other retrieved text.

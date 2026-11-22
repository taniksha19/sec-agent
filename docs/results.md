# Results

Every measured number in this project. Leave a cell blank if it was not measured. Never estimate.

| Phase | Metric | Baseline | Final | How it was measured |
| --- | --- | --- | --- | --- |
| 1 | Recall@5 (vector-only) | 0.478 (@1 0.190, @10 0.616) | 0.478 (Phase 1 ships the baseline) | `make eval` on 66 golden questions (machine-checked, see decisions.md); bge-small + Qdrant top-5, Nemotron 3.5 Super answers, Gemini 3.5 Flash-Lite judge; 2026-11-22 |
| 1 | Answer accuracy (vector-only) | 0.545 overall; fact 0.609, table 0.444, multi-hop 0.000, narrative 0.750, unanswerable 1.000 | 0.545 | Same run. 43% of answerable questions were refused ("not in the filings"); 0 invented answers on unanswerable ones |
| 2 | Recall@1 / @5 / @10 | vector: 0.190 / 0.478 / 0.599 | hybrid + NVIDIA rerank of 50, contextual chunk headers: 0.728 / 0.841 / 0.858 | `make eval`, 66 golden questions (58 answerable), 2026-11-22. Before headers: hybrid + NVIDIA 0.582 / 0.780 / 0.780; hybrid 0.198 / 0.513 / 0.612; + MiniLM (30): 0.233 / 0.431 / 0.556; + bge (30): 0.371 / 0.539 / 0.694; + NVIDIA (20 / 30): R@5 0.720 / 0.737. With headers: vector 0.362 / 0.543 / 0.703; hybrid 0.397 / 0.586 / 0.716 |
| 2 | Answer accuracy | vector: 0.530 | 0.758 (fact 0.870, table 0.667, multi-hop 0.222, narrative 1.000, unanswerable 1.000) | Same runs. Hybrid 0.545; + MiniLM 0.470; + bge 0.606; + NVIDIA 20 / 30 candidates: 0.742 / 0.727. Vector re-measured after the index rebuild (0.545 in Phase 1; one narrative question flips with the judge) |
| 2 | Citation faithfulness | | 0.978 (44 of 45 non-refusal answers) | Gemini 3.5 Flash-Lite judges each answer against only the chunks it cites; refusals excluded. Measured on the final config only |
| 2 | Refused when answerable | vector: 0.431 | 0.224 | Share of answerable questions answered "not in the filings" |
| 2 | Rerank time per question | | NVIDIA hosted ~0.4 s (30 chunks); bge (CPU) 13.5 s; MiniLM (CPU) 2.1 s | Timed on one question, Ryzen 5 5500U; NVIDIA's first call ~4 s (cold start) |
| 3 | Recall and p95 latency, backend A vs. backend B | | | |
| 4 | Attack success rate with and without sanitizer; false-positive rate | | | |
| 4 | Added latency from sanitizer | | | |
| 5 | Cost per query: strong-only vs. routed (at matched accuracy) | | | |
| 5 | Answer accuracy and p95 latency: cheap-only vs. strong-only vs. routed | | | |
| 6 | Load test: requests/s, p50/p95 latency, pod count | | | |
| 6 | CI: merge blocked on a deliberate regression (yes/no, evidence) | | | |

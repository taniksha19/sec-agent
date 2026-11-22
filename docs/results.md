# Results

Every measured number in this project. Leave a cell blank if it was not measured. Never estimate.

| Phase | Metric | Baseline | Final | How it was measured |
| --- | --- | --- | --- | --- |
| 1 | Recall@5 (vector-only) | 0.478 (@1 0.190, @10 0.616) | 0.478 (Phase 1 ships the baseline) | `make eval` on 66 golden questions (machine-checked, see decisions.md); bge-small + Qdrant top-5, Nemotron 3.5 Super answers, Gemini 3.5 Flash-Lite judge; 2026-11-22 |
| 1 | Answer accuracy (vector-only) | 0.545 overall; fact 0.609, table 0.444, multi-hop 0.000, narrative 0.750, unanswerable 1.000 | 0.545 | Same run. 43% of answerable questions were refused ("not in the filings"); 0 invented answers on unanswerable ones |
| 2 | Recall@1 / @5 / @10 | vector: 0.190 / 0.478 / 0.599 | hybrid + NVIDIA rerank: 0.565 / 0.737 / 0.741 | `make eval` on the 66 golden questions, 2026-11-22. Hybrid alone: 0.198 / 0.513 / 0.612; + MiniLM: 0.233 / 0.431 / 0.556; + bge: 0.371 / 0.539 / 0.694 |
| 2 | Answer accuracy | vector: 0.530 | hybrid + NVIDIA rerank: 0.727 (fact 0.870, table 0.611, multi-hop 0.111, narrative 1.000, unanswerable 1.000) | Same runs. Hybrid 0.545; + MiniLM 0.470; + bge 0.606. Vector re-measured after the index rebuild (0.545 in Phase 1; one narrative question flips with the judge) |
| 2 | Rerank time per question (30 candidates) | | NVIDIA hosted ~0.4 s; bge (CPU) 13.5 s; MiniLM (CPU) 2.1 s | Timed on one question, Ryzen 5 5500U; NVIDIA's first call ~4 s (cold start) |
| 3 | Recall and p95 latency, backend A vs. backend B | | | |
| 4 | Attack success rate with and without sanitizer; false-positive rate | | | |
| 4 | Added latency from sanitizer | | | |
| 5 | Cost per query: strong-only vs. routed (at matched accuracy) | | | |
| 5 | Answer accuracy and p95 latency: cheap-only vs. strong-only vs. routed | | | |
| 6 | Load test: requests/s, p50/p95 latency, pod count | | | |
| 6 | CI: merge blocked on a deliberate regression (yes/no, evidence) | | | |

# Results

Every measured number in this project. Leave a cell blank if it was not measured. Never estimate.

| Phase | Metric | Baseline | Final | How it was measured |
| --- | --- | --- | --- | --- |
| 1 | Recall@5 (vector-only) | 0.478 (@1 0.190, @10 0.616) | 0.478 (Phase 1 ships the baseline) | `make eval` on 66 golden questions (machine-checked, see decisions.md); bge-small + Qdrant top-5, Nemotron 3.5 Super answers, Gemini 3.5 Flash-Lite judge; 2026-11-22 |
| 1 | Answer accuracy (vector-only) | 0.545 overall; fact 0.609, table 0.444, multi-hop 0.000, narrative 0.750, unanswerable 1.000 | 0.545 | Same run. 43% of answerable questions were refused ("not in the filings"); 0 invented answers on unanswerable ones |
| 2 | Recall@5, answer accuracy (vector-only vs. hybrid + rerank) | | | |
| 3 | Recall and p95 latency, backend A vs. backend B | | | |
| 4 | Attack success rate with and without sanitizer; false-positive rate | | | |
| 4 | Added latency from sanitizer | | | |
| 5 | Cost per query: strong-only vs. routed (at matched accuracy) | | | |
| 5 | Answer accuracy and p95 latency: cheap-only vs. strong-only vs. routed | | | |
| 6 | Load test: requests/s, p50/p95 latency, pod count | | | |
| 6 | CI: merge blocked on a deliberate regression (yes/no, evidence) | | | |

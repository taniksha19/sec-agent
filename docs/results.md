# Results

Every measured number in this project. Leave a cell blank if it was not measured. Never estimate.

| Phase | Metric | Baseline | Final | How it was measured |
| --- | --- | --- | --- | --- |
| 1 | Recall@5 (vector-only) | | | |
| 1 | Answer accuracy (vector-only) | | | |
| 2 | Recall@5, answer accuracy (vector-only vs. hybrid + rerank) | | | |
| 3 | Recall and p95 latency, backend A vs. backend B | | | |
| 4 | Attack success rate with and without sanitizer; false-positive rate | | | |
| 4 | Added latency from sanitizer | | | |
| 5 | Cost per query: strong-only vs. routed (at matched accuracy) | | | |
| 5 | Answer accuracy and p95 latency: cheap-only vs. strong-only vs. routed | | | |
| 6 | Load test: requests/s, p50/p95 latency, pod count | | | |
| 6 | CI: merge blocked on a deliberate regression (yes/no, evidence) | | | |

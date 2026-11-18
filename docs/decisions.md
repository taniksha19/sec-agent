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

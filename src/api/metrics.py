"""Prometheus metrics for the agent API, served on /metrics (inside docker compose only: the
gateway does not forward that path).

Recorded per /v1/ask request from what the agent returns, so library code stays free of
metrics. Per-step timing (each LLM call, search, rerank) lives in the Langfuse trace.
"""

from prometheus_client import Counter, Histogram

from api.models import ToolCall

# Agent runs take seconds to minutes (hosted-LLM speed varies a lot), so buckets are wide.
LATENCY_BUCKETS = (1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180)
KNOWN_TOOLS = {"calculate", "compare", "lookup_table"}

REQUESTS = Counter(
    "agent_requests_total", "Agent requests by outcome: answered, refused or error", ["outcome"]
)
LATENCY = Histogram("agent_request_seconds", "Agent run wall time", buckets=LATENCY_BUCKETS)
LLM_CALLS = Counter("agent_llm_calls_total", "LLM calls made by the agent")
TOKENS = Counter("agent_llm_tokens_total", "LLM tokens by direction", ["direction"])
ROUNDS = Histogram(
    "agent_retrieval_rounds", "Most retrieval rounds any sub-question needed", buckets=(1, 2, 3)
)
TOOL_CALLS = Counter("agent_tool_calls_total", "Tool calls by tool and result", ["tool", "ok"])


def tool_label(call: ToolCall) -> str:
    # Tool names come from model output; an unknown name must not become a new time series.
    return call.name if call.name in KNOWN_TOOLS else "other"


def record(out: dict) -> None:
    REQUESTS.labels("refused" if out["refused"] else "answered").inc()
    LATENCY.observe(out["latency_ms"] / 1000)
    LLM_CALLS.inc(out["llm_calls"])
    TOKENS.labels("input").inc(out["input_tokens"])
    TOKENS.labels("output").inc(out["output_tokens"])
    ROUNDS.observe(out["rounds_used"])
    for call in out["tool_calls"]:
        TOOL_CALLS.labels(tool_label(call), str(call.ok).lower()).inc()


def record_error(seconds: float) -> None:
    REQUESTS.labels("error").inc()
    LATENCY.observe(seconds)

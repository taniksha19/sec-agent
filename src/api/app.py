"""HTTP API over the Phase 3 agent.

Run: uv run uvicorn api.app:app --port 8000
Then: curl -X POST localhost:8000/v1/ask -H "Content-Type: application/json" \\
        -d '{"question": "How much did AMD spend on R&D in 2025?"}'

Auth, rate limits and request logging live in the Go gateway (gateway/) in front of this service;
in docker compose this API has no host port.
"""

import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from langfuse import propagate_attributes
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import metrics
from api.models import AskRequest, AskResponse, Citation, Usage
from retrieval.base import SearchResult

log = logging.getLogger(__name__)
QUOTE_CHARS = 300
# Rejects other Host headers, so a web page can't reach the local API through DNS rebinding.
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "api"]
# The gateway sets X-Request-ID (16 hex chars); anything else is not copied into traces.
REQUEST_ID_RE = re.compile(r"[0-9a-f]{16}")


def citation(n: int, result: SearchResult) -> Citation:
    c = result.chunk
    body = c.text.split("\n", 1)[-1]  # drop the context header line
    return Citation(
        n=n,
        chunk_id=c.chunk_id,
        company=c.company,
        ticker=c.ticker,
        form_type=c.form_type,
        fiscal_year=c.fiscal_year,
        fiscal_period=c.fiscal_period,
        section_title=c.section_title,
        source_url=c.source_url,
        page=c.page,
        quote=body[:QUOTE_CHARS],
    )


def to_response(out: dict) -> AskResponse:
    sources = out["sources"]
    return AskResponse(
        answer=out["answer"],
        refused=out["refused"],
        citations=[citation(n, sources[n - 1]) for n in out["cited"]],
        sub_questions=[s.question for s in out["subs"]],
        rounds=out["rounds_used"],
        tool_calls=out["tool_calls"],
        usage=Usage(
            llm_calls=out["llm_calls"],
            input_tokens=out["input_tokens"],
            output_tokens=out["output_tokens"],
        ),
        latency_ms=out["latency_ms"],
    )


def create_app(agent=None) -> FastAPI:
    """Tests pass a fake agent; the server builds the real one once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            from agent.graph import from_env

            app.state.agent = from_env()
        else:
            app.state.agent = agent
        yield

    app = FastAPI(
        title="SEC Filings Research Agent",
        version="0.3.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    # Scraped by Prometheus inside docker compose; the gateway's route allowlist excludes it.
    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    # A sync route: FastAPI runs it in a worker thread, so a slow agent run doesn't block others.
    @app.post("/v1/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        request_id = request.headers.get("x-request-id", "")
        metadata = {"request_id": request_id} if REQUEST_ID_RE.fullmatch(request_id) else None
        start = time.monotonic()
        try:
            # Ties the Langfuse trace to the gateway's log line for the same request.
            with propagate_attributes(trace_name="ask", metadata=metadata):
                out = request.app.state.agent.run(body.question)
        except Exception:
            # Upstream errors (LLM, reranker, Qdrant) may carry URLs or request details: they go
            # to the local log and the Langfuse trace (span status), never to the client.
            log.exception("agent run failed")
            metrics.record_error(time.monotonic() - start)
            raise HTTPException(502, "The research agent failed upstream.") from None
        metrics.record(out)
        return to_response(out)

    return app


app = create_app()

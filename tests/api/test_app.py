import pytest
from fastapi.testclient import TestClient

from agent.graph import SubQuestion
from api.app import create_app
from api.models import ToolCall
from ingest.models import Chunk
from retrieval.base import SearchResult


def source(n):
    chunk = Chunk(
        chunk_id=f"acc:item_8:{n}",
        text=f"AMD (AMD) 10-K annual report, fiscal 2025. Financial Statements.\nbody text {n}",
        ticker="AMD",
        company="Advanced Micro Devices",
        form_type="10-K",
        accession_number="acc",
        filing_date="2026-02-01",
        fiscal_year=2025,
        fiscal_period="FY",
        source_url="https://www.sec.gov/x",
        section_id="item_8",
        section_title="Financial Statements",
        chunk_index=n,
    )
    return SearchResult(chunk=chunk, score=1.0)


class FakeAgent:
    def __init__(self, answer="R&D was $8,091 million [2].", cited=(2,), fail=False):
        self.answer, self.cited, self.fail = answer, list(cited), fail

    def run(self, question):
        if self.fail:
            raise RuntimeError("upstream https://internal.example/secret-path 500")
        return {
            "answer": self.answer,
            "refused": not self.cited,
            "sources": [source(1), source(2)],
            "cited": self.cited,
            "subs": [SubQuestion(question=question, ticker="AMD")],
            "rounds_used": 1,
            "tool_calls": [ToolCall(name="calculate", arguments={"expression": "1+1"}, ok=True)],
            "llm_calls": 3,
            "input_tokens": 100,
            "output_tokens": 20,
            "latency_ms": 1234,
        }


def client(agent):
    return TestClient(create_app(agent), base_url="http://localhost")


def test_healthz():
    with client(FakeAgent()) as c:
        assert c.get("/healthz").json() == {"status": "ok"}


def test_ask_returns_only_cited_sources_without_context_header():
    with client(FakeAgent()) as c:
        r = c.post("/v1/ask", json={"question": "AMD R&D 2025?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert [cit["n"] for cit in body["citations"]] == [2]
    assert body["citations"][0]["quote"] == "body text 2"
    assert body["tool_calls"][0]["name"] == "calculate"
    assert body["usage"] == {"llm_calls": 3, "input_tokens": 100, "output_tokens": 20}


def test_refusal_has_no_citations():
    agent = FakeAgent("The provided filings do not contain this information.", cited=())
    with client(agent) as c:
        body = c.post("/v1/ask", json={"question": "q"}).json()
    assert body["refused"] is True and body["citations"] == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 1001}])
def test_bad_requests_are_rejected(payload):
    with client(FakeAgent()) as c:
        assert c.post("/v1/ask", json=payload).status_code == 422


def test_upstream_failure_is_502_without_details():
    with client(FakeAgent(fail=True)) as c:
        r = c.post("/v1/ask", json={"question": "q"})
    assert r.status_code == 502
    assert "internal.example" not in r.text


def test_rejects_unknown_host_and_hides_docs():
    client = TestClient(create_app(FakeAgent()), base_url="http://evil.example")
    assert client.get("/healthz").status_code == 400
    local = TestClient(create_app(FakeAgent()), base_url="http://localhost")
    assert local.get("/docs").status_code == 404
    assert local.get("/openapi.json").status_code == 404


def metric(c, line_start):
    for line in c.get("/metrics").text.splitlines():
        if line.startswith(line_start):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_count_outcomes_tokens_and_tools():
    with client(FakeAgent()) as c:
        answered = metric(c, 'agent_requests_total{outcome="answered"}')
        tokens = metric(c, 'agent_llm_tokens_total{direction="input"}')
        calc = metric(c, 'agent_tool_calls_total{ok="true",tool="calculate"}')
        assert c.post("/v1/ask", json={"question": "AMD R&D 2025?"}).status_code == 200
        assert metric(c, 'agent_requests_total{outcome="answered"}') == answered + 1
        assert metric(c, 'agent_llm_tokens_total{direction="input"}') == tokens + 100
        assert metric(c, 'agent_tool_calls_total{ok="true",tool="calculate"}') == calc + 1
    with client(FakeAgent(fail=True)) as c:
        errors = metric(c, 'agent_requests_total{outcome="error"}')
        assert c.post("/v1/ask", json={"question": "AMD R&D 2025?"}).status_code == 502
        assert metric(c, 'agent_requests_total{outcome="error"}') == errors + 1


def test_unknown_tool_names_share_one_label():
    from api.metrics import tool_label

    assert tool_label(ToolCall(name="calculate", arguments={}, ok=True)) == "calculate"
    assert tool_label(ToolCall(name="x" * 50, arguments={}, ok=False)) == "other"

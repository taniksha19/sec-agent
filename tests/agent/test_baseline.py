from datetime import date

from agent.baseline import BaselineRAG, build_prompt, parse_citations, source_label
from ingest.models import Chunk
from llm_gateway.client import THINK_RE, LLMResponse
from retrieval.base import SearchResult


def make_result(text="Revenue was $10.", section="item_7", score=0.9):
    chunk = Chunk(
        chunk_id=f"acc:{section}:0",
        text=text,
        ticker="AAPL",
        company="Apple",
        form_type="10-K",
        accession_number="acc",
        filing_date=date(2026, 10, 30),
        fiscal_year=2026,
        fiscal_period="FY",
        source_url="https://example",
        section_id=section,
        section_title="MD&A",
        chunk_index=0,
        page=23,
    )
    return SearchResult(chunk=chunk, score=score)


def test_source_label():
    assert source_label(make_result()) == "Apple 10-K FY2026 FY, item_7 (MD&A), p.23"


def test_build_prompt_numbers_sources():
    prompt = build_prompt("Q?", [make_result("A"), make_result("B", "item_8")])
    assert "[1] Apple" in prompt and "\nA\n" in prompt and "[2] Apple" in prompt
    assert prompt.endswith("Question: Q?")


def test_parse_citations_dedupes_and_drops_out_of_range():
    assert parse_citations("X [2]. Y [1][2]. Z [9].", 5) == [2, 1]


def test_think_tags_removed():
    assert THINK_RE.sub("", "<think>hmm\nok</think>Answer").strip() == "Answer"


class FakeStore:
    def search(self, query_vector, k, filters=None):
        return [make_result()][:k]


class FakeEmbedder:
    def embed_query(self, text):
        return [0.0]


class FakeLLM:
    def chat(self, system, user, max_tokens=800):
        assert "Revenue was $10." in user
        return LLMResponse("Revenue was $10 [1].", "fake", 100, 10)


def test_baseline_answer_wires_retrieval_and_citations():
    ans = BaselineRAG(FakeStore(), FakeEmbedder(), FakeLLM()).answer("What was revenue?")
    assert ans.cited == [1]
    assert ans.sources[0].chunk.ticker == "AAPL"
    assert (ans.input_tokens, ans.output_tokens) == (100, 10)

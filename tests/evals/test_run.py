from agent.baseline import NOT_FOUND
from evals.run import is_refusal, load_golden, parse_verdict, recall_at_k, summarize
from ingest.models import Chunk
from llm_gateway.cache import CachedLLM
from llm_gateway.client import LLMResponse


def chunk(ticker, text):
    return Chunk(
        chunk_id=f"{ticker}:{text}",
        text=text,
        ticker=ticker,
        company=ticker,
        form_type="10-K",
        accession_number="acc",
        filing_date="2026-01-01",
        period_of_report="2025-12-31",
        fiscal_year=2025,
        fiscal_period="FY",
        source_url="https://example.com",
        section_id="item_8",
        section_title="Financial Statements",
        chunk_index=0,
    )


def test_recall_counts_sources_found_in_top_k():
    chunks = [
        chunk("NVDA", "noise"),
        chunk("NVDA", "Total revenue $215,938 in the release"),
        chunk("NVDA", "R&D | 18,497 | y"),
    ]
    sources = [
        {"company": "NVDA", "match": ["215,938"]},
        {"company": "NVDA", "match": ["18,497"]},
    ]
    assert recall_at_k(sources, chunks, 1) == 0
    assert recall_at_k(sources, chunks, 2) == 0.5
    assert recall_at_k(sources, chunks, 3) == 1


def test_recall_matches_figures_across_millions_and_rounded_billions():
    big = [{"company": "NVDA", "match": ["215,938"]}]
    assert recall_at_k(big, [chunk("NVDA", "revenue of $215.9 billion, up 65%")], 1) == 1
    assert recall_at_k(big, [chunk("NVDA", "revenue of $215.4 billion")], 1) == 0
    two_decimals = [{"company": "TSLA", "match": ["94,827"]}]
    assert recall_at_k(two_decimals, [chunk("TSLA", "revenues of $94.83 billion")], 1) == 1
    from_text = [{"company": "MSFT", "match": ["77.7 billion"]}]
    assert recall_at_k(from_text, [chunk("MSFT", "Revenue | 77,673 | 65,585")], 1) == 1


def test_recall_skips_rounded_billions_under_ten():
    small = [{"company": "INTC", "match": ["2,214"]}]
    assert recall_at_k(small, [chunk("INTC", "a loss of $2.2 billion")], 1) == 0
    assert recall_at_k(small, [chunk("INTC", "loss | ( 2,214 )")], 1) == 1


def test_recall_requires_all_key_terms():
    sources = [{"company": "INTC", "match": ["TSMC", "Samsung", "UMC"]}]
    assert recall_at_k(sources, [chunk("INTC", "TSMC, Samsung, UMC and SMIC")], 1) == 1
    assert recall_at_k(sources, [chunk("INTC", "TSMC and Samsung")], 1) == 0


def test_recall_ignores_same_number_from_another_company():
    sources = [{"company": "AMD", "match": ["18,497"]}]
    assert recall_at_k(sources, [chunk("NVDA", "R&D | 18,497")], 1) == 0


def test_refusal_matches_with_or_without_period_and_case():
    assert is_refusal(NOT_FOUND)
    assert is_refusal("the provided filings do not contain this information")
    assert not is_refusal("Revenue was $94,827 million [1].")
    assert not is_refusal(f"Amazon's net sales were $700 billion. {NOT_FOUND}")


def test_parse_verdict_handles_fenced_json_and_garbage():
    assert parse_verdict('```json\n{"correct": true, "reason": "ok"}\n```') == (True, "ok")
    assert parse_verdict('{"correct": false, "reason": "wrong year"}') == (False, "wrong year")
    correct, reason = parse_verdict("I think it is right")
    assert not correct and "unparseable" in reason


def test_summarize_excludes_unanswerable_from_recall():
    rows = [
        {
            "type": "fact",
            "correct": True,
            "answer": "x",
            "recall": {"1": 1, "5": 1, "10": 1},
            "input_tokens": 100,
            "output_tokens": 10,
        },
        {
            "type": "fact",
            "correct": False,
            "answer": NOT_FOUND,
            "recall": {"1": 0, "5": 0, "10": 1},
            "input_tokens": 100,
            "output_tokens": 10,
        },
        {
            "type": "unanswerable",
            "correct": True,
            "answer": NOT_FOUND,
            "recall": None,
            "input_tokens": 100,
            "output_tokens": 10,
        },
    ]
    s = summarize(rows)
    assert s["recall"] == {"@1": 0.5, "@5": 0.5, "@10": 1.0}
    assert s["accuracy"] == 0.667
    assert s["accuracy_answerable"] == 0.5
    assert s["refused_answerable"] == 0.5
    assert s["by_type"]["unanswerable"] == {"n": 1, "accuracy": 1.0}


def test_golden_set_is_well_formed():
    golden = load_golden()
    assert len(golden) >= 50
    assert len({q["id"] for q in golden}) == len(golden)
    for q in golden:
        assert q["type"] in {"fact", "table", "multi_hop", "narrative", "unanswerable"}
        assert bool(q["sources"]) == (q["type"] != "unanswerable"), q["id"]
        if q["type"] == "unanswerable":
            assert q["expected_answer"] == NOT_FOUND


class CountingLLM:
    model = "m"
    base_url = "https://example.com/v1"
    extra_body = None

    def __init__(self):
        self.calls = 0

    def chat(self, system, user, max_tokens=800):
        self.calls += 1
        return LLMResponse(text=f"reply {self.calls}", model="m", input_tokens=5, output_tokens=2)


def test_cache_reuses_identical_calls_and_misses_on_change(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path)
    first = llm.chat("sys", "q")
    assert llm.chat("sys", "q") == first
    assert inner.calls == 1
    llm.chat("sys", "different q")
    assert inner.calls == 2


def test_cache_key_includes_request_options(tmp_path):
    inner = CountingLLM()
    CachedLLM(inner, tmp_path).chat("sys", "q")
    inner.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    CachedLLM(inner, tmp_path).chat("sys", "q")
    assert inner.calls == 2


class RecordingJudge:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat(self, system, user, max_tokens=800):
        self.prompts.append(user)
        return LLMResponse(text=self.reply, model="j", input_tokens=1, output_tokens=1)


def test_faithfulness_sends_only_cited_sources_and_skips_refusals():
    from agent.baseline import Answer
    from evals.run import faithfulness
    from retrieval.base import SearchResult

    srcs = [SearchResult(chunk=chunk("AAPL", f"source text {i}"), score=0.0) for i in range(3)]
    ans = Answer(question="q", text="Sales were $1 [2].", sources=srcs, cited=[2])
    judge = RecordingJudge('{"faithful": true, "reason": "ok"}')
    assert faithfulness(judge, ans) == (True, "ok")
    assert "source text 1" in judge.prompts[0] and "source text 0" not in judge.prompts[0]

    refusal = Answer(question="q", text=NOT_FOUND, sources=srcs, cited=[])
    assert faithfulness(judge, refusal) == (None, "refusal")
    assert len(judge.prompts) == 1


def test_parse_verdict_requires_literal_true():
    assert parse_verdict('{"faithful": "yes"}', "faithful")[0] is False
    assert parse_verdict('{"faithful": true}', "faithful")[0] is True


def test_parse_verdict_reads_verdict_from_malformed_json():
    broken = '```json\n{"faithful": true, "reason": "<one sentence>}'
    assert parse_verdict(broken, "faithful")[0] is True
    assert parse_verdict('{"correct": false, "reason": "x', "correct")[0] is False


def test_format_table_one_row_per_setup_and_blank_when_unmeasured():
    from evals.run import format_table

    recall = {"@1": 0.1, "@5": 0.5, "@10": 0.6}
    full = {"retrieval": "hybrid+nvidia", "recall": recall, "accuracy": 0.75}
    full |= {"faithfulness": 0.98, "refused_answerable": None}
    table = format_table([{"retrieval": "vector", "recall": recall}, full])
    lines = table.splitlines()
    assert len(lines) == 4
    assert lines[2] == "| vector | 0.100 | 0.500 | 0.600 |  |  |  |"
    assert lines[3] == "| hybrid+nvidia | 0.100 | 0.500 | 0.600 | 0.750 | 0.980 |  |"

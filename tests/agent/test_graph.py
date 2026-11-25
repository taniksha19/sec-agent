import json

from agent.graph import GRADE_PROMPT, MAX_ROUNDS, Agent
from ingest.models import Chunk
from llm_gateway.client import LLMResponse
from retrieval.base import SearchResult

TABLE = "AMD 10-K. Table: Consolidated Statements of Operations.\nDec 2025 | Dec 2024\nResearch and development | 8,091 | 6,456"


def result(ticker, n, text="text", is_table=False):
    chunk = Chunk(
        chunk_id=f"{ticker}:{n}",
        text=text,
        ticker=ticker,
        company=ticker,
        form_type="10-K",
        accession_number="acc",
        filing_date="2026-02-01",
        fiscal_year=2025,
        fiscal_period="FY",
        source_url="https://example.com",
        section_id="item_8",
        section_title="Financial Statements",
        chunk_index=n,
        is_table=is_table,
    )
    return SearchResult(chunk=chunk, score=1.0)


class FakeRAG:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, k=None, filters=None):
        self.calls.append((query, filters))
        ticker = (filters or {}).get("ticker", "ANY")
        if (filters or {}).get("is_table"):
            return [result(ticker, 99, TABLE, is_table=True)]
        return [result(ticker, len(self.calls))]


def reply(text="", tool_calls=None):
    return LLMResponse(
        text=text, model="m", input_tokens=10, output_tokens=2, tool_calls=tool_calls or []
    )


class ScriptedLLM:
    """Plan reply, then grade replies in order, then answer-turn replies in order."""

    def __init__(self, plan, grades, answers):
        self.plan, self.grades, self.answers = plan, list(grades), list(answers)
        self.answer_messages = []

    def chat(self, system, user, max_tokens=800):
        if system.startswith(GRADE_PROMPT):
            return reply(json.dumps(self.grades.pop(0)))
        return reply(json.dumps(self.plan))

    def complete(self, messages, tools=None, max_tokens=800):
        self.answer_messages.append(messages)
        return self.answers.pop(0)


COMPANIES = {"AMD": "Advanced Micro Devices", "NVDA": "NVIDIA"}
ONE_SUB = {"sub_questions": [{"question": "AMD R&D 2025?", "ticker": "AMD"}]}
YES = {"answered": True, "better_query": ""}


def test_grader_rewrites_query_until_answered():
    llm = ScriptedLLM(
        ONE_SUB,
        [
            {"answered": False, "better_query": "AMD research and development expense fiscal 2025"},
            YES,
        ],
        [reply("AMD spent $8,091 million [1].")],
    )
    rag = FakeRAG()
    out = Agent(rag, llm, COMPANIES).run("How much did AMD spend on R&D in 2025?")
    assert [q for q, _ in rag.calls] == [
        "AMD R&D 2025?",
        "AMD research and development expense fiscal 2025",
    ]
    assert all(f == {"ticker": "AMD"} for _, f in rag.calls)
    assert out["rounds_used"] == 2
    assert out["cited"] == [1] and not out["refused"]
    assert out["llm_calls"] == 4  # plan, 2 grades, answer


def test_retrieval_loop_stops_after_max_rounds():
    no = {"answered": False, "better_query": "try again"}
    llm = ScriptedLLM(
        ONE_SUB, [no] * 5, [reply("The provided filings do not contain this information.")]
    )
    rag = FakeRAG()
    out = Agent(rag, llm, COMPANIES).run("q")
    assert len(rag.calls) == MAX_ROUNDS
    assert out["rounds_used"] == MAX_ROUNDS
    assert out["refused"]


def test_planner_splits_per_company_and_drops_unknown_tickers():
    plan = {
        "sub_questions": [
            {"question": "NVDA R&D", "ticker": "NVDA"},
            {"question": "AMD R&D", "ticker": "AMD"},
            {"question": "Other R&D", "ticker": "ZZZZ"},
        ]
    }
    llm = ScriptedLLM(plan, [YES, YES, YES], [reply("answer [1] [2]")])
    rag = FakeRAG()
    out = Agent(rag, llm, COMPANIES).run("Compare R&D")
    assert [f for _, f in rag.calls] == [{"ticker": "NVDA"}, {"ticker": "AMD"}, None]
    assert len(out["sources"]) == 3


def test_bad_plan_falls_back_to_the_original_question():
    llm = ScriptedLLM({"oops": 1}, [YES], [reply("x")])
    rag = FakeRAG()
    Agent(rag, llm, COMPANIES).run("original question")
    assert rag.calls[0] == ("original question", None)


def call(name, args, id_="c1"):
    return {"id": id_, "name": name, "arguments": json.dumps(args)}


def test_answer_runs_tools_and_table_rows_become_citable_sources():
    lookup = {"ticker": "amd", "fiscal_year": 2025, "metric": "research and development"}
    llm = ScriptedLLM(
        ONE_SUB,
        [YES],
        [
            reply(tool_calls=[call("lookup_table", lookup, "a")]),
            reply(
                tool_calls=[
                    call("calculate", {"expression": "(8,091 - 6,456) / 6,456 * 100"}, "b"),
                    call("calculate", {"expression": "__import__('os')"}, "c"),
                    call("lookup_table", {**lookup, "ticker": "EVIL"}, "d"),
                    call("delete_everything", {}, "e"),
                ]
            ),
            reply("R&D grew 25.3% [2]."),
        ],
    )
    out = Agent(FakeRAG(), llm, COMPANIES).run("AMD R&D growth 2025?")
    assert [(t.name, t.ok) for t in out["tool_calls"]] == [
        ("lookup_table", True),
        ("calculate", True),
        ("calculate", False),
        ("lookup_table", False),
        ("delete_everything", False),
    ]
    assert out["sources"][1].chunk.chunk_id == "AMD:99"  # the table row's chunk, now [2]
    assert out["cited"] == [2]
    tool_msgs = [m for m in llm.answer_messages[-1] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["rows"][0]["source"] == 2
    assert abs(json.loads(tool_msgs[1]["content"])["value"] - 25.325) < 0.01


def test_answer_turns_are_capped_and_last_turn_offers_no_tools():
    from agent.graph import MAX_TOOL_STEPS

    loop = [reply(tool_calls=[call("calculate", {"expression": "1+1"})])] * MAX_TOOL_STEPS
    llm = ScriptedLLM(ONE_SUB, [YES], [*loop, reply("done [1]")])
    seen_tools = []
    original = llm.complete

    def complete(messages, tools=None, max_tokens=800):
        seen_tools.append(tools)
        return original(messages, tools, max_tokens)

    llm.complete = complete
    out = Agent(FakeRAG(), llm, COMPANIES).run("q")
    assert out["answer"] == "done [1]"
    assert seen_tools[-1] is None and all(seen_tools[:-1])


def test_non_object_tool_arguments_fail_the_call_not_the_run():
    bad = {"id": "x", "name": "calculate", "arguments": "[1]"}
    llm = ScriptedLLM(ONE_SUB, [YES], [reply(tool_calls=[bad]), reply("done")])
    out = Agent(FakeRAG(), llm, COMPANIES).run("q")
    assert [(t.name, t.ok, t.arguments) for t in out["tool_calls"]] == [("calculate", False, {})]
    assert out["answer"] == "done"


def test_tool_calls_are_capped_per_question():
    from agent.graph import MAX_TOOL_CALLS

    many = [call("calculate", {"expression": "1+1"}, f"id{i}") for i in range(MAX_TOOL_CALLS + 5)]
    llm = ScriptedLLM(ONE_SUB, [YES], [reply(tool_calls=many), reply("done")])
    out = Agent(FakeRAG(), llm, COMPANIES).run("q")
    oks = [t.ok for t in out["tool_calls"]]
    assert oks == [True] * MAX_TOOL_CALLS + [False] * 5
    tool_msgs = [m for m in llm.answer_messages[-1] if m["role"] == "tool"]
    assert len(tool_msgs) == len(many)  # every call id still gets a reply
    assert "budget" in tool_msgs[-1]["content"]


def test_evidence_from_every_round_is_kept_and_rewrites_are_capped():
    from agent.graph import MAX_QUERY_CHARS

    long_query = "x" * (MAX_QUERY_CHARS + 50)
    llm = ScriptedLLM(ONE_SUB, [{"answered": False, "better_query": long_query}, YES], [reply("a")])
    rag = FakeRAG()
    out = Agent(rag, llm, COMPANIES).run("q")
    assert len(rag.calls[1][0]) == MAX_QUERY_CHARS
    assert [r.chunk.chunk_id for r in out["sources"]] == ["AMD:1", "AMD:2"]

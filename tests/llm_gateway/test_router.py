"""Routing policies, provider fallback and the per-question token cap."""

from dataclasses import replace

import httpx
import openai
import pytest

from agent.graph import MAX_QUESTION_TOKENS, Agent
from llm_gateway.client import LLMResponse
from llm_gateway.router import POLICIES, Fallback, Router
from tests.agent.test_graph import COMPANIES, ONE_SUB, YES, FakeRAG, ScriptedLLM, call, reply

REQUEST = httpx.Request("POST", "https://llm.example/v1/chat/completions")


def status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


class StubLLM:
    def __init__(self, model, error=None):
        self.model, self.error, self.calls = model, error, 0

    def chat(self, system, user, max_tokens=800):
        return self.complete([], None, max_tokens)

    def complete(self, messages, tools=None, max_tokens=800):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(text="ok", model=self.model, input_tokens=1, output_tokens=1)


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=REQUEST),
        openai.APIConnectionError(request=REQUEST),
        status_error(openai.RateLimitError, 429),
        status_error(openai.InternalServerError, 503),
    ],
)
def test_provider_failures_fall_back_to_the_other_provider(error):
    backup = StubLLM("backup")
    llm = Fallback(StubLLM("primary", error), backup)
    assert llm.chat("s", "u").model == "backup"
    assert llm.complete([{"role": "user", "content": "u"}]).model == "backup"
    assert backup.calls == 2


def test_bad_requests_do_not_fall_back():
    backup = StubLLM("backup")
    llm = Fallback(StubLLM("primary", status_error(openai.BadRequestError, 400)), backup)
    with pytest.raises(openai.BadRequestError):
        llm.chat("s", "u")
    assert backup.calls == 0


def test_policies_cover_every_step_and_unknown_policy_fails():
    for policy in POLICIES.values():
        assert set(policy) == {"plan", "grade", "answer"}
    with pytest.raises(ValueError):
        Router({}, "cheapest")


class Labelled(ScriptedLLM):
    """A ScriptedLLM whose replies carry its own model name."""

    def __init__(self, model, *args):
        super().__init__(*args)
        self.model = model

    def chat(self, system, user, max_tokens=800):
        return replace(super().chat(system, user, max_tokens), model=self.model)

    def complete(self, messages, tools=None, max_tokens=800):
        return replace(super().complete(messages, tools, max_tokens), model=self.model)


def test_routed_policy_sends_plan_and_grade_to_fast_and_answer_to_strong():
    fast = Labelled("fast", ONE_SUB, [YES], [])
    strong = Labelled("strong", ONE_SUB, [], [reply("AMD spent $8,091 million [1].")])
    out = Agent(FakeRAG(), Router({"fast": fast, "strong": strong}, "routed"), COMPANIES).run("q")
    assert out["models"] == ["fast", "fast", "strong"]
    assert out["answer"].startswith("AMD spent")


def test_answer_turn_drops_tools_past_the_token_cap():
    big = replace(
        reply(tool_calls=[call("calculate", {"expression": "1+1"})]),
        input_tokens=MAX_QUESTION_TOKENS,
    )
    llm = ScriptedLLM(ONE_SUB, [YES], [big, reply("done [1]")])
    offered = []
    original = llm.complete

    def complete(messages, tools=None, max_tokens=800):
        offered.append(tools is not None)
        return original(messages, tools, max_tokens)

    llm.complete = complete
    out = Agent(FakeRAG(), llm, COMPANIES).run("q")
    assert offered == [True, False]
    assert out["answer"] == "done [1]"


def test_missing_gemini_key_fails_instead_of_sending_another_key(monkeypatch):
    from llm_gateway import router

    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "nvidia-key")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        router.from_env("routed")


def test_fallback_mid_tool_conversation_sends_tool_history_as_text():
    class Recorder(StubLLM):
        def complete(self, messages, tools=None, max_tokens=800):
            self.seen = messages
            return super().complete(messages, tools, max_tokens)

    backup = Recorder("backup")
    llm = Fallback(StubLLM("primary", status_error(openai.RateLimitError, 429)), backup)
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "x1",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "x1", "content": '{"value": 2}'},
    ]
    llm.complete(history)
    assert [m["role"] for m in backup.seen] == ["user", "assistant", "user"]
    assert "calculate({})" in backup.seen[1]["content"] and "tool_calls" not in backup.seen[1]
    assert backup.seen[2]["content"].endswith('{"value": 2}')

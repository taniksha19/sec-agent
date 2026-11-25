"""Minimal chat client for any OpenAI-compatible endpoint (NVIDIA hosted, Ollama, Groq, ...).

Configured from .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. Routing is in llm_gateway/router.py.
"""

import os
import re
import time
from dataclasses import dataclass, field

from langfuse import get_client, observe
from openai import OpenAI

# Some open models (Nemotron, Qwen) emit their reasoning in <think> tags before the answer.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Nemotron 3.5 thinks by default, which spent a whole 1500-token budget on one RAG answer with
# nothing left for the reply; answers were as good without it. Gemini rejects this field (400),
# so it is opt-in per client.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    # Tool calls the model asked for: [{"id", "name", "arguments" (a JSON string)}]
    tool_calls: list[dict] = field(default_factory=list)


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        extra_body: dict | None = None,
        min_interval: float = 0.0,
        max_retries: int = 6,
        timeout: float = 60,
    ) -> None:
        # `is None`, not `or`: an empty value must never pick up another provider's key or model.
        self.model = os.environ["LLM_MODEL"] if model is None else model
        self.extra_body = extra_body
        self.min_interval = min_interval  # seconds between calls, to stay under per-minute caps
        self._last_call = 0.0
        self.base_url = os.environ["LLM_BASE_URL"] if base_url is None else base_url
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ["LLM_API_KEY"] if api_key is None else api_key,
            max_retries=max_retries,  # free tiers rate-limit (429); the SDK backs off and retries
            timeout=timeout,  # seconds per attempt; the SDK default is 600
        )

    def chat(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.complete(messages, max_tokens=max_tokens)

    # Traced as a Langfuse generation (a no-op without LANGFUSE_* keys). Inputs are set by hand:
    # automatic capture would record `self`, and the prompt is enough to debug a call.
    @observe(as_type="generation", name="llm", capture_input=False, capture_output=False)
    def complete(
        self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 800
    ) -> LLMResponse:
        """One chat turn over a full message list, optionally offering OpenAI-format tools."""
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,  # reproducible answers for evals
            max_tokens=max_tokens,
            extra_body=self.extra_body,
            **({"tools": tools} if tools else {}),
        )
        message = resp.choices[0].message
        text = THINK_RE.sub("", message.content or "").strip()
        calls = [
            {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
            for c in message.tool_calls or []
        ]
        usage = resp.usage
        out = LLMResponse(
            text=text,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            tool_calls=calls,
        )
        get_client().update_current_generation(
            model=self.model,
            input=messages,
            output={"text": text, "tool_calls": calls} if calls else text,
            model_parameters={"max_tokens": max_tokens, "tools": len(tools or [])},
            usage_details={"input": out.input_tokens, "output": out.output_tokens},
        )
        return out

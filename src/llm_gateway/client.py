"""Minimal chat client for any OpenAI-compatible endpoint (NVIDIA hosted, Ollama, Groq, ...).

Configured from .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. Phase 5 adds routing on top.
"""

import os
import re
import time
from dataclasses import dataclass

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


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        extra_body: dict | None = None,
        min_interval: float = 0.0,
    ) -> None:
        self.model = model or os.environ["LLM_MODEL"]
        self.extra_body = extra_body
        self.min_interval = min_interval  # seconds between calls, to stay under per-minute caps
        self._last_call = 0.0
        self.base_url = base_url or os.environ["LLM_BASE_URL"]
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or os.environ["LLM_API_KEY"],
            max_retries=6,  # free tiers rate-limit (429); the SDK backs off between retries
            timeout=60,  # the SDK default is 600 s per attempt
        )

    def chat(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,  # reproducible answers for evals
            max_tokens=max_tokens,
            extra_body=self.extra_body,
        )
        text = THINK_RE.sub("", resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

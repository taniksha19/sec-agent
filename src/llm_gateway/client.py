"""Minimal chat client for any OpenAI-compatible endpoint (NVIDIA hosted, Ollama, Groq, ...).

Configured from .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. Phase 5 adds routing on top.
"""

import os
import re
from dataclasses import dataclass

from openai import OpenAI

# Some open models (Nemotron, Qwen) emit their reasoning in <think> tags before the answer.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


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
    ) -> None:
        self.model = model or os.environ["LLM_MODEL"]
        self._client = OpenAI(
            base_url=base_url or os.environ["LLM_BASE_URL"],
            api_key=api_key or os.environ["LLM_API_KEY"],
        )

    def chat(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,  # reproducible answers for evals
            max_tokens=max_tokens,
            # Nemotron 3.5 thinks by default, which spent a whole 1500-token budget on one RAG
            # answer with nothing left for the reply. Answers were as good without it.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = THINK_RE.sub("", resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

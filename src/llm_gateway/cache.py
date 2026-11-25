"""On-disk cache for chat calls, so eval reruns don't repeat paid or rate-limited requests.

The key covers the endpoint, model, request options and both prompts. Calls run at
temperature 0, so a cached reply stands in for a fresh one; change any input and the call
is made again.
"""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from llm_gateway.client import LLMClient, LLMResponse


class CachedLLM:
    def __init__(self, llm: LLMClient, cache_dir: Path, read: bool = True) -> None:
        """read=False always calls the API (to time real calls) but still writes the cache."""
        self.llm = llm
        self.model = llm.model
        self.cache_dir = Path(cache_dir)
        self.read = read

    def _path(self, *request) -> Path:
        llm = self.llm
        key = json.dumps([llm.base_url, llm.model, llm.extra_body, *request], sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _cached(self, path: Path, call) -> LLMResponse:
        if self.read and path.exists():
            return LLMResponse(**json.loads(path.read_text(encoding="utf-8")))
        resp = call()
        if resp.text or resp.tool_calls:  # don't pin an empty reply; retry it next run
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(resp)), encoding="utf-8")
            os.replace(tmp, path)  # atomic, so an interrupted write can't leave a broken entry
        return resp

    def chat(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        # Key layout predates complete(); kept so earlier cached replies still hit.
        path = self._path(system, user, max_tokens)
        return self._cached(path, lambda: self.llm.chat(system, user, max_tokens))

    def complete(
        self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 800
    ) -> LLMResponse:
        path = self._path("complete", messages, tools, max_tokens)
        return self._cached(path, lambda: self.llm.complete(messages, tools, max_tokens))

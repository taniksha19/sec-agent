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
    def __init__(self, llm: LLMClient, cache_dir: Path) -> None:
        self.llm = llm
        self.model = llm.model
        self.cache_dir = Path(cache_dir)

    def _path(self, system: str, user: str, max_tokens: int) -> Path:
        llm = self.llm
        key = json.dumps(
            [llm.base_url, llm.model, llm.extra_body, system, user, max_tokens], sort_keys=True
        )
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def chat(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        path = self._path(system, user, max_tokens)
        if path.exists():
            return LLMResponse(**json.loads(path.read_text(encoding="utf-8")))
        resp = self.llm.chat(system, user, max_tokens)
        if resp.text:  # don't pin an empty reply; retry it next run
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(resp)), encoding="utf-8")
            os.replace(tmp, path)  # atomic, so an interrupted write can't leave a broken entry
        return resp

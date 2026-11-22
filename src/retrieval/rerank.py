"""Cross-encoder reranking: score each (question, chunk) pair jointly and re-sort.

Retrieval returns a wide candidate list (dense or hybrid); the reranker reads the question with
each chunk's text, which is slower than vector search but much better at picking the chunk that
actually answers. Scores are cached on disk because CPU reranking is the slowest retrieval step
and eval reruns ask the same questions.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import httpx
from fastembed.rerank.cross_encoder import TextCrossEncoder

from retrieval.base import SearchResult
from retrieval.embeddings import DEFAULT_CACHE_DIR

MODELS = {
    "minilm": "Xenova/ms-marco-MiniLM-L-6-v2",  # local CPU, 2.1 s per 30 chunks
    "bge": "BAAI/bge-reranker-base",  # local CPU, 13.5 s per 30 chunks
    "nvidia": "nvidia/nemotron-3.5-rerank-8b",  # hosted on NVIDIA GPUs, ~0.4 s per 30 chunks
}
HOSTED = {"nvidia"}
NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking"

# How many retrieved chunks the reranker re-sorts. Measured with the NVIDIA reranker (66 golden
# questions): 20 -> R@5 0.720, 30 -> 0.737, 50 -> 0.780. 50 is the hybrid prefetch ceiling.
# Local models scale linearly with it (bge: 13.5 s per 30 chunks on a laptop CPU).
CANDIDATES = 50

RETRIES = 4
RETRY_STATUS = {429, 500, 502, 503, 504}


def parse_rankings(data: dict, n: int) -> list[float]:
    """Scores in passage order. Rejects any response that doesn't score every passage exactly
    once with a finite number: a defaulted score could float an unscored (possibly injected)
    chunk to the top, and it would be cached."""
    rankings = data.get("rankings") if isinstance(data, dict) else None
    if not isinstance(rankings, list):
        raise ValueError("rerank response has no 'rankings' list")
    scores: dict[int, float] = {}
    for item in rankings:
        index, logit = item.get("index"), float(item.get("logit", "nan"))
        if not isinstance(index, int) or not 0 <= index < n or index in scores:
            raise ValueError(f"rerank response has a bad or duplicate index: {index!r}")
        if not math.isfinite(logit):
            raise ValueError(f"rerank response has a non-finite score for index {index}")
        scores[index] = logit
    if len(scores) != n:
        raise ValueError(f"rerank response scored {len(scores)} of {n} passages")
    return [scores[i] for i in range(n)]


class Reranker:
    def __init__(self, name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.name = name
        self.model_name = MODELS[name]
        self._model_dir = Path(cache_dir) / "fastembed"
        self._score_dir = Path(cache_dir) / "rerank"
        self._model: TextCrossEncoder | None = None

    @property
    def model(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(self.model_name, cache_dir=str(self._model_dir))
        return self._model

    def _cache_path(self, question: str) -> Path:
        digest = hashlib.sha256(f"{self.model_name}\n{question}".encode()).hexdigest()
        return self._score_dir / digest[:2] / f"{digest}.json"

    def scores(self, question: str, texts: list[str]) -> list[float]:
        """One score per text; higher means more relevant. Cached per question and text."""
        path = self._cache_path(question)
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        missing = [i for i, key in enumerate(keys) if key not in cached]
        if missing:
            todo = [texts[i] for i in missing]
            new = (
                self._hosted(question, todo)
                if self.name in HOSTED
                else self.model.rerank(question, todo)
            )
            for i, score in zip(missing, new, strict=True):
                cached[keys[i]] = float(score)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cached), encoding="utf-8")
            os.replace(tmp, path)  # atomic, so an interrupted write can't leave a broken entry
        return [cached[key] for key in keys]

    def _hosted(self, question: str, texts: list[str]) -> list[float]:
        body = {
            "model": self.model_name,
            "query": {"text": question},
            "passages": [{"text": t} for t in texts],
        }
        headers = {"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"}
        for attempt in range(RETRIES):
            last = attempt == RETRIES - 1
            try:
                resp = httpx.post(NVIDIA_RERANK_URL, json=body, headers=headers, timeout=30)
            except httpx.TransportError:
                if last:
                    raise
                time.sleep(2**attempt)
                continue
            if resp.status_code not in RETRY_STATUS or last:
                break
            # Free tier rate limit or a cold start.
            time.sleep(min(float(resp.headers.get("Retry-After", 2**attempt)), 30))
        resp.raise_for_status()
        return parse_rankings(resp.json(), len(texts))

    def rerank(self, question: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        scores = self.scores(question, [r.chunk.text for r in results])
        ranked = sorted(zip(scores, results, strict=True), key=lambda p: p[0], reverse=True)
        return [SearchResult(chunk=r.chunk, score=s) for s, r in ranked[:k]]

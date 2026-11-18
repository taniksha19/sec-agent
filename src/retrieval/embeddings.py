"""Local CPU embeddings via fastembed (ONNX, no torch), with an on-disk cache for documents."""

import hashlib
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384

# bge v1.5 expects this instruction on queries (not documents) for retrieval.
# fastembed's query_embed does NOT add it for this model (checked in fastembed 0.9.1:
# OnnxTextEmbedding inherits the base query_embed, which just calls embed), so we add it here.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.model_name = model_name
        self.dim = DIM
        self.vector_dir = Path(cache_dir) / "embeddings"
        self._model_dir = Path(cache_dir) / "fastembed"
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        # Loaded lazily so cache-only reruns never touch the model.
        if self._model is None:
            self._model = TextEmbedding(self.model_name, cache_dir=str(self._model_dir))
        return self._model

    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()
        return self.vector_dir / key[:2] / f"{key}.npy"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        paths = [self._cache_path(t) for t in texts]
        vectors: list[np.ndarray | None] = [np.load(p) if p.exists() else None for p in paths]

        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            new = self.model.embed([texts[i] for i in missing])
            for i, vector in zip(missing, new, strict=True):
                paths[i].parent.mkdir(parents=True, exist_ok=True)
                np.save(paths[i], vector.astype(np.float32))
                vectors[i] = vector
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([BGE_QUERY_PREFIX + text]))).tolist()

"""Keyword (BM25) vectors via fastembed, for hybrid search.

fastembed's Qdrant/bm25 gives each document term a BM25 term-frequency weight; the collection's
IDF modifier makes Qdrant apply inverse document frequency at query time, so the corpus-wide
statistics never have to be computed here. Cheap on CPU, so not cached.
"""

from pathlib import Path

from fastembed import SparseTextEmbedding

from retrieval.base import SparseVector
from retrieval.embeddings import DEFAULT_CACHE_DIR

MODEL_NAME = "Qdrant/bm25"


class SparseEmbedder:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self._model_dir = Path(cache_dir) / "fastembed"
        self._model: SparseTextEmbedding | None = None

    @property
    def model(self) -> SparseTextEmbedding:
        if self._model is None:
            self._model = SparseTextEmbedding(MODEL_NAME, cache_dir=str(self._model_dir))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [_to_sparse(e) for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> SparseVector:
        # query_embed weights each query term once (no TF), as BM25 expects.
        return _to_sparse(next(iter(self.model.query_embed(text))))


def _to_sparse(embedding) -> SparseVector:
    return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())

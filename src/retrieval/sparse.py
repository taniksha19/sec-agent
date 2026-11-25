"""Keyword (BM25) vectors via fastembed, for hybrid search.

fastembed's Qdrant/bm25 gives each document term a BM25 term-frequency weight; the collection's
IDF modifier makes Qdrant apply inverse document frequency at query time, so the corpus-wide
statistics never have to be computed here. Cheap on CPU, so not cached.
"""

import math
from collections import Counter
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


def with_idf(vectors: list[SparseVector]) -> list[SparseVector]:
    """Multiply each document term weight by its IDF over `vectors` (the whole corpus).

    For stores without Qdrant's IDF modifier (Pinecone): query . (idf * doc) equals Qdrant's
    sum of query * idf * doc, with Qdrant's formula ln((N - n + 0.5) / (n + 0.5) + 1),
    checked against Qdrant's scores on the real corpus. Adding documents later changes N and
    n, so the whole corpus must be re-weighted and re-uploaded."""
    df = Counter(i for v in vectors for i in set(v.indices))
    n = len(vectors)
    idf = {i: math.log((n - c + 0.5) / (c + 0.5) + 1) for i, c in df.items()}
    return [
        SparseVector(
            indices=v.indices,
            values=[w * idf[i] for i, w in zip(v.indices, v.values, strict=True)],
        )
        for v in vectors
    ]


def _to_sparse(embedding) -> SparseVector:
    return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())

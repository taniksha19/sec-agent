"""The one interface every vector store backend implements.

Qdrant is the primary backend; a second backend is added in Phase 3.
"""

from typing import Any, Protocol

from pydantic import BaseModel

from ingest.models import Chunk

# Filters are exact-match on Chunk metadata fields, e.g. {"ticker": "AAPL", "form_type": "10-K"}.
# A list value means "any of", e.g. {"ticker": ["AAPL", "MSFT"]}.
Filters = dict[str, Any]


class SearchResult(BaseModel):
    chunk: Chunk
    score: float


class SparseVector(BaseModel):
    """Keyword (BM25) term weights: token ids and their weights."""

    indices: list[int]
    values: list[float]


class VectorStore(Protocol):
    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
    ) -> None:
        """Insert or replace chunks by chunk_id. vectors[i] is the embedding of chunks[i];
        sparse_vectors[i], if given, is its keyword vector for hybrid search."""
        ...

    def search(
        self, query_vector: list[float], k: int, filters: Filters | None = None
    ) -> list[SearchResult]:
        """Return the top-k chunks by dense similarity, highest score first."""
        ...

    def hybrid_search(
        self,
        query_vector: list[float],
        query_sparse: SparseVector,
        k: int,
        filters: Filters | None = None,
    ) -> list[SearchResult]:
        """Return the top-k chunks from dense and keyword search merged by rank fusion."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete chunks by chunk_id. Missing ids are ignored."""
        ...

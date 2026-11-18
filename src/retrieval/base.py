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


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks by chunk_id. vectors[i] is the embedding of chunks[i]."""
        ...

    def search(
        self, query_vector: list[float], k: int, filters: Filters | None = None
    ) -> list[SearchResult]:
        """Return the top-k chunks by similarity, highest score first."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete chunks by chunk_id. Missing ids are ignored."""
        ...

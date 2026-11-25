"""Pinecone backend for the VectorStore interface, for comparison with Qdrant.

Pinecone keeps dense and sparse vectors in separate indexes, so hybrid search queries both and
fuses the lists here with the same reciprocal rank fusion Qdrant uses. Both backends then run
the same method and differ only in the engine. Every record carries the whole chunk as
metadata, so results need no second lookup.
"""

from concurrent.futures import ThreadPoolExecutor

from ingest.models import Chunk
from retrieval.base import Filters, SearchResult, SparseVector
from retrieval.qdrant_store import PREFETCH_LIMIT

DENSE_INDEX = "sec-dense"
SPARSE_INDEX = "sec-sparse"
TAGS = {"project": "sec-agent"}

# Qdrant's RRF: score = sum over lists of 1 / (rank + RRF_K), rank from 0. Measured: this
# reproduces Qdrant's fused scores exactly; the textbook k=60 does not.
RRF_K = 2

# Pinecone caps one upsert at 2 MB; a 384-float vector plus a chunk of text is ~10 KB as JSON.
UPSERT_BATCH = 100
# Per request; a hung query fails instead of blocking the caller.
QUERY_TIMEOUT_S = 30

_FILTERABLE_FIELDS = set(Chunk.model_fields) - {"text"}


def to_pinecone_filter(filters: Filters | None) -> dict | None:
    """Exact match per field; a list value means any-of. All fields must match."""
    if not filters:
        return None
    out = {}
    for field, value in filters.items():
        if field not in _FILTERABLE_FIELDS:
            raise ValueError(f"Unknown filter field: {field!r}")
        if isinstance(value, list | tuple | set):
            out[field] = {"$in": list(value)}
        else:
            out[field] = {"$eq": value}
    return out


def to_metadata(chunk: Chunk) -> dict:
    # Pinecone metadata has no null; missing optional fields come back as their defaults.
    return {k: v for k, v in chunk.model_dump(mode="json").items() if v is not None}


def rrf(lists: list[list[SearchResult]], k: int) -> list[SearchResult]:
    """Fuse ranked lists; equal scores are ordered by chunk_id, as in QdrantStore."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in lists:
        for rank, r in enumerate(results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1 / (rank + RRF_K)
            chunks[cid] = r.chunk
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [SearchResult(chunk=chunks[cid], score=scores[cid]) for cid in order[:k]]


class PineconeStore:
    def __init__(self, dense_index, sparse_index) -> None:
        """Takes `pc.Index(...)` handles; `create_indexes` makes them on first use."""
        self.dense = dense_index
        self.sparse = sparse_index
        self._pool = ThreadPoolExecutor(max_workers=2)

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
    ) -> None:
        """sparse_vectors must already carry IDF (`sparse.with_idf`): Pinecone has no IDF
        modifier, unlike the Qdrant collection."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but {len(sparse_vectors)} sparse vectors")
        meta = [to_metadata(c) for c in chunks]
        dense = [
            {"id": c.chunk_id, "values": list(v), "metadata": m}
            for c, v, m in zip(chunks, vectors, meta, strict=True)
        ]
        _upsert(self.dense, dense)
        if sparse_vectors is not None:
            sparse = [
                {"id": c.chunk_id, "sparse_values": s.model_dump(), "metadata": m}
                for c, s, m in zip(chunks, sparse_vectors, meta, strict=True)
            ]
            _upsert(self.sparse, sparse)

    def search(
        self, query_vector: list[float], k: int, filters: Filters | None = None
    ) -> list[SearchResult]:
        return _results(
            self.dense.query(
                top_k=k,
                vector=list(query_vector),
                filter=to_pinecone_filter(filters),
                include_metadata=True,
                timeout=QUERY_TIMEOUT_S,
            )
        )

    def hybrid_search(
        self,
        query_vector: list[float],
        query_sparse: SparseVector,
        k: int,
        filters: Filters | None = None,
    ) -> list[SearchResult]:
        if k > PREFETCH_LIMIT:
            raise ValueError(f"k={k} exceeds PREFETCH_LIMIT={PREFETCH_LIMIT}")
        pfilter = to_pinecone_filter(filters)
        # Two indexes, two round trips: run them at once, as Qdrant runs its prefetches.
        dense = self._pool.submit(
            self.dense.query,
            top_k=PREFETCH_LIMIT,
            vector=list(query_vector),
            filter=pfilter,
            include_metadata=True,
            timeout=QUERY_TIMEOUT_S,
        )
        sparse = self._pool.submit(
            self.sparse.query,
            top_k=PREFETCH_LIMIT,
            sparse_vector=query_sparse.model_dump(),
            filter=pfilter,
            include_metadata=True,
            timeout=QUERY_TIMEOUT_S,
        )
        return rrf([_results(dense.result()), _results(sparse.result())], k)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.dense.delete(ids=ids)
        self.sparse.delete(ids=ids)


def create_indexes(pc, dim: int) -> tuple:
    """Create the two serverless indexes if missing (free Starter plan: AWS us-east-1 only)."""
    from pinecone import ServerlessSpec

    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    if not pc.has_index(DENSE_INDEX):
        pc.create_index(DENSE_INDEX, spec, dimension=dim, metric="cosine", tags=TAGS)
    if not pc.has_index(SPARSE_INDEX):
        pc.create_index(SPARSE_INDEX, spec, metric="dotproduct", vector_type="sparse", tags=TAGS)
    return pc.Index(DENSE_INDEX), pc.Index(SPARSE_INDEX)


def _upsert(index, records: list[dict]) -> None:
    resp = index.upsert(vectors=records, batch_size=UPSERT_BATCH, show_progress=False)
    if resp.has_errors:
        raise RuntimeError(f"{resp.failed_item_count} records failed to upload: {resp.errors[:1]}")


def _results(response) -> list[SearchResult]:
    return [
        SearchResult(chunk=Chunk.model_validate(m.metadata), score=m.score)
        for m in response.matches
    ]

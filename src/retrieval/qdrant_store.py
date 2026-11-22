"""Qdrant backend for the VectorStore interface.

Each point holds a named dense vector and a named BM25 sparse vector, so dense, keyword and
hybrid search all run inside Qdrant.
"""

import uuid

from qdrant_client import QdrantClient, models

from ingest.models import Chunk
from retrieval.base import Filters, SearchResult, SparseVector

# Qdrant point ids must be ints or UUIDs, so chunk_ids are mapped with uuid5.
# The namespace is fixed so the same chunk_id always maps to the same point id.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "sec-agent/chunk")

DENSE = "dense"
SPARSE = "bm25"

# Each search contributes this many candidates to rank fusion. Also the ceiling for k, and
# sized to leave room for a reranker to re-sort a deeper list later.
PREFETCH_LIMIT = 50

_FILTERABLE_FIELDS = set(Chunk.model_fields) - {"text"}

_PAYLOAD_INDEXES = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "form_type": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
    "section_id": models.PayloadSchemaType.KEYWORD,
}

_UPSERT_BATCH = 256


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def to_qdrant_filter(filters: Filters | None) -> models.Filter | None:
    """Exact match per field; a list value means any-of. All fields must match."""
    if not filters:
        return None
    conditions = []
    for field, value in filters.items():
        if field not in _FILTERABLE_FIELDS:
            raise ValueError(f"Unknown filter field: {field!r}")
        if isinstance(value, list | tuple | set):
            match = models.MatchAny(any=list(value))
        else:
            match = models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=field, match=match))
    return models.Filter(must=conditions)


def _sparse(v: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=v.indices, values=v.values)


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, vector_size: int) -> None:
        self.client = client
        self.collection = collection
        if not client.collection_exists(collection):
            client.create_collection(
                collection,
                vectors_config={
                    DENSE: models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            for field, schema in _PAYLOAD_INDEXES.items():
                client.create_payload_index(collection, field_name=field, field_schema=schema)
        elif not isinstance(client.get_collection(collection).config.params.vectors, dict):
            raise RuntimeError(
                f"Collection {collection!r} uses the old unnamed-vector layout. "
                "Rebuild it: uv run python -m retrieval.index --recreate"
            )

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but {len(sparse_vectors)} sparse vectors")
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            named: dict = {DENSE: list(vector)}
            if sparse_vectors is not None:
                named[SPARSE] = _sparse(sparse_vectors[i])
            points.append(
                models.PointStruct(
                    id=point_id(chunk.chunk_id), vector=named, payload=chunk.model_dump(mode="json")
                )
            )
        for start in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(self.collection, points=points[start : start + _UPSERT_BATCH])

    def search(
        self, query_vector: list[float], k: int, filters: Filters | None = None
    ) -> list[SearchResult]:
        response = self.client.query_points(
            self.collection,
            query=list(query_vector),
            using=DENSE,
            limit=k,
            query_filter=to_qdrant_filter(filters),
            with_payload=True,
        )
        return _results(response)

    def hybrid_search(
        self,
        query_vector: list[float],
        query_sparse: SparseVector,
        k: int,
        filters: Filters | None = None,
    ) -> list[SearchResult]:
        if k > PREFETCH_LIMIT:
            raise ValueError(f"k={k} exceeds PREFETCH_LIMIT={PREFETCH_LIMIT}")
        qfilter = to_qdrant_filter(filters)
        response = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(
                    query=list(query_vector), using=DENSE, limit=PREFETCH_LIMIT, filter=qfilter
                ),
                models.Prefetch(
                    query=_sparse(query_sparse), using=SPARSE, limit=PREFETCH_LIMIT, filter=qfilter
                ),
            ],
            # Reciprocal rank fusion: merges by rank, so dense cosine and BM25 scores, which
            # are on different scales, never have to be normalized against each other.
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            # Fetch the whole fused list: RRF gives equal scores to mirrored ranks (1st + 3rd vs
            # 3rd + 1st), Qdrant orders ties arbitrarily, and R@1 moved 0.37-0.40 between runs.
            # Breaking ties by chunk_id here makes the top k reproducible.
            limit=2 * PREFETCH_LIMIT,
            with_payload=True,
        )
        results = sorted(_results(response), key=lambda r: (-r.score, r.chunk.chunk_id))
        return results[:k]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(
            self.collection,
            points_selector=models.PointIdsList(points=[point_id(i) for i in ids]),
        )


def _results(response) -> list[SearchResult]:
    return [
        SearchResult(chunk=Chunk.model_validate(point.payload), score=point.score)
        for point in response.points
    ]

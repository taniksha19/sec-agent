"""Qdrant backend for the VectorStore interface."""

import uuid

from qdrant_client import QdrantClient, models

from ingest.models import Chunk
from retrieval.base import Filters, SearchResult

# Qdrant point ids must be ints or UUIDs, so chunk_ids are mapped with uuid5.
# The namespace is fixed so the same chunk_id always maps to the same point id.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "sec-agent/chunk")

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


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, vector_size: int) -> None:
        self.client = client
        self.collection = collection
        if not client.collection_exists(collection):
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            for field, schema in _PAYLOAD_INDEXES.items():
                client.create_payload_index(collection, field_name=field, field_schema=schema)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector=list(vector),
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(self.collection, points=points[start : start + _UPSERT_BATCH])

    def search(
        self, query_vector: list[float], k: int, filters: Filters | None = None
    ) -> list[SearchResult]:
        response = self.client.query_points(
            self.collection,
            query=list(query_vector),
            limit=k,
            query_filter=to_qdrant_filter(filters),
            with_payload=True,
        )
        return [
            SearchResult(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(
            self.collection,
            points_selector=models.PointIdsList(points=[point_id(i) for i in ids]),
        )

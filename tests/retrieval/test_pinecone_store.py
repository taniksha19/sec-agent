"""PineconeStore against a fake index: filters, metadata round trip, and Qdrant-equal fusion."""

from types import SimpleNamespace

import pytest

from retrieval.base import SearchResult, SparseVector
from retrieval.pinecone_store import (
    PineconeStore,
    rrf,
    to_metadata,
    to_pinecone_filter,
)


class FakeIndex:
    """Records upserts and queries; query returns preset matches in order."""

    def __init__(self, ranked_ids: list[str] | None = None) -> None:
        self.records: dict[str, dict] = {}
        self.ranked_ids = ranked_ids or []
        self.queries: list[dict] = []
        self.deleted: list[str] = []

    def upsert(self, vectors, batch_size, show_progress):
        for v in vectors:
            self.records[v["id"]] = v
        return SimpleNamespace(has_errors=False)

    def query(self, **kwargs):
        self.queries.append(kwargs)
        matches = [
            SimpleNamespace(id=i, score=1.0 / (n + 1), metadata=self.records[i]["metadata"])
            for n, i in enumerate(self.ranked_ids[: kwargs["top_k"]])
        ]
        return SimpleNamespace(matches=matches)

    def delete(self, ids):
        self.deleted += ids


def sparse(i: int) -> SparseVector:
    return SparseVector(indices=[i], values=[1.0])


@pytest.fixture
def store(chunks):
    ids = [c.chunk_id for c in chunks]
    # Dense ranks chunks 0,1,2,3; sparse ranks 2,3,0 (chunk 1 has no keyword hit).
    s = PineconeStore(FakeIndex(ids), FakeIndex([ids[2], ids[3], ids[0]]))
    s.upsert(chunks, [[0.0] * 4] * len(chunks), [sparse(i) for i in range(len(chunks))])
    return s


def test_metadata_round_trips_without_nulls(chunks):
    chunk = chunks[0].model_copy(update={"page": None, "fiscal_year": 2024})
    meta = to_metadata(chunk)
    assert "page" not in meta  # Pinecone metadata has no null
    # Pinecone returns numbers as floats.
    back = SearchResult(chunk={**meta, "fiscal_year": 2024.0, "chunk_index": 0.0}, score=1)
    assert back.chunk == chunk


def test_filters_translate_and_unknown_fields_fail():
    assert to_pinecone_filter(None) is None
    assert to_pinecone_filter({"ticker": ["AMD", "NVDA"], "fiscal_year": 2025}) == {
        "ticker": {"$in": ["AMD", "NVDA"]},
        "fiscal_year": {"$eq": 2025},
    }
    with pytest.raises(ValueError):
        to_pinecone_filter({"text": "x"})


def test_rrf_matches_qdrant_formula_and_breaks_ties_by_id(chunks):
    a = [SearchResult(chunk=c, score=0) for c in chunks[:2]]
    b = [SearchResult(chunk=c, score=0) for c in [chunks[1], chunks[0]]]
    fused = rrf([a, b], k=2)
    # Both get 1/2 + 1/3: a tie, ordered by chunk_id.
    assert [r.score for r in fused] == pytest.approx([1 / 2 + 1 / 3] * 2)
    assert [r.chunk.chunk_id for r in fused] == sorted(c.chunk_id for c in chunks[:2])


def test_hybrid_search_fuses_both_indexes_with_same_filter(store, chunks):
    results = store.hybrid_search([0.0] * 4, sparse(0), k=3, filters={"ticker": "AAPL"})
    # chunk 0: 1/2 + 1/4 and chunk 2: 1/4 + 1/2 (tie, by id); chunk 3: 1/5 + 1/3;
    # chunk 1: 1/3, dense only.
    assert results[0].score == pytest.approx(0.75)
    assert {r.chunk.chunk_id for r in results[:2]} == {chunks[0].chunk_id, chunks[2].chunk_id}
    assert results[2].chunk == chunks[3]
    for q in store.dense.queries + store.sparse.queries:
        assert q["filter"] == {"ticker": {"$eq": "AAPL"}}
        assert q["include_metadata"]
        assert q["top_k"] == 50
    assert store.sparse.queries[0]["sparse_vector"] == {"indices": [0], "values": [1.0]}


def test_upsert_checks_lengths_and_delete_hits_both(store, chunks):
    with pytest.raises(ValueError):
        store.upsert(chunks, [[0.0] * 4])
    assert store.sparse.records[chunks[0].chunk_id]["sparse_values"] == sparse(0).model_dump()
    store.delete([chunks[0].chunk_id])
    assert store.dense.deleted == store.sparse.deleted == [chunks[0].chunk_id]
    with pytest.raises(ValueError):
        store.hybrid_search([0.0] * 4, sparse(0), k=51)

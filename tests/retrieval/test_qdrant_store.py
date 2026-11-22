"""QdrantStore against Qdrant's in-process local mode, so no server is needed."""

import pytest
from qdrant_client import QdrantClient, models

from retrieval.base import SparseVector
from retrieval.qdrant_store import PREFETCH_LIMIT, QdrantStore

DIM = 4

# Local mode ignores payload indexes and warns about it; that is expected here.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def one_hot(i: int) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(DIM)]


@pytest.fixture
def store(chunks) -> QdrantStore:
    s = QdrantStore(QdrantClient(":memory:"), "test", vector_size=DIM)
    s.upsert(chunks, [one_hot(i) for i in range(len(chunks))])
    return s


def test_search_returns_nearest_first_with_full_payload(store, chunks):
    results = store.search(one_hot(2), k=2)
    assert len(results) == 2
    assert results[0].chunk == chunks[2]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score >= results[1].score


def test_search_with_scalar_filters(store, chunks):
    results = store.search(one_hot(0), k=10, filters={"ticker": "AAPL", "fiscal_year": 2023})
    assert [r.chunk.chunk_id for r in results] == [chunks[1].chunk_id]


def test_search_with_any_of_filter(store, chunks):
    results = store.search(one_hot(0), k=10, filters={"ticker": ["MSFT", "NVDA"]})
    assert {r.chunk.ticker for r in results} == {"MSFT", "NVDA"}


def test_upsert_replaces_by_chunk_id(store, chunks):
    updated = chunks[0].model_copy(update={"text": "Updated text."})
    store.upsert([updated], [one_hot(0)])
    results = store.search(one_hot(0), k=10)
    assert len(results) == len(chunks)
    assert results[0].chunk.text == "Updated text."


def test_delete_removes_and_ignores_missing(store, chunks):
    store.delete([chunks[0].chunk_id, "no-such-id"])
    ids = {r.chunk.chunk_id for r in store.search(one_hot(0), k=10)}
    assert chunks[0].chunk_id not in ids
    assert len(ids) == len(chunks) - 1


def test_existing_collection_is_reused(chunks):
    client = QdrantClient(":memory:")
    QdrantStore(client, "test", vector_size=DIM).upsert(chunks, [one_hot(0)] * len(chunks))
    again = QdrantStore(client, "test", vector_size=DIM)
    assert len(again.search(one_hot(0), k=10)) == len(chunks)


def test_mismatched_lengths_raise(chunks):
    s = QdrantStore(QdrantClient(":memory:"), "test", vector_size=DIM)
    with pytest.raises(ValueError):
        s.upsert(chunks, [one_hot(0)])


def sparse_for(word_ids: list[int]) -> SparseVector:
    return SparseVector(indices=word_ids, values=[1.0] * len(word_ids))


@pytest.fixture
def hybrid_store(chunks) -> QdrantStore:
    # Dense vectors point at chunk i; sparse vectors give chunk i the single "word" 100 + i.
    s = QdrantStore(QdrantClient(":memory:"), "test", vector_size=DIM)
    s.upsert(
        chunks,
        [one_hot(i) for i in range(len(chunks))],
        [sparse_for([100 + i]) for i in range(len(chunks))],
    )
    return s


def test_hybrid_returns_matches_from_both_dense_and_keyword(hybrid_store, chunks):
    # Dense query matches chunk 0, keyword query matches chunk 3: both must come back on top.
    results = hybrid_store.hybrid_search(one_hot(0), sparse_for([103]), k=2)
    assert {r.chunk.chunk_id for r in results} == {chunks[0].chunk_id, chunks[3].chunk_id}


def test_hybrid_ranks_agreement_first(hybrid_store, chunks):
    results = hybrid_store.hybrid_search(one_hot(2), sparse_for([102]), k=3)
    assert results[0].chunk == chunks[2]


def test_hybrid_applies_filters(hybrid_store, chunks):
    results = hybrid_store.hybrid_search(
        one_hot(0), sparse_for([100]), k=10, filters={"ticker": "MSFT"}
    )
    assert [r.chunk.ticker for r in results] == ["MSFT"]


def test_hybrid_rejects_k_above_prefetch(hybrid_store):
    with pytest.raises(ValueError):
        hybrid_store.hybrid_search(one_hot(0), sparse_for([100]), k=PREFETCH_LIMIT + 1)


def test_old_unnamed_vector_collection_is_rejected():
    client = QdrantClient(":memory:")
    client.create_collection(
        "old", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE)
    )
    with pytest.raises(RuntimeError, match="--recreate"):
        QdrantStore(client, "old", vector_size=DIM)

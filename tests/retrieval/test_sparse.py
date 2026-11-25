"""Runs the real BM25 model. Skipped if it can't be downloaded (e.g. offline)."""

import math

import pytest

from retrieval.base import SparseVector
from retrieval.sparse import SparseEmbedder, with_idf


@pytest.fixture(scope="module")
def sparse():
    e = SparseEmbedder()
    try:
        _ = e.model
    except Exception as err:  # network or download failure
        pytest.skip(f"BM25 model unavailable: {err}")
    return e


def test_with_idf_uses_qdrant_formula():
    docs = [
        SparseVector(indices=[1, 2], values=[1.0, 2.0]),
        SparseVector(indices=[1], values=[3.0]),
        SparseVector(indices=[3], values=[1.0]),
    ]
    out = with_idf(docs)
    idf = lambda n: math.log((3 - n + 0.5) / (n + 0.5) + 1)  # noqa: E731
    assert out[0].values == pytest.approx([1.0 * idf(2), 2.0 * idf(1)])
    assert out[1].values == pytest.approx([3.0 * idf(2)])
    assert out[2].indices == [3]


def test_shared_terms_share_token_ids(sparse):
    doc = sparse.embed_documents(["Intel competes with TSMC and Samsung in foundry services."])[0]
    query = sparse.embed_query("TSMC foundry")
    assert len(doc.indices) == len(doc.values) > 0
    assert set(query.indices) <= set(doc.indices)


def test_query_terms_weighted_once(sparse):
    q = sparse.embed_query("revenue revenue revenue")
    assert q.values == [1.0]

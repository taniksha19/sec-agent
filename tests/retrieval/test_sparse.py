"""Runs the real BM25 model. Skipped if it can't be downloaded (e.g. offline)."""

import pytest

from retrieval.sparse import SparseEmbedder


@pytest.fixture(scope="module")
def sparse():
    e = SparseEmbedder()
    try:
        _ = e.model
    except Exception as err:  # network or download failure
        pytest.skip(f"BM25 model unavailable: {err}")
    return e


def test_shared_terms_share_token_ids(sparse):
    doc = sparse.embed_documents(["Intel competes with TSMC and Samsung in foundry services."])[0]
    query = sparse.embed_query("TSMC foundry")
    assert len(doc.indices) == len(doc.values) > 0
    assert set(query.indices) <= set(doc.indices)


def test_query_terms_weighted_once(sparse):
    q = sparse.embed_query("revenue revenue revenue")
    assert q.values == [1.0]

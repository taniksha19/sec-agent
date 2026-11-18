"""Runs the real model. Skipped if it can't be downloaded (e.g. offline)."""

import numpy as np
import pytest

from retrieval.embeddings import DEFAULT_CACHE_DIR, DIM, Embedder


@pytest.fixture(scope="module")
def model():
    # Share the downloaded model with the rest of the repo; only the vector cache is per-test.
    try:
        return Embedder(cache_dir=DEFAULT_CACHE_DIR).model
    except Exception as e:  # network or download failure
        pytest.skip(f"embedding model unavailable: {e}")


@pytest.fixture
def embedder(model, tmp_path):
    e = Embedder(cache_dir=tmp_path)
    e._model = model
    return e


def test_dimensions_and_normalization(embedder):
    docs = embedder.embed_documents(["Apple reported revenue growth.", "Risk factors."])
    assert len(docs) == 2
    assert all(len(v) == DIM for v in docs)
    assert np.linalg.norm(docs[0]) == pytest.approx(1.0, abs=1e-3)
    assert len(embedder.embed_query("What was Apple's revenue?")) == DIM


def test_query_ranks_relevant_doc_first(embedder):
    docs = ["The company's revenue grew 8% year over year.", "The cafeteria serves lunch daily."]
    d = np.array(embedder.embed_documents(docs))
    q = np.array(embedder.embed_query("How much did revenue grow?"))
    scores = d @ q
    assert scores[0] > scores[1]


def test_document_embeddings_are_cached(embedder, tmp_path):
    first = embedder.embed_documents(["cached text"])
    assert len(list((tmp_path / "embeddings").rglob("*.npy"))) == 1

    class Boom:
        def embed(self, texts):
            raise AssertionError("model should not be called on a cache hit")

    embedder._model = Boom()
    assert embedder.embed_documents(["cached text"])[0] == pytest.approx(first[0])

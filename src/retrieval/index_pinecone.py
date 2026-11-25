"""Load all processed chunks into Pinecone (dense + sparse indexes), for the backend comparison.

Usage: uv run python -m retrieval.index_pinecone

Needs PINECONE_API_KEY in .env. Dense vectors come from the embedding cache that
retrieval.index filled, so nothing is re-embedded. Re-running overwrites records by chunk_id.
Free Starter plan: a full upload is ~0.3M of the 2M monthly write units.
"""

import logging
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone

from retrieval.embeddings import Embedder
from retrieval.index import load_chunks
from retrieval.pinecone_store import PineconeStore, create_indexes
from retrieval.sparse import SparseEmbedder, with_idf

log = logging.getLogger(__name__)

BATCH = 1000


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("pinecone").setLevel(logging.WARNING)
    load_dotenv()
    embedder = Embedder()
    chunks = load_chunks()
    log.info("Loaded %d chunks; computing BM25 vectors and IDF", len(chunks))
    # IDF needs the whole corpus, so every sparse vector is built before any upload.
    sparse = with_idf(SparseEmbedder().embed_documents([c.text for c in chunks]))

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    dense_index, sparse_index = create_indexes(pc, embedder.dim)
    store = PineconeStore(dense_index, sparse_index)

    start = time.monotonic()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vectors = embedder.embed_documents([c.text for c in batch])
        store.upsert(batch, vectors, sparse[i : i + BATCH])
        log.info("%d/%d chunks (%.0f s)", i + len(batch), len(chunks), time.monotonic() - start)

    # Counts are eventually consistent; wait until both indexes report every chunk.
    for _ in range(60):
        counts = [
            ix.describe_index_stats().total_vector_count for ix in (dense_index, sparse_index)
        ]
        if counts == [len(chunks)] * 2:
            break
        time.sleep(5)
    minutes = (time.monotonic() - start) / 60
    log.info("Done in %.1f min. Dense / sparse records: %s", minutes, counts)


if __name__ == "__main__":
    main()

"""Embed all processed chunks and load them into Qdrant.

Usage: uv run python -m retrieval.index [--recreate]

Embeddings are cached on disk, so re-running only embeds new or changed chunks.
"""

import argparse
import logging
import os
import time
from pathlib import Path

from qdrant_client import QdrantClient

from ingest.models import Chunk
from retrieval.embeddings import Embedder
from retrieval.qdrant_store import QdrantStore

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
COLLECTION = "filings"
BATCH = 1024


def load_chunks(processed_dir: Path = PROCESSED_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(processed_dir.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as f:
            chunks += [Chunk.model_validate_json(line) for line in f]
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recreate", action="store_true", help="Drop the collection first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    if args.recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    embedder = Embedder()
    store = QdrantStore(client, COLLECTION, embedder.dim)
    chunks = load_chunks()
    log.info("Loaded %d chunks", len(chunks))

    start = time.monotonic()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        store.upsert(batch, embedder.embed_documents([c.text for c in batch]))
        done = i + len(batch)
        rate = done / (time.monotonic() - start)
        log.info(
            "%d/%d chunks (%.0f/s, ~%.0f min left)",
            done,
            len(chunks),
            rate,
            (len(chunks) - done) / rate / 60,
        )

    count = client.count(COLLECTION).count
    log.info(
        "Done in %.1f min. Collection '%s' has %d points.",
        (time.monotonic() - start) / 60,
        COLLECTION,
        count,
    )


if __name__ == "__main__":
    main()

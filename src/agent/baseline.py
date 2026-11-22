"""Phase 1 baseline: vector search top-k, then an LLM answer with citations.

Usage: uv run python -m agent.baseline "How much did Apple spend on R&D in fiscal 2026?"

No filters, no reranking, no prompt hardening: this is the number later phases must beat.
Retrieval is dense-only, or hybrid (dense + BM25, rank fusion) when given a sparse embedder.
"""

import os
import re
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from llm_gateway.client import NO_THINKING, LLMClient
from retrieval.base import SearchResult, VectorStore
from retrieval.embeddings import Embedder
from retrieval.qdrant_store import QdrantStore
from retrieval.rerank import CANDIDATES, Reranker
from retrieval.sparse import SparseEmbedder

TOP_K = 5
COLLECTION = "filings"
NOT_FOUND = "The provided filings do not contain this information."

SYSTEM_PROMPT = f"""You answer questions about companies using SEC filing excerpts.
Rules:
- Use only the numbered sources. Do not use outside knowledge.
- Cite every claim with its source number in brackets, like [2]. Cite only sources you used.
- Give exact figures with units and the fiscal period they cover.
- If the sources do not answer the question, reply exactly: "{NOT_FOUND}"
- Be concise."""

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[SearchResult]
    cited: list[int] = field(default_factory=list)  # 1-based source numbers used in the text
    input_tokens: int = 0
    output_tokens: int = 0


def source_label(result: SearchResult) -> str:
    c = result.chunk
    period = f"FY{c.fiscal_year} {c.fiscal_period}" if c.fiscal_year else str(c.period_of_report)
    page = f", p.{c.page}" if c.page else ""
    return f"{c.company} {c.form_type} {period}, {c.section_id} ({c.section_title}){page}"


def build_prompt(question: str, results: list[SearchResult]) -> str:
    blocks = [f"[{i}] {source_label(r)}\n{r.chunk.text}" for i, r in enumerate(results, 1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Source numbers cited in the answer, in order, ignoring numbers with no matching source."""
    seen: list[int] = []
    for m in CITATION_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


class BaselineRAG:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLMClient,
        k: int = TOP_K,
        sparse: SparseEmbedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.store, self.embedder, self.llm, self.k = store, embedder, llm, k
        self.sparse, self.reranker = sparse, reranker

    @property
    def retrieval(self) -> str:
        name = "hybrid" if self.sparse else "vector"
        return f"{name}+{self.reranker.name}" if self.reranker else name

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        k = k or self.k
        n = max(k, CANDIDATES) if self.reranker else k
        dense = self.embedder.embed_query(question)
        if self.sparse is None:
            results = self.store.search(dense, n)
        else:
            results = self.store.hybrid_search(dense, self.sparse.embed_query(question), n)
        return self.reranker.rerank(question, results, k) if self.reranker else results

    def answer(self, question: str) -> Answer:
        return self.generate(question, self.retrieve(question))

    def generate(self, question: str, results: list[SearchResult]) -> Answer:
        resp = self.llm.chat(SYSTEM_PROMPT, build_prompt(question, results))
        return Answer(
            question=question,
            text=resp.text,
            sources=results,
            cited=parse_citations(resp.text, len(results)),
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )


def from_env(retrieval: str = "hybrid", rerank: str | None = "nvidia") -> BaselineRAG:
    """Defaults are the Phase 2 winner (docs/decisions.md); vector/no rerank is the baseline."""
    load_dotenv()
    embedder = Embedder()
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    llm = LLMClient(extra_body=NO_THINKING)
    sparse = SparseEmbedder() if retrieval == "hybrid" else None
    store = QdrantStore(client, COLLECTION, embedder.dim)
    reranker = Reranker(rerank) if rerank else None
    return BaselineRAG(store, embedder, llm, sparse=sparse, reranker=reranker)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How much did Apple spend on R&D in fiscal 2026?"
    ans = from_env().answer(question)
    print(f"Q: {ans.question}\n\n{ans.text}\n\nSources:")
    for i, r in enumerate(ans.sources, 1):
        mark = "*" if i in ans.cited else " "
        print(f" {mark}[{i}] {source_label(r)}  (score {r.score:.3f})")
    print(f"\n(* = cited) tokens in/out: {ans.input_tokens}/{ans.output_tokens}")


if __name__ == "__main__":
    main()

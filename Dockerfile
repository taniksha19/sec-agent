# FastAPI service over the Phase 3 agent. Built and run by `docker compose up`.
FROM python:3.12.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONPATH=/app/src PATH="/app/.venv/bin:$PATH"

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data/companies.csv ./data/companies.csv

# Code and packages stay root-owned; the app user can only write the cache.
RUN useradd --create-home app && mkdir -p /app/data/cache && chown app /app/data/cache
USER app

# Download both embedding models at build time, not on the first request.
RUN python -c "from retrieval.embeddings import Embedder; from retrieval.sparse import SparseEmbedder; \
Embedder().model; SparseEmbedder().model"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

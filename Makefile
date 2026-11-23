.PHONY: install up down download test test-gateway lint eval api

install:
	uv sync

up:
	docker compose up -d

down:
	docker compose down

download:
	uv run python -m ingest.download

test:
	uv run pytest

# Go isn't needed locally: format check, vet and test inside the Go image.
test-gateway:
	docker run --rm -v "$(CURDIR)/gateway:/src" -w /src golang:1.26.11-alpine sh check.sh

lint:
	uv run ruff check src tests evals
	uv run ruff format --check src tests evals

eval:
	uv run python -m evals.run --all

api:
	uv run uvicorn api.app:app --port 8000

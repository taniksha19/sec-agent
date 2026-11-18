.PHONY: install up down download test lint

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

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

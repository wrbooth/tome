# CLAUDE.md

## Commands

```bash
uv run pytest tests/ -v                          # Run all unit tests
uv run pytest tests/ --cov=. --cov-report=term-missing  # Tests with coverage
uv run pytest tests/test_ingest_pure.py -v       # Run single test file

uv run python ingest.py <file> --title "..."     # Ingest a document
uv run python batch_reingest.py --all --clear-first  # Full reingest
uv run python search.py --q "query" --k 20       # Search
uv run python documents.py list                  # List documents
uv run python embed.py --provider openai         # Generate embeddings
uv run python test_search.py --serial            # Integration search tests
```

## What This Is

A historical document indexing and search system. Ingests PDFs/TXT files, chunks them with heading-aware splitting, stores in PostgreSQL (pgvector) + Meilisearch, and provides hybrid search (semantic + keyword) with cross-encoder reranking and LLM answer generation.

## Project Structure

```
*.py              # Core modules (ingest, search, embed, documents, config, etc.)
api/main.py       # FastAPI wrapper around search
sql/schema.sql    # PostgreSQL schema (documents, passages, entities, years)
tests/            # pytest unit tests (mocked, no external services needed)
test_*.py         # Integration tests at root (require live services)
data/             # Source documents for ingestion
```

## Conventions

- **Package management**: use `uv`, not pip or poetry
- **Testing**: pytest with `unittest.mock` for external services. Tests must run without Postgres/Meilisearch/OpenAI
- `answer_generator.py` calls `get_openai_client()` at import time (line 11) -- tests must patch `answer_generator.client` directly
- `ingest.py` loads spaCy at import time (line 31-35) -- patch `ingest.nlp` if needed

## External Services

Requires Docker Compose services to be running for ingestion/search:
- **PostgreSQL 16 + pgvector** (port 5432) -- passages, embeddings, entities
- **Meilisearch** (port 7700) -- hybrid keyword+semantic search
- **OpenAI API** -- embeddings (text-embedding-3-small), query analysis (gpt-4o-mini), answers (gpt-5-mini)

## Gotchas

- `is_quality_heading()` has 20+ regex filters that reject noise -- new heading patterns must pass all of them; check ordering matters (e.g. appendix check must come before coordinate filter)
- `chunk_text_with_headings()` only splits at paragraph boundaries (newlines), not mid-paragraph
- The existing `test_search.py` at root is an integration test requiring live services; `tests/test_search.py` is the unit test
- `merge_heading_results()` has a known bug: crashes with KeyError if first heading lacks a `confidence` key and a duplicate has one

# CLAUDE.md

## Commands

```bash
uv run pytest tests/unit/ -v                          # Run all unit tests
uv run pytest tests/unit/ --cov=src --cov-report=term-missing  # Tests with coverage
uv run pytest tests/unit/test_ingest_pure.py -v       # Run single test file
uv run pytest tests/integration/ -v                   # Run integration tests (require live services)

uv run tome-ingest <file> --title "..."              # Ingest a document
uv run tome-batch-reingest --all --clear-first       # Full reingest
uv run tome-search --q "query" --k 20                # Search
uv run tome-documents list                           # List documents

uv run uvicorn tome.api.app:app --reload             # Run API server
```

## What This Is

A historical document indexing and search system. Ingests PDFs/TXT files, chunks them with heading-aware splitting, stores in PostgreSQL + Meilisearch (which handles hybrid BM25 + vector search with auto-generated OpenAI embeddings), and provides cross-encoder reranking and LLM answer generation.

## Project Structure

```
src/tome/                  # Main package (src layout)
  config.py                 # Pydantic BaseSettings configuration
  models.py                 # Shared Pydantic data models
  documents.py              # Document management CLI + logic
  ingestion/                # Document ingestion pipeline
    ingest.py               # Main ingestion orchestrator + CLI
    batch_ingest.py         # Batch ingestion CLI
    batch_reingest.py       # Re-ingestion CLI
    chunking.py             # Text chunking with heading-aware splitting
    heading_detection.py    # Heading detection (regex, typography, TOC)
    pdf_extraction.py       # PDF/TXT text extraction
    entities.py             # NER entity extraction (spaCy)
    storage.py              # Postgres + Meilisearch storage
  search/                   # Search and retrieval
    search.py               # Hybrid search + reranking + CLI
    answer_generator.py     # LLM answer generation
  api/                      # FastAPI application
    app.py                  # App factory, middleware, SPA serving
    routes/
      search.py             # Search + streaming endpoints
      documents.py          # Document CRUD + stats endpoints
      upload.py             # Document upload + ingestion tasks
    static/                 # Frontend SPA assets
tests/
  unit/                     # Mocked unit tests (no external services)
  integration/              # Integration tests (require live services)
sql/schema.sql              # PostgreSQL schema
docs/                       # Design docs and guides
```

## Conventions

- **Package management**: use `uv`, not pip or poetry
- **Testing**: pytest with `unittest.mock` for external services. Tests must run without Postgres/Meilisearch/OpenAI
- **Configuration**: Pydantic `BaseSettings` in `tome.config.Settings`, loaded from env vars and `.env`
- `answer_generator.py` uses lazy loading via `_get_client()` -- tests patch `tome.search.answer_generator._get_client`
- `entities.py` uses lazy loading via `_get_nlp()` for spaCy -- tests patch `tome.ingestion.entities._get_nlp`

## External Services

Requires Docker Compose services to be running for ingestion/search:
- **PostgreSQL 16** (port 5432) -- passages, entities, document metadata
- **Meilisearch** (port 7700) -- hybrid BM25 + vector search (auto-generates embeddings via OpenAI)
- **OpenAI API** -- embeddings via Meilisearch (text-embedding-3-small), query analysis (gpt-4o-mini), answers (gpt-5-mini)

## Gotchas

- `is_quality_heading()` has 20+ regex filters that reject noise -- new heading patterns must pass all of them; check ordering matters (e.g. appendix check must come before coordinate filter)
- `chunk_text_with_headings()` only splits at paragraph boundaries (newlines), not mid-paragraph
- Integration tests in `tests/integration/` require live Postgres/Meilisearch services
- `merge_heading_results()` has a known bug: crashes with KeyError if first heading lacks a `confidence` key and a duplicate has one

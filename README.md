# Tome

A document indexing and search system. Ingests PDFs and text files, chunks them with heading-aware splitting, stores structured data in PostgreSQL and indexes passages in Meilisearch, then provides hybrid search (semantic + keyword) with cross-encoder reranking and LLM answer generation.

## Features

- **Document ingestion** -- PDF and TXT with heading-aware chunking
- **Hybrid search** -- BM25 + vector similarity via Meilisearch's built-in hybrid search
- **Cross-encoder reranking** -- precision improvement on retrieved results
- **LLM answer generation** -- synthesized answers with source citations
- **Entity extraction** -- named entities and years via spaCy NER
- **Query type detection** -- specialized handling for "who is/was" and factoid queries
- **Page-aware results** -- exact page numbers and citations
- **Web UI + CLI** -- FastAPI SPA and command-line tools

## Architecture

```mermaid
flowchart LR
    A[PDF / TXT] --> B[Text Extraction]
    B --> C[Heading-Aware Chunking]
    C --> D[Entity Extraction]
    D --> E[(PostgreSQL)]
    C --> F[(Meilisearch)]
    F -- auto-embeds via OpenAI --> F

    H[Query] --> I[Query Analysis]
    I --> F
    F -- hybrid BM25 + vector --> J[Candidates]
    J --> K[Cross-Encoder Rerank]
    K --> L[LLM Answer Generation]
    L --> M[Results + Citations]
```

## Prerequisites

- **Python** 3.10+
- **[uv](https://docs.astral.sh/uv/)** -- Python package manager
- **Docker** and **Docker Compose** -- for PostgreSQL and Meilisearch
- **OpenAI API key** -- for embeddings and answer generation

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd tome

# Install dependencies (including dev tools)
uv sync --all-groups
```

### 2. Start services

```bash
docker compose build
docker compose up -d
```

This starts:
- **PostgreSQL 16** on port 5432 (passages, entities, metadata)
- **Meilisearch** on port 7700 (hybrid BM25 + vector search with auto-generated OpenAI embeddings)
- **Tome API** on port 8000 (FastAPI with web UI)

> To run only the infrastructure (Postgres + Meilisearch) and develop locally, use `docker compose up -d postgres meili` and run the API with `uv run uvicorn tome.api.app:app --reload`.

### 3. Set up the database

```bash
psql postgresql://tome:tome@localhost:5432/tome -f sql/schema.sql
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Default database and Meilisearch settings connect to the Docker Compose services out of the box. See [Configuration](#configuration) for all options.

### 5. Ingest a document

```bash
uv run tome-ingest data/sample.pdf --title "My Document" --authors "Author Name" --pub-year 2020
```

Ingestion handles chunking, entity extraction, Meilisearch indexing, and embedding generation in one step.

### 6. Search

```bash
uv run tome-search --q "your search query" --k 20
```

### 7. Run the web UI (optional)

```bash
uv run uvicorn tome.api.app:app --reload
```

Then open http://localhost:8000 in your browser.

## CLI Reference

All commands are run via `uv run <command>`.

| Command | Description |
|---|---|
| `tome-ingest <file>` | Ingest a single document |
| `tome-batch-ingest <path>` | Batch ingest a directory of documents |
| `tome-batch-reingest --all --clear-first` | Full re-ingestion from scratch |
| `tome-search --q "query" --k 20` | Search across all documents |
| `tome-documents list` | List all documents |
| `tome-documents info <id>` | Get document details |
| `tome-documents delete <id>` | Delete a document |
| `tome-documents stats` | System statistics |

Run any command with `--help` for full options.

## Configuration

Configuration is managed via environment variables or a `.env` file. All settings are defined in `src/tome/config.py` using Pydantic `BaseSettings`.

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `tome` | Database name |
| `DB_USER` | `tome` | Database user |
| `DB_PASSWORD` | `tome` | Database password |
| `MEILI_URL` | `http://localhost:7700` | Meilisearch URL |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (used by Meilisearch's OpenAI embedder) |
| `QUERY_ANALYSIS_MODEL` | `gpt-4o-mini` | Model for query analysis |
| `ANSWER_MODEL` | `gpt-5-mini` | Model for answer generation |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |

## Project Structure

```
Dockerfile                   # API container build
docker-compose.yml           # Full stack (Postgres + Meilisearch + API)
src/tome/                   # Main package (src layout)
  config.py                  # Pydantic BaseSettings configuration
  models.py                  # Shared Pydantic data models
  documents.py               # Document management CLI + logic
  ingestion/                 # Document ingestion pipeline
    ingest.py                # Main ingestion orchestrator + CLI
    batch_ingest.py          # Batch ingestion CLI
    batch_reingest.py        # Re-ingestion CLI
    chunking.py              # Text chunking with heading-aware splitting
    heading_detection.py     # Heading detection (regex, typography, TOC)
    pdf_extraction.py        # PDF/TXT text extraction
    entities.py              # NER entity extraction (spaCy)
    storage.py               # Postgres + Meilisearch storage
  search/                    # Search and retrieval
    search.py                # Hybrid search + reranking + CLI
    answer_generator.py      # LLM answer generation
  api/                       # FastAPI application
    app.py                   # App factory, request logging, SPA serving
    routes/
      search.py              # Search + streaming endpoints
      documents.py           # Document CRUD + stats endpoints
      upload.py              # Document upload + ingestion tasks
    static/                  # Frontend SPA assets
tests/
  unit/                      # Mocked unit tests (no external services)
  integration/               # Integration tests (require live services)
sql/schema.sql               # PostgreSQL schema
docs/                        # Design docs and guides
```

## Development

### Running tests

```bash
# Unit tests (no external services required)
uv run pytest tests/unit/ -v

# Unit tests with coverage
uv run pytest tests/unit/ --cov=src --cov-report=term-missing

# Integration tests (requires running Docker services)
uv run pytest tests/integration/ -v
```

### RAG evaluation harness

`tests/integration/test_search.py` is an end-to-end evaluation suite that measures both **retrieval quality** and **answer accuracy** against 20 curated questions with known expected pages and answers.

- **Retrieval quality** — grades where the expected page lands in ranked results (EXCELLENT at rank 1-3, down to FAILED if not found in top-k)
- **Answer accuracy** — uses an LLM judge to compare generated answers against expected answers, with a heuristic fallback when the judge is unavailable

```bash
# Run the full eval suite
uv run pytest tests/integration/test_search.py -v

# Run a single question by keyword
uv run pytest tests/integration/test_search.py -v -k "morgan"
```

### Linting

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
```

### Re-ingestion

When you need to start fresh (e.g., after schema changes or chunking improvements):

```bash
uv run tome-batch-reingest --all --clear-first
```

## Troubleshooting

| Problem | Fix |
|---|---|
| spaCy model not found | `uv run python -m spacy download en_core_web_sm` |
| OpenAI API errors | Check `OPENAI_API_KEY` in your `.env` |
| Cannot connect to Postgres | Verify `docker compose up -d` is running |
| Cannot connect to Meilisearch | Verify `docker compose up -d` is running |
| Stale search index | Run `uv run tome-batch-reingest --all --clear-first` |

## License

See individual component licenses for details.

# Codex Modernization Plan

## High Priority

### 1. Fix Embedding Model Mismatch
**Status:** Complete

Standardized everything on `text-embedding-3-small` (1536 dimensions):
- `embed.py:30` — function default changed from `text-embedding-3-large` to `text-embedding-3-small`
- `embed.py:117` — validation dimension changed from 1024 to 1536
- `sql/schema.sql:21` — column definition changed from `vector(768)` to `vector(1536)`
- Live database — altered column to `vector(1536)` and rebuilt ivfflat index
- CLI default (`embed.py:124`) and query embedding (`search.py:373`) were already correct

### 2. Fix Meilisearch BM25 Returning Zero Results
**Status:** Complete

Two issues found:
1. **Client version mismatch:** `meilisearch` Python client v0.20.0 was incompatible with Meilisearch server v1.18.0. The client's Pydantic models couldn't parse the server's responses (e.g., `taskUid` returned as int but expected as string). The `add_documents()` call silently failed. Upgraded client to v0.40.0.
2. **Primary key ambiguity:** Documents sent to Meilisearch contain both `id` and `document_id` fields. Newer Meilisearch versions refuse to auto-detect the primary key when multiple `*id` fields exist. Added explicit `primary_key="id"` to `add_documents()` in `ingest.py:583` and `documents.py:198`.

### 3. Consolidate LLM Calls with Structured Output
**Status:** Complete

Replaced 4 separate LLM calls (`detect_query_type`, `extract_person_from_query`, `extract_entities_from_query`, `generate_query_expansions`) with a single `analyze_query()` function that uses OpenAI structured outputs (`response_format` with `json_schema`). Also eliminated the duplicate `detect_query_type` call inside `apply_query_boosting()` — it now receives the pre-computed analysis result.

Changes:
- `search.py`: Removed 4 old functions, added `analyze_query()` with strict JSON schema. Updated `apply_query_boosting()` to accept `query_analysis` dict instead of raw query string. Updated `main()` to call `analyze_query()` once and pass the result through.
- `api/main.py`: Updated to import `analyze_query` instead of `detect_query_type`, pass analysis result to `apply_query_boosting()`.

Per-query LLM calls reduced from 6+ to 3 (query analysis, query embedding, answer generation).

### 4. Unify Dependency Management
**Status:** Complete

Migrated from Poetry to uv. Removed conflicting dependency specs.

Changes:
- Deleted `requirements.txt`, `api/requirements.txt`, and `poetry.lock`
- Rewrote `pyproject.toml`: loosened version constraints, switched build-backend from poetry-core to hatchling
- Generated `uv.lock` for reproducible installs
- Updated `api/Dockerfile` to use uv (multi-stage copy from `ghcr.io/astral-sh/uv`)
- Added `.venv/` to `.gitignore`
- `uv sync` upgraded many stale packages (pymupdf, spacy, transformers, fastapi, openai, etc.)

## Medium Priority

### 5. Replace Manual Boosting with Cross-Encoder Re-Ranker
**Status:** Not started

`apply_query_boosting()` (`search.py:383-612`) issues individual SQL queries per candidate per entity for re-ranking using hardcoded multiplicative boost factors. This is slow, brittle, and the boost values are hand-tuned.

**Changes needed:**
- Replace the boosting logic with a cross-encoder re-ranker (e.g., Cohere Rerank API, or a local model like `ms-marco-MiniLM`)
- Remove the entity-based boosting code and its associated LLM calls
- The `passage_entities` and `passage_years` tables can still be useful for filtering, but shouldn't drive scoring

### 6. Use Meilisearch Native Hybrid Search
**Status:** Not started

The codebase manually implements RRF fusion between Meilisearch BM25 results and pgvector cosine similarity results. It also generates LLM-based query expansions and feeds each back into Meilisearch individually. Meilisearch now has a native hybrid search mode that handles this internally.

**Changes needed:**
- Evaluate Meilisearch's built-in hybrid search capabilities
- Replace manual RRF fusion code if Meilisearch hybrid covers the use case
- Remove or simplify `generate_query_expansions_openai()` if no longer needed
- Consider whether pgvector is still needed or if Meilisearch can handle both lexical and semantic search

### 7. Extract Shared Modules
**Status:** Not started

`get_db_connection()` is duplicated identically across 7 files. Every module independently calls `load_dotenv()` — `search.py` calls it 6 times in different functions. The Meilisearch URL env var is inconsistently named (`MEILI_URL` vs `MEILISEARCH_URL`). The OpenAI client is instantiated independently in multiple places.

**Changes needed:**
- Create a `config.py` module that loads env once and exposes configuration
- Create a `db.py` module with the shared `get_db_connection()` function
- Standardize the Meilisearch URL env var name
- Create a shared OpenAI client singleton

## Low Priority

### 8. Use Tiktoken for Chunking
**Status:** Not started

Chunking estimates tokens as `word_count * 1.3` (`ingest.py:396`). `tiktoken` is already a dependency (imported in `embed.py`) but never used for chunk size calculation.

**Changes needed:**
- Replace the `word_count * 1.3` heuristic with actual `tiktoken` token counting in `chunk_text_with_headings()`
- Consider semantic/recursive chunking strategies that respect section boundaries
- Evaluate whether the current `max_tokens=300` limit is appropriate

### 9. Refactor API to Share Code with CLI
**Status:** Not started

`api/main.py` reimplements the search pipeline but skips answer generation and query expansion. It adds `..` to `sys.path` to import from the parent directory. The API and CLI should share the same core logic.

**Changes needed:**
- Extract core search logic into a shared module (e.g., `search_engine.py`)
- Have both `api/main.py` and `search.py` CLI call into the shared module
- Add answer generation to the API
- Remove the `sys.path` hack in favor of proper package structure

### 10. Refactor Subprocess Orchestration to Direct Imports
**Status:** Not started

`batch_reingest.py`, `test_search.py`, and others call scripts via `subprocess.run()` rather than importing functions. This loses error details (e.g., the empty stderr from the embedding failure).

**Changes needed:**
- Refactor `batch_reingest.py` to import and call functions from `ingest.py` and `embed.py` directly
- Refactor `test_search.py` to import search functions instead of running `search.py` as a subprocess
- Reserve subprocess usage for true process isolation needs

### 11. Centralize Model Configuration
**Status:** Not started

Model names are hardcoded in multiple places with different defaults:
- `gpt-4o-mini` in `search.py` (4 places)
- `gpt-4` as a default in `answer_generator.py:42`
- `gpt-5-mini-2025-08-07` as the actual default at `answer_generator.py:186`
- `text-embedding-3-small` / `text-embedding-3-large` inconsistency in `embed.py` and `search.py`

**Changes needed:**
- Define all model names in `config.py` or as environment variables
- Reference the centralized config throughout the codebase
- Make it easy to swap models without editing multiple files

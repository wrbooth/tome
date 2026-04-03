# CLAUDE.md

## Commands

```bash
uv run pytest tests/unit/ -v                          # Unit tests (no external services)
uv run pytest tests/unit/ --cov=src --cov-report=term-missing  # Tests with coverage
uv run pytest tests/integration/ -v                   # Integration tests (require Docker services)
uv run ruff check src/ tests/                         # Lint
uv run ruff format --check src/ tests/                # Format check
uv run mypy src/                                      # Type check
uv run uvicorn tome.api.app:app --reload              # Run API server
```

## Conventions

- Use `uv`, not pip or poetry
- Unit tests must run without Postgres/Meilisearch/OpenAI — mock all external services
- Lazy-loaded clients have specific patch targets:
  - `tome.search.answer_generator._get_client` (OpenAI)
  - `tome.ingestion.entities._get_nlp` (spaCy)

## Gotchas

- `is_quality_heading()` has 20+ regex filters that reject noise — new heading patterns must pass all of them; ordering matters (e.g. appendix check must come before coordinate filter)
- `chunk_text_with_headings()` only splits at paragraph boundaries (newlines), not mid-paragraph
- `merge_heading_results()` has a known bug: crashes with KeyError if first heading lacks a `confidence` key and a duplicate has one

## Verification

After making changes, always run:
```bash
uv run pytest tests/unit/ -v && uv run ruff check src/ tests/
```

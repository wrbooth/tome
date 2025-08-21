# Codex — Indexing & Search POC

A proof-of-concept system for ingesting documents, chunking them correctly, indexing them (lexical + vector), and retrieving relevant passages with page-aware provenance.

## Features

- **Document Ingestion**: PDF and TXT file support with text extraction
- **Smart Chunking**: 800-1200 token chunks with page boundary respect
- **Hybrid Search**: BM25 (Meilisearch) + Vector similarity (pgvector) with RRF fusion
- **Entity Extraction**: Named entities and years using spaCy NER
- **Query Type Detection**: Specialized handling for "who is/was" and factoid queries
- **Page-Aware Results**: Exact page numbers and citations

## Quick Start

### 1. Start Services

```bash
# Start Postgres (pgvector), Meilisearch, and optional API
docker compose up -d
```

### 2. Set Up Database

```bash
# Create schema
psql postgresql://codex:codex@localhost:5432/codex -f sql/schema.sql
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 4. Set Environment Variables

Create a `.env` file:

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=codex
DB_USER=codex
DB_PASSWORD=codex

# Meilisearch
MEILI_URL=http://localhost:7700

# OpenAI (optional, for embeddings)
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Ingest a Document

```bash
# Ingest a PDF
python ingest.py data/federalist.pdf --title "The Federalist Papers" --authors "Alexander Hamilton,James Madison,John Jay" --pub-year 1788

# Ingest a TXT file
python ingest.py data/constitution.txt --title "U.S. Constitution" --pub-year 1787
```

### 6. Generate Embeddings

```bash
# Using OpenAI (requires API key)
python embed.py --provider openai --model text-embedding-3-large

# Using local model (fallback)
python embed.py --provider local --model intfloat/e5-base-v2
```

### 7. Search

```bash
# Basic search
python search.py --q "factions in Federalist No. 10" --k 20

# Person query
python search.py --q "who was Publius" --k 10

# Filter by document
python search.py --q "separation of powers" --doc your_document_id --k 15
```

## Architecture

### Services

- **Postgres + pgvector**: Metadata storage and vector similarity search
- **Meilisearch**: BM25 lexical search and faceting
- **Optional FastAPI**: HTTP interface for search

### Data Flow

1. **Ingestion**: PDF/TXT → Text extraction → Chunking → Database storage
2. **Indexing**: Entity extraction → Meilisearch indexing
3. **Embedding**: OpenAI/local model → pgvector storage
4. **Search**: Query → BM25 + Vector → RRF fusion → Results

## CLI Commands

### `ingest.py`

Ingest documents and create passages.

```bash
python ingest.py <file_path> [options]

Options:
  --title TEXT      Document title
  --authors TEXT    Comma-separated list of authors
  --pub-year INT    Publication year
```

### `embed.py`

Generate embeddings for passages.

```bash
python embed.py [options]

Options:
  --provider [openai|local]  Embedding provider
  --model TEXT               Model name/path
  --batch-size INT           Batch size for processing
  --limit INT                Maximum passages to process
```

### `search.py`

Search passages using hybrid retrieval.

```bash
python search.py [options]

Options:
  --q TEXT         Search query (required)
  --k INT          Number of results (default: 20)
  --doc TEXT       Filter by document ID
```

## Search Features

### Query Type Detection

- **"who is/was" queries**: Boost passages with matching PERSON entities
- **Factoid queries**: Special handling for place/time queries
- **General queries**: Standard hybrid search

### RRF Fusion

Reciprocal Rank Fusion combines BM25 and vector similarity:

```
score = 1 / (k + rank)
k = 60
```

### Entity Boosting

For "who is/was X" queries, passages containing PERSON entity "X" get a 1.5x score boost.

## Database Schema

### Core Tables

- `documents`: Document metadata
- `passages`: Text chunks with embeddings
- `passage_entities`: Named entities per passage
- `passage_years`: Years mentioned in passages
- `places`: Place normalization data

### Indexes

- Full-text search on passage text
- Vector similarity on embeddings
- Entity and year filtering

## Testing

### Acceptance Tests

Use the seed corpus for testing:

1. **The Federalist Papers** (Project Gutenberg)
2. **U.S. Constitution** (plain text)

### Gold Queries

1. "What does Federalist No. 10 argue about factions?"
2. "Separation of powers in the Federalist"
3. "Necessary and Proper Clause"
4. "Who was Publius?"
5. "What taverns existed in Athens in the first 10 years of its founding?"

### Success Criteria

- ≥70% of queries have gold passage in top-5 results
- Correct page numbers in citations

## Development

### Project Structure

```
.
├── docker-compose.yml    # Service definitions
├── sql/
│   └── schema.sql       # Database schema
├── data/                # Document storage
├── ingest.py           # Document ingestion
├── embed.py            # Embedding generation
├── search.py           # Search interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Quality Checks

- Row parity: Meilisearch docs == Postgres passages
- Chunk quality: <1% chunks <20 words
- Embedding coverage: ≥99% passages have embeddings

## Troubleshooting

### Common Issues

1. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **OpenAI API errors**: Check API key in `.env` file
3. **Database connection**: Ensure Postgres is running on port 5432
4. **Meilisearch connection**: Check if service is running on port 7700

### Diagnostics

```bash
# Check database connection
psql postgresql://codex:codex@localhost:5432/codex -c "SELECT COUNT(*) FROM passages;"

# Check Meilisearch
curl http://localhost:7700/health

# Check embedding coverage
psql postgresql://codex:codex@localhost:5432/codex -c "SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL;"
```

## Next Steps

After POC validation:

1. Add OCR support for scanned documents
2. Implement reranker for precision improvement
3. Add footnote-aware chunking
4. Expand entity extraction and normalization
5. Build web interface
6. Add caching and performance optimizations

## License

This is a proof-of-concept implementation. See individual component licenses for details.









# Codex — Indexing & Search POC (no-frontend)

**Objective:** Prove we can ingest a document, chunk it correctly, index it (lexical + vector), and retrieve relevant passages with page-aware provenance. No frontend, queues, or caching.

---

## Scope (strict)

- Inputs: **PDF (text-based)** and **TXT** only (skip OCR for POC).
- Outputs: Postgres rows in `documents`, `passages`; Meilisearch index `passages`; pgvector embeddings in `passages.embedding`.
- Interfaces: **CLI scripts** only.
- Success: Given a seed corpus and queries, the top-5 results include the gold passage for ≥70% of queries, with correct page numbers.

---

## Minimal stack (docker compose)

```yaml
version: "3.9"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: codex
      POSTGRES_PASSWORD: codex
      POSTGRES_DB: codex
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]

  meili:
    image: getmeili/meilisearch:latest
    environment:
      MEILI_NO_ANALYTICS: "true"
    command: ["meilisearch", "--env", "development"]
    ports: ["7700:7700"]

  api: # optional for POC; used only to expose /search if you want HTTP
    build: ./api
    environment:
      DATABASE_URL: postgresql://codex:codex@postgres:5432/codex
      MEILI_URL: http://meili:7700
    depends_on: [postgres, meili]
    ports: ["8000:8000"]

volumes:
  pgdata:
```

> Omit MinIO/Redis for this POC. Store normalized JSONL files on local disk under `./data/normalized/`.

---

## DB schema (minimal)

```sql
create extension if not exists vector;

create table if not exists documents (
  id uuid primary key,
  title text not null,
  authors text[],
  pub_year int,
  language text,
  source_path text not null
);

create table if not exists passages (
  id uuid primary key,
  document_id uuid references documents(id) on delete cascade,
  page int not null,
  block_id text,
  char_start int,
  char_end int,
  headings_path text[],
  text text not null,
  embedding vector(1024)
);

create index if not exists idx_passages_tsv on passages using gin (to_tsvector('simple', text));
create index if not exists idx_passages_vec on passages using ivfflat (embedding vector_cosine_ops) with (lists = 100);
```

### Optional tables for entity & year extraction

```sql
create table if not exists passage_entities (
  passage_id uuid references passages(id) on delete cascade,
  entity text,
  ent_type text,            -- PERSON, ORG, GPE, FAC, etc.
  norm_entity text,
  primary key (passage_id, entity)
);

create table if not exists passage_years (
  passage_id uuid references passages(id) on delete cascade,
  year int,
  primary key (passage_id, year)
);

create table if not exists places (
  place_key text primary key,
  display_name text,
  country text,
  region text,
  founding_year int
);
```

---

## Meilisearch index

Index name: `passages`

Primary key: `id`

Searchable attributes: `text`, `headings_path`

Filterable attributes: `document_id`, `page`, `years`, `entities_person`, `entities_place`

Sortable attributes: `page`

---

## CLI commands (single-process scripts)

### 1) `ingest.py`

- Inputs: path(s) to PDF/TXT.
- Steps:
  1. Extract text with page numbers (PyMuPDF for PDF, plain read for TXT; enforce text-based PDFs only).
  2. **Chunking**: split by paragraphs; merge to 800–1,200 tokens, respect page boundaries; 15% overlap across page joins only.
  3. Write `documents` and `passages` rows.
  4. Run entity + year extraction (spaCy NER + regex for years); write into `passage_entities` and `passage_years`.
  5. Push documents to Meilisearch with `{id, text, page, document_id, headings_path, entities, years}`.

### 2) `embed.py`

- Inputs: none (reads unembedded passages from DB in batches).
- Embeddings provider:
  - Default: OpenAI text-embedding-3-large **or** local `intfloat/e5-base-v2` as fallback.
- Writes normalized vectors to `passages.embedding`.

### 3) `search.py`

- Inputs: `--q "query string" [--k 20] [--doc document_id]`.
- Steps:
  1. Detect query type: "who is/was" vs factoid vs general.
  2. Meili BM25: top K (e.g., 200) → `(id, rank)`.
  3. Dense: embed `q` → pgvector ANN top K (e.g., 200) → `(id, rank)`.
  4. **RRF fusion** with `k=60` → take top 20.
  5. If query matches `who is/was X`, boost passages with PERSON entity = X and definitional patterns (`X was`, `, a `, `, the `).
  6. If query matches pattern like `what taverns.*Athens.*first 10 years`, then:
     - Disambiguate Athens via `places` table (or prompt if multiple).
     - Build synonym-expanded query (tavern|inn|public house|alehouse|ordinary|hostelry).
     - Filter by `years` in [founding, founding+10].
  7. Print table: `rank, score, title, page, snippet`.

---

## Reference chunker (rules)

- Max tokens ≈ 1,000 (count words as proxy for tokens at POC).
- Start new chunk on heading or large whitespace gap.
- Do **not** split mid-sentence; join paragraphs until size limit.
- Keep `page` as the page of the **first** paragraph in chunk; if chunk spans pages, note a `headings_path` entry like `{"spans_pages": true}` for later refinement.

---

## RRF (reciprocal rank fusion)

```
score = 1 / (k + rank)
# ranks start at 1
k = 60
```

Fuse BM25 and vector lists by summing scores per `passage_id`.

---

## Acceptance tests

### Seed corpus (text-based, public domain)

- **The Federalist Papers** (Project Gutenberg plain text)
- **U.S. Constitution** (plain text)

### Gold queries (examples)

1. "What does Federalist No. 10 argue about factions?"
2. "Separation of powers in the Federalist"
3. "Necessary and Proper Clause"
4. "Who was Publius?" (test person entity + definitional boost)
5. "What taverns existed in Athens in the first 10 years of its founding?" (test place+year filter and synonyms)

### Criteria

- For each query, top-5 contains a chunk quoting the relevant section, with the correct page number (or section if TXT; emulate page as logical section for TXT).

---

## Diagnostics & correctness checks

- **Row parity**: count of Meili docs == count of `passages` rows.
- **Empty/very short chunks**: fail the run if >1% of chunks < 20 words.
- **Embedding coverage**: `% passages.embedding is not null` ≥ 99%.
- **Manual spot-check**: `search.py --q "factions"` prints snippets and page numbers; cross-check against source PDF.

---

## Example: `search.py` output

```
#> python search.py --q "factions in Federalist No. 10" --k 10
1  0.713  Federalist Papers  p. 45  "The latent causes of faction are thus sown in the nature of man…"
2  0.611  Federalist Papers  p. 46  "A zeal for different opinions concerning religion, concerning government…"
...
```

---

## Optional HTTP shim (later)

If needed, expose a simple FastAPI `/search?q=` that wraps `search.py` logic. Not required to pass POC.

---

## README quickstart (copy-paste)

```bash
# 0) start services
docker compose up -d

# 1) create schema
psql postgresql://codex:codex@localhost:5432/codex -f sql/schema.sql

# 2) ingest a document
python ingest.py data/federalist.pdf

# 3) embed passages
python embed.py --provider openai --model text-embedding-3-large --batch 128
#   or:  python embed.py --provider local --model intfloat/e5-base-v2

# 4) search
python search.py --q "factions in Federalist No. 10" --k 20
```

---

## Risks

- If PDFs are scanned images, extraction fails; choose text-based sources for POC.
- Token counting approximation may under/over split; acceptable for POC.
- Embedding provider changes vector dimension; validate schema before insert.
- Entity + year extraction is crude at POC stage; accuracy improves with later refinement.

---

## Next step after POC

- Add OCR path (OCRmyPDF) and language detection
- Add reranker (cross-encoder) for precision @ top-5
- Add footnote-aware chunking and figures indexing
- Expand gazetteer and controlled vocabulary for richer queries


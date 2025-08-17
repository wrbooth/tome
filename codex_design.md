# Codex — Local Containerized Design

A local first, containerized system for uploading source texts and making them searchable with natural language queries while preserving academic provenance.

## Goals

1. Make primary sources searchable with natural language while keeping exact citations and page links
2. Run locally in containers so we can validate the approach without cloud lock in
3. Keep the pipeline transparent and auditable for historians
4. Support multiple formats PDF, TXT, DOCX and scanned material with OCR

## Non goals for the first pass

* Full cloud automation
* Enterprise authentication and access tiers
* Large scale multimodal images beyond basic figure OCR

## Primary user stories

1. As a historian, I can upload a PDF scan of a book and receive page level search results that show the text snippet and a citation like p 214
2. As a researcher, I can ask a natural language question and get an answer that quotes exact lines with page numbers and links to the page viewer
3. As a power user, I can filter by author, publication year, language, and search only footnotes or only figures

## Architecture overview

```
[Browser]  →  [React (Vite) SPA]  →  [API gateway FastAPI]
                                 ↘
                               [Task queue Redis RQ]
                                 ↘
                         [Workers Python]
                 ┌─────────────┬─────────────┬─────────────┐
                 │             │             │             │
            [MinIO]       [Postgres]   [Meilisearch]   [Vector DB]
           object store     metadata      BM25 index     pgvector
```

Optional services for later

* OpenSearch for combined lexical and vector retrieval
* Qdrant or Weaviate as a separate vector service

## Components

### Model providers & configuration (proprietary OK)

* We can call proprietary model APIs from workers/API (never directly from the browser): OpenAI, Anthropic, or AWS Bedrock.
* Use them for: embeddings, reranking (cross-encoders or API-based rerankers), and the RAG reader.
* Provider selection by env vars, e.g. `PROVIDER=openai|anthropic|bedrock|local` with per-task overrides (`EMBEDDINGS_PROVIDER`, `RERANK_PROVIDER`, `LLM_PROVIDER`).
* Add a server-side proxy for all model calls to keep keys out of the client and to implement caching, rate-limit backoff, and circuit breakers.
* Keep a local fallback (e.g., `sentence-transformers` for embeddings, smaller local LLM for smoke tests) so the system remains usable offline.

### Frontend

* React (Vite) single-page app for upload, search, result previews, and a simple page viewer with text highlights
* Served from an Nginx container in production mode; dev mode can use Vite dev server

### API service

* FastAPI that exposes upload, search, and citation endpoints
* Streams retrieval results with Server Sent Events for progressive delivery

### Workers

* Python workers execute ingestion steps convert, OCR, parse, chunk, embed, index
* Each step writes artifacts to MinIO and metadata to Postgres

### Storage

* MinIO S3 compatible store for raw files, page images, and normalized JSONL
* Postgres with pgvector for metadata and embeddings
* Meilisearch for BM25 and light faceting

## Data model

### documents table

```sql
create table documents (
  id uuid primary key,
  work_id uuid,
  edition_id uuid,
  title text,
  authors text[],
  pub_year int,
  coverage_start int,
  coverage_end int,
  language text,
  source_url text,
  license text,
  pages_count int,
  created_at timestamp default now()
);
```

### files table

```sql
create table files (
  id uuid primary key,
  document_id uuid references documents(id),
  file_type text,
  s3_uri text,
  checksum text,
  ocr_quality numeric,
  created_at timestamp default now()
);
```

### passages table

```sql
create extension if not exists vector;

create table passages (
  id uuid primary key,
  document_id uuid references documents(id),
  page int,
  block_id text,
  char_start int,
  char_end int,
  headings_path text[],
  footnote_ids text[],
  bbox int[],
  lang text,
  shard int,
  hash text,
  text text,
  embedding vector(1024)
);
create index on passages using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index on passages using gin (to_tsvector('simple', text));
```

### search_features table optional

```sql
create table search_features (
  passage_id uuid primary key references passages(id),
  entities jsonb,
  places jsonb,
  dates jsonb,
  figure_ids text[]
);
```

## Ingestion pipeline

1. Detect file type and run conversion
   * PDF text extraction with coordinates and page numbers via pdfplumber or PyMuPDF
   * OCR for scanned pages with OCRmyPDF or Tesseract with language packs
   * DOCX to text with python docx
2. Normalize to JSONL with page id, blocks, bbox, footnote markers
3. Language detection per block with fasttext or langdetect
4. Chunking strategy
   * Split by headings then paragraphs
   * Merge until around one thousand tokens with about fifteen percent overlap across boundaries only if needed
   * Keep footnote text linked to its anchor either inline or as a sibling chunk with backlinks
   * Store breadcrumb path Work › Volume › Book › Chapter › Section in headings_path and optionally append a compact form to chunk text for extra signal
5. Embeddings
   * Use a multilingual model such as BGE m3 or E5 mistral class and store vectors in pgvector
6. Lexical index
   * Push passage text plus headings and captions into Meilisearch
7. Thesaurus
   * Maintain a light controlled vocabulary for historical spellings and Latinisms to support query expansion
8. Deduplication
   * MinHash shingles at passage level to collapse near duplicates across scans and editions

All steps are idempotent and recorded in Postgres so failed runs can resume

## Retrieval strategies

### Candidate generation

* Hybrid recall that merges BM25 from Meilisearch and vector similarity from pgvector using reciprocal rank fusion
* Optional SPLADE sparse representation later to improve recall for rare terms and spelling variants
* Fielded filters over author, pub year, language, edition, and a page range filter when useful

### Query rewriting

* HyDE a short synthetic answer to create an auxiliary embedding for recall on vague questions
* Synonym and spelling expansion from the controlled vocabulary
* Optional multi step decomposition for queries that imply a sequence for example person then location then date

### Reranking

* Cross encoder reranker applied to the top set of candidates for example top one hundred then keep the best forty
* Maximal marginal relevance to diversify and avoid near duplicate chunks

### Context assembly for RAG

* Fetch the parent page or section for each top passage to give the reader enough context without blowing the token budget
* The reader prompt must quote exact lines and include page numbers like p 214 and must refuse to answer without citations

### Footnotes and figures

* Index footnote chunks with a specific field so users can filter to footnotes only
* For figures and tables run OCR on captions store figure numbers and a small thumbnail in MinIO and allow a figures only filter

## Search API design

### Upload

```
POST  api v1 upload
body multipart file
resp document_id
```

### Search

```
GET  api v1 search
query q, filters, top_k
resp list of result items
```

Result item shape

```json
{
  "document_id": "…",
  "passage_id": "…",
  "title": "…",
  "headings_path": ["Work", "Ch 3"],
  "page": 214,
  "snippet": "…",
  "score": 0.82,
  "source": {
    "s3_uri": "s3 minio bucket…",
    "bbox": [x1, y1, x2, y2]
  }
}
```

### Answer

```
POST  api v1 answer
body q plus optional result set ids for targeted reading
resp structured answer with citations
```

## Example docker compose for local dev

```yaml
version: "3.9"
services:
  postgres:
    image: pgvector pgvector:pg16
    environment:
      POSTGRES_USER: hist
      POSTGRES_PASSWORD: hist
      POSTGRES_DB: hist
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  meilisearch:
    image: getmeili meilisearch:latest
    environment:
      MEILI_NO_ANALYTICS: "true"
    ports:
      - "7700:7700"
    command: ["meilisearch", "--env", "development"]

  minio:
    image: minio minio
    command: server /data
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio:/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  api:
    build: ./api
    environment:
      DATABASE_URL: postgresql hist hist hist postgres 5432 hist
      MEILI_URL: http meili localhost 7700
      MINIO_ENDPOINT: localhost
      MINIO_ACCESS_KEY: minio
      MINIO_SECRET_KEY: minio123
      REDIS_URL: redis 6379
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - meilisearch
      - minio
      - redis

  worker:
    build: ./worker
    environment:
      DATABASE_URL: postgresql hist hist hist postgres 5432 hist
      MEILI_URL: http meili localhost 7700
      MINIO_ENDPOINT: minio
      MINIO_ACCESS_KEY: minio
      MINIO_SECRET_KEY: minio123
      REDIS_URL: redis 6379
    depends_on:
      - postgres
      - meilisearch
      - minio
      - redis

  web:
    build: ./web
    environment:
      REACT_APP_API_URL: http api localhost 8000
    ports:
      - "3000:3000"
    depends_on:
      - api
volumes:
  pgdata:
  minio:
```

Note The image names and environment values above are placeholders replace with your builds The spaces in some URLs are placeholders for clarity set real URIs in your project

## Embedding and retrieval reference code sketches

### Compute embeddings in the worker

```python
from sentence_transformers import SentenceTransformer
import psycopg
import numpy as np

model = SentenceTransformer("intfloat e5 base v2")

with psycopg.connect(dsn) as conn:
  cur = conn.cursor()
  cur.execute("select id, text from passages where embedding is null limit 1000")
  rows = cur.fetchall()
  vecs = model.encode([r[1] for r in rows], normalize_embeddings=True)
  for (pid, _), v in zip(rows, vecs):
      cur.execute("update passages set embedding = %s where id = %s", (list(v), pid))
  conn.commit()
```

### Hybrid search with reciprocal rank fusion

```python
from meilisearch import Client as Meili
import psycopg

RRF_K = 60

def rrf(rank):
    return 1.0  (RRF_K + rank)

# get lexical candidates
def meili_candidates(q, k=200):
    # return list of (passage_id, rank)
    ...

# get vector candidates from pgvector
def vector_candidates(q_embed, k=200):
    # return list of (passage_id, rank)
    ...

# fuse
scores = defaultdict(float)
for i, pid in enumerate(meili_ids):
    scores[pid] += rrf(i)
for j, pid in enumerate(vector_ids):
    scores[pid] += rrf(j)

# take top N and rerank with a cross encoder
```

## RAG prompting guardrails

* The model answers only from provided passages
* The model quotes short lines and always adds page markers such as p 214
* If uncertain, the model returns top passages instead of an answer

Example prompt header

```
You are a careful research assistant for historians Answer only from the passages provided Quote exact lines and include page numbers in the form p N If no passage supports the claim say that and list the most relevant passages instead
```

## Evaluation plan

### Data set

* Build a seed set of one hundred to two hundred scholar style questions on a small collection with gold passages and page numbers

### Metrics

* Retrieval quality nDCG at ten and Recall at fifty
* Answer faithfulness checked by citation match rate and human review

### Workflow

* Nightly job runs the evaluation suite and writes a report per collection and per language

## Provenance and transparency

* Every chunk stores a back pointer to the page image bounding box so the UI can highlight the exact quote
* Show the edition in result cards and allow users to jump across editions of the same work
* Display OCR confidence and warn when the page is low quality

## Security and privacy

* All services run in an isolated docker network with only the API and web exposed
* MinIO has distinct buckets for uploads normalized text and derived artifacts
* Documents can be marked private public or shared within a group later when authentication is added

## Roadmap from local to production

1. Replace Meilisearch with OpenSearch or Elastic for stronger fielding and custom analyzers
2. Move storage from MinIO to S3 while keeping the same object layout to minimize code changes
3. Add a dedicated vector service Qdrant or Weaviate or keep pgvector if size remains moderate
4. Introduce a trained reranker and a small evaluator set to prevent quality regressions

## Risks and mitigations

* OCR noise use high quality OCR and surface confidence stats so users can judge reliability
* Hallucinations force citations in every answer and provide a show me tab with the raw passages
* Many duplicate scans and editions collapse with shingle based dedup and present an edition switcher in the UI

## Developer workflow

* Pre commit hook for text normalizer and JSONL schema validation
* Seed script that ingests a small public domain corpus so new developers can test end to end locally
* Make targets for compose up compose down migrate db and run tests

## Appendix controlled vocabulary example

```
colour → color
publick → public
antient → ancient
moscovia → moscow
```

## Appendix suggested libraries

* pdfplumber or PyMuPDF for text and layout extraction
* OCRmyPDF Tesseract for OCR
* unstructured or doctr for layout detection
* sentence transformers for embeddings
* Meilisearch python client
* psycopg pgvector
* FastAPI and Pydantic
* React (Vite) React and TipTap for a footnote aware page viewer


# Codex Multi-Document Support Guide

This guide covers the new multi-document capabilities added to Codex, including batch ingestion, document management, and enhanced search features.

## Overview

The multi-document support includes:

- **Batch Ingestion**: Process multiple documents at once
- **Document Management**: List, delete, and manage documents
- **Document-Aware Embedding**: Generate embeddings for specific documents
- **Enhanced Search**: Filter searches by document
- **Batch Re-ingestion**: Re-process documents with options

## New Scripts

### 1. Batch Ingestion (`batch_ingest.py`)

Process multiple documents at once with support for metadata files and parallel processing.

#### Basic Usage

```bash
# Ingest all PDF/TXT files in a directory
python batch_ingest.py data/

# Ingest with metadata file
python batch_ingest.py data/ --metadata data/metadata.json

# Recursive directory processing
python batch_ingest.py data/ --recursive

# Parallel processing (4 workers)
python batch_ingest.py data/ --parallel 4

# Save results to file
python batch_ingest.py data/ --output results.json
```

#### Metadata File Format

Create a JSON or CSV file with document metadata:

**JSON format** (`metadata.json`):
```json
[
  {
    "filename": "document1.pdf",
    "title": "Document Title",
    "authors": ["Author 1", "Author 2"],
    "pub_year": 2023,
    "language": "en"
  },
  {
    "filename": "document2.txt",
    "title": "Another Document",
    "authors": ["Single Author"],
    "pub_year": 2022,
    "language": "en"
  }
]
```

**CSV format** (`metadata.csv`):
```csv
filename,title,authors,pub_year,language
document1.pdf,Document Title,"Author 1,Author 2",2023,en
document2.txt,Another Document,Single Author,2022,en
```

### 2. Document Management (`documents.py`)

Manage documents in the system with various commands.

#### List Documents

```bash
# List all documents with stats
python documents.py list

# JSON output
python documents.py list --format json
```

#### Document Information

```bash
# Get detailed info about a document
python documents.py info <document_id>

# JSON output
python documents.py info <document_id> --format json
```

#### Delete Documents

```bash
# Delete with confirmation
python documents.py delete <document_id>

# Force delete (no confirmation)
python documents.py delete <document_id> --force
```

#### Re-index Documents

```bash
# Re-index a document in Meilisearch
python documents.py reindex <document_id>
```

#### System Statistics

```bash
# Get overall system stats
python documents.py stats
```

### 3. Enhanced Embedding (`embed.py`)

Generate embeddings with document-aware processing.

#### Basic Usage

```bash
# Generate embeddings for all passages
python embed.py

# Generate embeddings for specific document
python embed.py --doc <document_id>

# Generate embeddings for multiple documents
python embed.py --documents <doc_id1>,<doc_id2>,<doc_id3>

# Use local model instead of OpenAI
python embed.py --provider local --model intfloat/e5-base-v2
```

### 4. Batch Re-ingestion (`batch_reingest.py`)

Re-ingest documents with various options.

#### Basic Usage

```bash
# Re-ingest all documents
python batch_reingest.py --all

# Re-ingest specific documents
python batch_reingest.py <doc_id1> <doc_id2> <doc_id3>

# Clear everything first, then re-ingest all
python batch_reingest.py --all --clear-first

# Clear specific documents first
python batch_reingest.py --all --clear-docs <doc_id1>,<doc_id2>

# Skip embedding generation
python batch_reingest.py --all --skip-embedding

# Show debug information
python batch_reingest.py --all --debug
```

## Enhanced Search Features

### Document Filtering

All search commands now support document filtering:

```bash
# Search in specific document
python search.py --q "your query" --doc <document_id>

# API search with document filter
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "document_id": "<document_id>"}'
```

### API Endpoints

The FastAPI service includes new endpoints:

```bash
# List all documents
curl http://localhost:8000/documents

# Get system statistics
curl http://localhost:8000/stats

# Search with document filter
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "document_id": "<document_id>"}'
```

## Workflow Examples

### 1. Initial Setup with Multiple Documents

```bash
# 1. Ingest multiple documents
python batch_ingest.py data/ --metadata data/metadata.json --parallel 2

# 2. Generate embeddings for all documents
python embed.py

# 3. Verify documents are loaded
python documents.py list
```

### 2. Adding New Documents

```bash
# 1. Add new documents to data/ directory
# 2. Update metadata.json if needed
# 3. Ingest new documents
python batch_ingest.py data/new_document.pdf --metadata data/metadata.json

# 4. Generate embeddings for new document
python embed.py --doc <new_document_id>
```

### 3. Updating Existing Documents

```bash
# 1. Re-ingest specific document
python batch_reingest.py <document_id> --debug

# 2. Or re-ingest all documents
python batch_reingest.py --all --clear-first
```

### 4. Document Maintenance

```bash
# 1. Check document status
python documents.py list

# 2. Get detailed info about problematic document
python documents.py info <document_id>

# 3. Re-index if needed
python documents.py reindex <document_id>

# 4. Delete if necessary
python documents.py delete <document_id>
```

## Database Schema Updates

The database schema now properly supports multiple documents:

- `documents` table with unique IDs
- `passages` table with `document_id` foreign key
- Proper indexing for document filtering
- Meilisearch includes `document_id` field for filtering

## Best Practices

### 1. Metadata Management

- Use metadata files for consistent document information
- Include publication year and authors when available
- Use consistent naming conventions

### 2. Batch Processing

- Start with sequential processing for small batches
- Use parallel processing for large document sets
- Monitor system resources during parallel processing

### 3. Embedding Strategy

- Generate embeddings after all documents are ingested
- Use document-specific embedding for large document sets
- Monitor embedding coverage with `documents.py stats`

### 4. Search Optimization

- Use document filtering to improve search relevance
- Combine document filtering with query type detection
- Monitor search performance with different document sets

## Troubleshooting

### Common Issues

1. **Document not found**: Check if document exists with `documents.py list`
2. **Embedding failures**: Verify OpenAI API key or use local model
3. **Meilisearch errors**: Check if Meilisearch is running and accessible
4. **Database connection**: Verify database credentials in `.env` file

### Debug Commands

```bash
# Check system status
python documents.py stats

# Verify document exists
python documents.py info <document_id>

# Test search with document filter
python search.py --q "test query" --doc <document_id>

# Check embedding coverage
python embed.py --doc <document_id>
```

## Performance Considerations

- **Large document sets**: Use parallel processing and document-specific embedding
- **Memory usage**: Monitor during batch operations
- **API limits**: Be aware of OpenAI rate limits for embeddings
- **Storage**: Monitor database and Meilisearch storage usage

## Future Enhancements

Planned improvements for multi-document support:

- Progress tracking for long-running operations
- Configuration management for document processing
- Advanced document deduplication
- Document versioning and update tracking
- Bulk export/import capabilities





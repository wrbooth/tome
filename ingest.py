#!/usr/bin/env python3
"""
Codex Document Ingestion Script

Handles PDF/TXT ingestion, text extraction, chunking, and database storage.

This module serves as the orchestration layer and re-exports all public
functions from the sub-modules for backward compatibility.
"""

import os
import sys
import logging
import click
import fitz  # noqa: F401 — kept for backward compat (tests patch ingest.fitz)
from typing import Optional

from config import configure_logging

logger = logging.getLogger(__name__)

from config import db_connection, get_db_connection, get_meili_client  # noqa: F401 — re-export for patch compat

# Re-export all public functions for backward compatibility
from heading_detection import (  # noqa: F401
    detect_heading_patterns,
    is_quality_heading,
    extract_headings_from_outline,
    extract_headings_from_toc_pages,
    extract_headings_by_typography,
    filter_appendices,
    merge_heading_detection_methods,
    detect_heading_patterns_from_file,
    merge_heading_results,
    rank_headings_by_relevance,
    merge_heading_detection,
    debug_headings,
)
from pdf_extraction import extract_text_with_font_info, extract_text_from_txt  # noqa: F401
from chunking import count_tokens, chunk_text_with_headings  # noqa: F401
from entities import extract_entities_and_years  # noqa: F401
from storage import store_document, store_passages, index_in_meilisearch  # noqa: F401


def ingest_document(file_path: str, title: Optional[str] = None,
                    authors: Optional[str] = None, pub_year: Optional[int] = None,
                    debug: bool = False) -> str:
    """
    Ingest a document into the Codex system. Returns the document ID.

    Can be called directly from Python (e.g. batch_reingest) or via the CLI.
    """
    if file_path.lower().endswith('.pdf'):
        logger.info("Extracting text from PDF: %s", file_path)
        pages = extract_text_with_font_info(file_path)
    elif file_path.lower().endswith('.txt'):
        logger.info("Extracting text from TXT: %s", file_path)
        pages = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    logger.info("Extracted %d pages", len(pages))

    logger.info("Merging heading detection...")
    is_pdf = file_path.lower().endswith('.pdf')
    pages = merge_heading_detection(pages, is_pdf=is_pdf)

    if debug:
        debug_headings(pages)

    logger.info("Chunking text...")
    chunks = chunk_text_with_headings(pages, document_title=title)
    logger.info("Created %d chunks", len(chunks))

    if not title:
        title = os.path.basename(file_path)

    author_list = [a.strip() for a in authors.split(',') if a.strip()] if authors else None

    with db_connection() as conn:
        doc_id = store_document(conn, title, file_path, author_list, pub_year)
        logger.info("Stored document: %s (ID: %s)", title, doc_id)

        passage_ids = store_passages(conn, doc_id, chunks)
        logger.info("Stored %d passages", len(passage_ids))

        index_in_meilisearch(chunks, passage_ids, doc_id)

    logger.info("Ingestion complete!")
    return doc_id


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--title', help='Document title')
@click.option('--authors', help='Comma-separated list of authors')
@click.option('--pub-year', type=int, help='Publication year')
@click.option('--debug', is_flag=True, help='Show debug information for heading detection')
def main(file_path: str, title: str, authors: str, pub_year: int, debug: bool):
    """Ingest a document (PDF or TXT) into the Codex system."""
    configure_logging(logging.DEBUG if debug else logging.INFO)
    try:
        ingest_document(file_path, title, authors, pub_year, debug)
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

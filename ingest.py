#!/usr/bin/env python3
"""
Codex Document Ingestion Script

Handles PDF/TXT ingestion, text extraction, chunking, and database storage.

This module serves as the orchestration layer and re-exports all public
functions from the sub-modules for backward compatibility.
"""

import logging
import sys
from pathlib import Path

import click
import fitz  # noqa: F401 — kept for backward compat (tests patch ingest.fitz)

from config import configure_logging

logger = logging.getLogger(__name__)

from chunking import chunk_text_with_headings, count_tokens  # noqa: F401
from config import (  # noqa: F401 — re-export for patch compat
    db_connection,
    get_db_connection,
    get_meili_client,
)
from entities import extract_entities_and_years  # noqa: F401

# Re-export all public functions for backward compatibility
from heading_detection import (  # noqa: F401
    debug_headings,
    detect_heading_patterns,
    detect_heading_patterns_from_file,
    extract_headings_by_typography,
    extract_headings_from_outline,
    extract_headings_from_toc_pages,
    filter_appendices,
    is_quality_heading,
    merge_heading_detection,
    merge_heading_detection_methods,
    merge_heading_results,
    rank_headings_by_relevance,
)
from pdf_extraction import (
    extract_text_from_txt,
    extract_text_with_font_info,
)
from storage import index_in_meilisearch, store_document, store_passages


def ingest_document(
    file_path: str,
    title: str | None = None,
    authors: str | None = None,
    pub_year: int | None = None,
    debug: bool = False,
) -> str:
    """
    Ingest a document into the Codex system. Returns the document ID.

    Can be called directly from Python (e.g. batch_reingest) or via the CLI.
    """
    if file_path.lower().endswith(".pdf"):
        logger.info("Extracting text from PDF: %s", file_path)
        pages = extract_text_with_font_info(file_path)
    elif file_path.lower().endswith(".txt"):
        logger.info("Extracting text from TXT: %s", file_path)
        pages = extract_text_from_txt(file_path)
    else:
        msg = f"Unsupported file type: {file_path}"
        raise ValueError(msg)

    logger.info("Extracted %d pages", len(pages))

    logger.info("Merging heading detection...")
    is_pdf = file_path.lower().endswith(".pdf")
    pages = merge_heading_detection(pages, is_pdf=is_pdf)

    if debug:
        debug_headings(pages)

    logger.info("Chunking text...")
    chunks = chunk_text_with_headings(pages, document_title=title)
    logger.info("Created %d chunks", len(chunks))

    if not title:
        title = Path(file_path).name

    author_list = (
        [a.strip() for a in authors.split(",") if a.strip()] if authors else None
    )

    with db_connection() as conn:
        doc_id = store_document(conn, title, file_path, author_list, pub_year)
        logger.info("Stored document: %s (ID: %s)", title, doc_id)

        passage_ids = store_passages(conn, doc_id, chunks)
        logger.info("Stored %d passages", len(passage_ids))

        index_in_meilisearch(chunks, passage_ids, doc_id)

    logger.info("Ingestion complete!")
    return doc_id


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--title", help="Document title")
@click.option("--authors", help="Comma-separated list of authors")
@click.option("--pub-year", type=int, help="Publication year")
@click.option(
    "--debug", is_flag=True, help="Show debug information for heading detection"
)
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

#!/usr/bin/env python3
"""
Codex Batch Re-ingestion Script

Handles re-ingestion of multiple documents with options to:
- Clear database and Meilisearch first
- Re-ingest specific documents
- Re-ingest all documents
- Selective clearing
"""

import logging
import sys
from pathlib import Path
from typing import Any

import click
from psycopg2.extras import RealDictCursor

from config import configure_logging, db_connection, get_meili_client

logger = logging.getLogger(__name__)


def clear_database():
    """Clear all data from the database."""
    logger.info("Clearing database...")

    with db_connection() as conn:
        with conn.cursor() as cur:
            # Clear passages first (due to foreign key constraint)
            cur.execute("DELETE FROM passages")
            logger.info("  Deleted %d passages", cur.rowcount)

            # Clear documents
            cur.execute("DELETE FROM documents")
            logger.info("  Deleted %d documents", cur.rowcount)

        logger.info("Database cleared successfully")


def clear_documents(documents: list[str]):
    """Clear specific documents from the database."""
    logger.info("Clearing %d documents from database...", len(documents))

    with db_connection() as conn:
        with conn.cursor() as cur:
            for doc_id in documents:
                # Delete passages first
                cur.execute("DELETE FROM passages WHERE document_id = %s", (doc_id,))
                passage_count = cur.rowcount

                # Delete document
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                doc_count = cur.rowcount

                logger.info(
                    "  Deleted document %s: %d document, %d passages",
                    doc_id,
                    doc_count,
                    passage_count,
                )

        logger.info("Documents cleared successfully")


def clear_meilisearch():
    """Clear all data from Meilisearch."""
    logger.info("Clearing Meilisearch...")

    try:
        client = get_meili_client()

        # Delete the index if it exists
        try:
            client.index("passages").delete()
            logger.info("  Deleted Meilisearch index")
        except Exception as e:
            if "not found" in str(e).lower():
                logger.info("  Meilisearch index already empty")
            else:
                logger.error("  Error deleting Meilisearch index: %s", e)

    except Exception as e:
        logger.error("Error connecting to Meilisearch: %s", e)


def get_document_info(conn, document_id: str) -> dict[str, Any] | None:
    """Get document information."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, title, source_path, authors, pub_year
            FROM documents
            WHERE id = %s
        """,
            (document_id,),
        )
        return cur.fetchone()


def list_all_documents() -> list[dict[str, Any]]:
    """List all documents in the database."""
    with db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
                SELECT id, title, source_path, authors, pub_year
                FROM documents
                ORDER BY title
            """)
        return [dict(row) for row in cur.fetchall()]


def reingest_document(
    file_path: str,
    title: str,
    authors: str | None = None,
    pub_year: int | None = None,
    debug: bool = False,
) -> bool:
    """Re-ingest a single document."""
    logger.info("Re-ingesting document: %s", Path(file_path).name)

    try:
        from ingest import ingest_document

        ingest_document(file_path, title, authors, pub_year, debug)
        logger.info("Successfully re-ingested: %s", Path(file_path).name)
        return True
    except Exception as e:
        logger.error("Error re-ingesting %s: %s", Path(file_path).name, e)
        return False


def run_embedding(document_ids: list[str] | None = None):
    """Run the embedding process."""
    logger.info("Running embedding process...")

    try:
        from embed import run_embeddings

        success = run_embeddings(document_ids)
        if success:
            logger.info("Embedding completed successfully")
        else:
            logger.error("Embedding generation failed")
        return success
    except Exception as e:
        logger.error("Error running embedding: %s", e)
        return False


@click.command()
@click.argument("documents", nargs=-1)
@click.option("--all", is_flag=True, help="Re-ingest all documents")
@click.option(
    "--clear-first", is_flag=True, help="Clear database and Meilisearch first"
)
@click.option("--clear-docs", help="Comma-separated list of document IDs to clear")
@click.option("--skip-embedding", is_flag=True, help="Skip embedding generation")
@click.option("--debug", is_flag=True, help="Show debug information")
def main(  # noqa: C901
    documents: list[str],
    all: bool,  # noqa: A002
    clear_first: bool,
    clear_docs: str | None,
    skip_embedding: bool,
    debug: bool,
):
    """Batch re-ingest documents."""
    configure_logging(logging.DEBUG if debug else logging.INFO)

    logger.info("=== Codex Batch Re-ingestion ===")

    # Determine which documents to process
    if all:
        logger.info("Re-ingesting all documents...")
        all_docs = list_all_documents()
        if not all_docs:
            logger.info("No documents found in database")
            return

        documents_to_process = all_docs
        logger.info("Found %d documents to re-ingest", len(documents_to_process))

    elif documents:
        logger.info("Re-ingesting %d specific documents...", len(documents))
        with db_connection() as conn:
            documents_to_process = []

            for doc_id in documents:
                doc_info = get_document_info(conn, doc_id)
                if doc_info:
                    documents_to_process.append(doc_info)
                else:
                    logger.warning("Document %s not found in database", doc_id)

        if not documents_to_process:
            logger.info("No valid documents found")
            return
    else:
        logger.error("Must specify documents to re-ingest or use --all")
        return

    # Clear operations
    if clear_first:
        logger.info("Clearing database and Meilisearch...")
        clear_database()
        clear_meilisearch()
    elif clear_docs:
        doc_ids_to_clear = [doc_id.strip() for doc_id in clear_docs.split(",")]
        logger.info("Clearing specific documents: %s", ", ".join(doc_ids_to_clear))
        clear_documents(doc_ids_to_clear)

    # Re-ingest documents
    logger.info("Re-ingesting %d documents...", len(documents_to_process))
    successful = 0
    failed = 0

    for i, doc in enumerate(documents_to_process, 1):
        logger.info(
            "[%d/%d] Processing: %s", i, len(documents_to_process), doc["title"]
        )

        # Check if source file exists
        if not Path(doc["source_path"]).exists():
            logger.error("Source file not found: %s", doc["source_path"])
            failed += 1
            continue

        # Prepare metadata
        authors = ", ".join(doc["authors"]) if doc["authors"] else None
        pub_year = doc["pub_year"]

        # Re-ingest
        success = reingest_document(
            doc["source_path"], doc["title"], authors, pub_year, debug
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    logger.info("=== Re-ingestion Summary ===")
    logger.info("Total: %d", len(documents_to_process))
    logger.info("Successful: %d", successful)
    logger.info("Failed: %d", failed)

    if failed > 0:
        logger.info("Failed documents:")
        for _, doc in enumerate(documents_to_process):
            if not Path(doc["source_path"]).exists():
                logger.info("  - %s: Source file not found", doc["title"])

    # Run embedding if requested
    if not skip_embedding and successful > 0:
        logger.info("Generating embeddings...")

        if all:
            # Generate embeddings for all documents
            embedding_success = run_embedding()
        else:
            # Generate embeddings for specific documents
            doc_ids = [doc["id"] for doc in documents_to_process]
            embedding_success = run_embedding(doc_ids)

        if embedding_success:
            logger.info("Embedding generation completed")
        else:
            logger.error("Embedding generation failed")

    logger.info("=== Re-ingestion Complete ===")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Codex Document Management CLI

Provides commands for managing documents in the system:
- List documents with stats
- Delete documents
- Re-index documents in Meilisearch
- Get document details
"""

import json
import logging
import sys
from typing import Any

import click
from psycopg2.extras import RealDictCursor
from tabulate import tabulate

from codex.config import configure_logging, db_connection, get_meili_client

logger = logging.getLogger(__name__)


def get_system_stats(conn) -> dict[str, Any]:
    """Get overall system statistics.

    Returns a dict with keys: documents, passages, embedded_passages,
    entities, years, and embedding_coverage (formatted string).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        doc_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM passages")
        passage_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL")
        embedded_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM passage_entities")
        entity_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM passage_years")
        year_count = cur.fetchone()[0]

    return {
        "documents": doc_count,
        "passages": passage_count,
        "embedded_passages": embedded_count,
        "entities": entity_count,
        "years": year_count,
        "embedding_coverage": f"{(embedded_count / passage_count * 100):.1f}%"
        if passage_count > 0
        else "0%",
    }


def get_document_stats(conn, document_id: str) -> dict[str, Any]:
    """Get detailed stats for a document."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Document info
        cur.execute(
            """
            SELECT id, title, authors, pub_year, source_path
            FROM documents
            WHERE id = %s
        """,
            (document_id,),
        )
        doc = cur.fetchone()

        if not doc:
            return None

        # Passage count
        cur.execute(
            """
            SELECT COUNT(*) as total_passages,
                   COUNT(CASE WHEN embedding IS NOT NULL
                         THEN 1 END) as embedded_passages
            FROM passages
            WHERE document_id = %s
        """,
            (document_id,),
        )
        passage_stats = cur.fetchone()

        # Entity count
        cur.execute(
            """
            SELECT COUNT(*) as entity_count
            FROM passage_entities pe
            JOIN passages p ON pe.passage_id = p.id
            WHERE p.document_id = %s
        """,
            (document_id,),
        )
        entity_stats = cur.fetchone()

        # Year count
        cur.execute(
            """
            SELECT COUNT(*) as year_count
            FROM passage_years py
            JOIN passages p ON py.passage_id = p.id
            WHERE p.document_id = %s
        """,
            (document_id,),
        )
        year_stats = cur.fetchone()

        # Page range
        cur.execute(
            """
            SELECT MIN(page) as min_page, MAX(page) as max_page
            FROM passages
            WHERE document_id = %s
        """,
            (document_id,),
        )
        page_stats = cur.fetchone()

        return {
            "document": dict(doc),
            "passages": dict(passage_stats),
            "entities": dict(entity_stats),
            "years": dict(year_stats),
            "pages": dict(page_stats),
        }


def list_documents(conn) -> list[dict[str, Any]]:
    """List all documents with basic stats."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                d.id,
                d.title,
                d.authors,
                d.pub_year,
                COUNT(p.id) as passage_count,
                COUNT(CASE WHEN p.embedding IS NOT NULL THEN 1 END) as embedded_count,
                MIN(p.page) as min_page,
                MAX(p.page) as max_page
            FROM documents d
            LEFT JOIN passages p ON d.id = p.document_id
            GROUP BY d.id, d.title, d.authors, d.pub_year
            ORDER BY d.title
        """)
        return [dict(row) for row in cur.fetchall()]


def delete_document(conn, document_id: str, confirm: bool = True) -> bool:
    """Delete a document and all its passages."""
    if confirm:
        # Get document info for confirmation
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT title FROM documents WHERE id = %s", (document_id,))
            doc = cur.fetchone()
            if not doc:
                logger.warning("Document %s not found", document_id)
                return False

            # Get passage count
            cur.execute(
                "SELECT COUNT(*) FROM passages WHERE document_id = %s", (document_id,)
            )
            passage_count = cur.fetchone()[0]

            logger.info("About to delete document: %s", doc["title"])
            logger.info("This will also delete %d passages", passage_count)

            if not click.confirm("Are you sure you want to continue?"):
                logger.info("Deletion cancelled")
                return False

    try:
        with conn.cursor() as cur:
            # Delete document (passages are removed via ON DELETE CASCADE)
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            doc_deleted = cur.rowcount

            conn.commit()

            logger.info("Deleted %d document (passages cascaded)", doc_deleted)
            return True

    except Exception as e:
        conn.rollback()
        logger.error("Error deleting document: %s", e)
        return False


def reindex_document_meilisearch(conn, document_id: str) -> bool:
    """Re-index a document in Meilisearch."""
    try:
        # Get document passages
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.id, p.text, p.page, p.headings_path, d.title
                FROM passages p
                JOIN documents d ON p.document_id = d.id
                WHERE p.document_id = %s
            """,
                (document_id,),
            )
            passages = cur.fetchall()

        if not passages:
            logger.warning("No passages found for document %s", document_id)
            return False

        # Prepare documents for indexing
        from codex.ingestion.ingest import extract_entities_and_years

        documents = []
        for passage in passages:
            # Extract entities and years
            entities, years = extract_entities_and_years(passage["text"])

            person_entities = [
                e["entity"] for e in entities if e["ent_type"] == "PERSON"
            ]
            place_entities = [
                e["entity"] for e in entities if e["ent_type"] in ["GPE", "FAC"]
            ]

            documents.append(
                {
                    "id": passage["id"],
                    "document_id": document_id,
                    "text": passage["text"],
                    "page": passage["page"],
                    "headings_path": passage["headings_path"],
                    "years": years,
                    "entities_person": person_entities,
                    "entities_place": place_entities,
                }
            )

        # Index in Meilisearch
        client = get_meili_client()
        index = client.index("passages")
        index.add_documents(documents, primary_key="id")

        logger.info(
            "Re-indexed %d passages for document %s", len(documents), document_id
        )
        return True

    except ImportError:
        logger.warning("Meilisearch client not available")
        return False
    except Exception as e:
        logger.error("Error re-indexing document: %s", e)
        return False


@click.group()
def cli():
    """Codex Document Management CLI."""


@cli.command()
@click.option(
    "--format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
def list(format: str):  # noqa: A001, A002
    """List all documents with stats."""
    configure_logging()
    try:
        with db_connection() as conn:
            documents = list_documents(conn)

            if format == "json":
                click.echo(json.dumps(documents, indent=2))
            else:
                if not documents:
                    click.echo("No documents found")
                    return

                # Prepare table data
                table_data = []
                for doc in documents:
                    authors = ", ".join(doc["authors"]) if doc["authors"] else "Unknown"
                    embedding_pct = (
                        f"{(doc['embedded_count'] / doc['passage_count'] * 100):.1f}%"
                        if doc["passage_count"] > 0
                        else "0%"
                    )
                    page_range = (
                        f"{doc['min_page']}-{doc['max_page']}"
                        if doc["min_page"] and doc["max_page"]
                        else "N/A"
                    )

                    table_data.append(
                        [
                            doc["id"][:8] + "...",
                            doc["title"][:50]
                            + ("..." if len(doc["title"]) > 50 else ""),
                            authors[:30] + ("..." if len(authors) > 30 else ""),
                            doc["pub_year"] or "Unknown",
                            doc["passage_count"],
                            f"{doc['embedded_count']} ({embedding_pct})",
                            page_range,
                        ]
                    )

                headers = [
                    "ID",
                    "Title",
                    "Authors",
                    "Year",
                    "Passages",
                    "Embedded",
                    "Pages",
                ]
                click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

    except Exception as e:
        logger.error("Error listing documents: %s", e)
        sys.exit(1)


@cli.command()
@click.argument("document_id")
@click.option(
    "--format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
def info(document_id: str, format: str):  # noqa: A002
    """Get detailed information about a document."""
    configure_logging()
    try:
        with db_connection() as conn:
            stats = get_document_stats(conn, document_id)

            if not stats:
                click.echo(f"Document {document_id} not found")
                sys.exit(1)

            if format == "json":
                click.echo(json.dumps(stats, indent=2))
            else:
                doc = stats["document"]
                passages = stats["passages"]
                entities = stats["entities"]
                years = stats["years"]
                pages = stats["pages"]

                click.echo(f"Document: {doc['title']}")
                click.echo(f"ID: {doc['id']}")
                authors = ", ".join(doc["authors"]) if doc["authors"] else "Unknown"
                click.echo(f"Authors: {authors}")
                click.echo(f"Year: {doc['pub_year'] or 'Unknown'}")
                click.echo(f"Source: {doc['source_path']}")
                click.echo()

                click.echo("Statistics:")
                total = passages["total_passages"]
                embedded = passages["embedded_passages"]
                click.echo(f"  Passages: {total} total, {embedded} embedded")
                click.echo(f"  Entities: {entities['entity_count']}")
                click.echo(f"  Years: {years['year_count']}")
                click.echo(f"  Pages: {pages['min_page']} - {pages['max_page']}")

                if passages["total_passages"] > 0:
                    embedding_pct = (
                        passages["embedded_passages"] / passages["total_passages"]
                    ) * 100
                    click.echo(f"  Embedding coverage: {embedding_pct:.1f}%")

    except Exception as e:
        logger.error("Error getting document info: %s", e)
        sys.exit(1)


@cli.command()
@click.argument("document_id")
@click.option("--force", is_flag=True, help="Skip confirmation")
def delete(document_id: str, force: bool):
    """Delete a document and all its passages."""
    configure_logging()
    try:
        with db_connection() as conn:
            success = delete_document(conn, document_id, confirm=not force)

            if not success:
                sys.exit(1)

    except Exception as e:
        logger.error("Error deleting document: %s", e)
        sys.exit(1)


@cli.command()
@click.argument("document_id")
def reindex(document_id: str):
    """Re-index a document in Meilisearch."""
    configure_logging()
    try:
        with db_connection() as conn:
            success = reindex_document_meilisearch(conn, document_id)

            if not success:
                sys.exit(1)

    except Exception as e:
        logger.error("Error re-indexing document: %s", e)
        sys.exit(1)


@cli.command()
def stats():
    """Get overall system statistics."""
    configure_logging()
    try:
        with db_connection() as conn:
            s = get_system_stats(conn)

            click.echo("System Statistics:")
            click.echo(f"  Documents: {s['documents']}")
            click.echo(f"  Passages: {s['passages']}")
            click.echo(f"  Embedded passages: {s['embedded_passages']}")
            click.echo(f"  Entities: {s['entities']}")
            click.echo(f"  Years: {s['years']}")

            if s["passages"] > 0:
                click.echo(f"  Embedding coverage: {s['embedding_coverage']}")

    except Exception as e:
        logger.error("Error getting system stats: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    cli()

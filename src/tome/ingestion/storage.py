"""
Database and search index storage for the Tome ingestion pipeline.

Handles storing documents and passages in PostgreSQL, and indexing
passages in Meilisearch.
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

from tome.config import EMBEDDING_MODEL, get_meili_client, settings
from tome.ingestion.entities import extract_entities_and_years


def store_document(
    conn,
    title: str,
    source_path: str,
    authors: list[str] | None = None,
    pub_year: int | None = None,
) -> str:
    """Store document in database and return document ID."""
    doc_id = str(uuid.uuid4())

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (id, title, authors, pub_year, source_path)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (doc_id, title, authors, pub_year, source_path),
        )

    conn.commit()
    return doc_id


def store_passages(conn, doc_id: str, chunks: list[dict[str, Any]]) -> list[str]:
    """Store passages in database and return passage IDs."""
    passage_ids = []

    with conn.cursor() as cur:
        for chunk in chunks:
            passage_id = str(uuid.uuid4())
            passage_ids.append(passage_id)

            cur.execute(
                """
                INSERT INTO passages (id, document_id, page, text, headings_path)
                VALUES (%s, %s, %s, %s, %s)
            """,
                (
                    passage_id,
                    doc_id,
                    chunk["page"],
                    chunk["text"],
                    chunk["headings_path"],  # Pass the list directly
                ),
            )

            # Extract and store entities and years from original text (not prefixed)
            text_for_entities = chunk.get("original_text", chunk["text"])
            entities, years = extract_entities_and_years(text_for_entities)

            for entity in entities:
                cur.execute(
                    """
                    INSERT INTO passage_entities
                        (passage_id, entity, ent_type, norm_entity)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (passage_id, entity) DO NOTHING
                """,
                    (
                        passage_id,
                        entity["entity"],
                        entity["ent_type"],
                        entity["norm_entity"],
                    ),
                )

            for year in years:
                cur.execute(
                    """
                    INSERT INTO passage_years (passage_id, year)
                    VALUES (%s, %s)
                    ON CONFLICT (passage_id, year) DO NOTHING
                """,
                    (passage_id, year),
                )

    conn.commit()
    return passage_ids


def index_in_meilisearch(
    passages: list[dict[str, Any]], passage_ids: list[str], doc_id: str
):
    """Index passages in Meilisearch."""
    try:
        client = get_meili_client()

        # Prepare documents for indexing
        documents = []
        for passage, passage_id in zip(passages, passage_ids, strict=False):
            # Extract years and entities from original text (not prefixed)
            text_for_entities = passage.get("original_text", passage["text"])
            entities, years = extract_entities_and_years(text_for_entities)

            person_entities = [
                e["entity"] for e in entities if e["ent_type"] == "PERSON"
            ]
            place_entities = [
                e["entity"] for e in entities if e["ent_type"] in ["GPE", "FAC"]
            ]

            documents.append(
                {
                    "id": passage_id,
                    "document_id": doc_id,  # Add document_id for filtering
                    "text": passage["text"],  # Use prefixed text for search
                    "page": passage["page"],
                    "headings_path": passage["headings_path"],
                    "years": years,
                    "entities_person": person_entities,
                    "entities_place": place_entities,
                }
            )

        # Create or update index
        index = client.index("passages")

        # Check if index exists; if not, configure embedder and filterable attributes
        try:
            index_settings = index.get_settings()
            needs_setup = not index_settings.get("embedders")
        except Exception:
            needs_setup = True

        if needs_setup:
            openai_key = settings.openai_api_key
            if openai_key:
                task = index.update_embedders(
                    {
                        "default": {
                            "source": "openAi",
                            "apiKey": openai_key,
                            "model": EMBEDDING_MODEL,
                            "documentTemplate": "{{doc.text}}",
                        }
                    }
                )
                client.wait_for_task(task.task_uid, timeout_in_ms=300000)

            task = index.update_filterable_attributes(["document_id"])
            client.wait_for_task(task.task_uid, timeout_in_ms=60000)

        index.add_documents(documents, primary_key="id")

        logger.info(
            "Indexed %d passages in Meilisearch for document %s", len(documents), doc_id
        )

    except ImportError:
        logger.warning("Meilisearch client not available")
    except Exception as e:
        logger.warning("Failed to index in Meilisearch: %s", e)

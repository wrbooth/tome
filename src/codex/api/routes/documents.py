"""Document management API routes."""

import logging
import uuid

from fastapi import APIRouter, HTTPException
from psycopg2.extras import RealDictCursor

from codex.config import db_connection, get_meili_client
from codex.documents import get_document_stats, get_system_stats
from codex.models import DocumentDetail, DocumentInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all documents in the system."""
    try:
        with db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, title, authors, pub_year, source_path
                    FROM documents ORDER BY title
                """)
                documents = cur.fetchall()
            return [DocumentInfo(**dict(doc)) for doc in documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document_detail(document_id: str):
    """Get detailed information about a single document."""
    try:
        with db_connection() as conn:
            stats = get_document_stats(conn, document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if stats is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return stats


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its passages, entities, and years."""
    try:
        uuid.UUID(document_id)
    except (ValueError, AttributeError) as e:
        raise HTTPException(status_code=400, detail="Invalid document ID") from e

    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM passages WHERE document_id = %s", (document_id,)
                )
                passage_ids = [str(row[0]) for row in cur.fetchall()]

                cur.execute(
                    "DELETE FROM documents WHERE id = %s RETURNING id", (document_id,)
                )
                deleted = cur.fetchone()
                if not deleted:
                    raise HTTPException(
                        status_code=404, detail="Document not found"
                    )
            conn.commit()

        if passage_ids:
            try:
                client = get_meili_client()
                client.index("passages").delete_documents(passage_ids)
            except Exception as e:
                logger.warning("Meilisearch cleanup failed (non-fatal): %s", e)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"status": "deleted", "document_id": document_id}


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        with db_connection() as conn:
            return get_system_stats(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

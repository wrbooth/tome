"""
Codex API Service

FastAPI wrapper for search, document management, and streaming endpoints.
"""

import sys
import asyncio
import logging
import uuid
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from psycopg2.extras import RealDictCursor

from config import db_connection, get_meili_client
from documents import get_system_stats, get_document_stats
from search import search_codex, analyze_query, hybrid_search, rerank_candidates, get_passage_details
from answer_generator import stream_answer_with_llm
from models import (
    PassageDetail, DocumentInfo, QueryAnalysisInfo, QueryEntities,
    DocumentDetail, IngestTaskInfo,
)

app = FastAPI(title="Codex Search API", version="2.0.0")

# Sentinel for queue-based streaming bridge
_SENTINEL = object()

# CORS -- allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = 20
    document_id: Optional[str] = None

class SearchResult(BaseModel):
    """A single search result with full passage details."""
    passage_id: str
    title: str
    page: int
    snippet: str
    text: str
    headings_path: List[str] = []
    score: float

class SearchResponse(BaseModel):
    query_type: str
    query_analysis: QueryAnalysisInfo
    answer: str
    results: List[SearchResult]


# ---------------------------------------------------------------------------
# Background ingestion task store
# ---------------------------------------------------------------------------

_ingest_tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search passages and generate an answer."""
    try:
        result = await asyncio.to_thread(
            search_codex, request.query, k=request.k, document_id=request.document_id
        )

        search_results = []
        for r in result["results"]:
            text = r["text"]
            snippet = text[:200] + "..." if len(text) > 200 else text
            search_results.append(SearchResult(
                passage_id=r["id"],
                title=r["title"],
                page=r["page"],
                snippet=snippet,
                text=text,
                headings_path=r.get("headings_path") or [],
                score=r["score"],
            ))

        qa = result.get("query_analysis", {})
        query_analysis = QueryAnalysisInfo(
            query_type=qa.get("query_type", result["query_type"]),
            person=qa.get("person"),
            entities=QueryEntities(**qa.get("entities", {})),
            expansions=qa.get("expansions", []),
        )

        return SearchResponse(
            query_type=result["query_type"],
            query_analysis=query_analysis,
            answer=result["answer"],
            results=search_results,
        )

    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


def _sse_event(event: str, data) -> str:
    """Format a Server-Sent Event string."""
    import json as _json
    payload = _json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/search/stream")
async def search_stream(request: SearchRequest):
    """Stream search results + LLM answer via Server-Sent Events.

    Events emitted:
      - search_results: immediate results + query analysis
      - token: individual LLM answer tokens
      - done: signals completion
      - error: on failure
    """

    async def _generate():
        try:
            qa = await asyncio.to_thread(analyze_query, request.query)

            candidates = await asyncio.to_thread(
                hybrid_search, request.query, 200, request.document_id
            )

            if not candidates:
                yield _sse_event("search_results", {
                    "query_type": qa["query_type"], "query_analysis": qa, "results": [],
                })
                yield _sse_event("token", "No candidates found.")
                yield _sse_event("done", {})
                return

            def _rerank_and_detail():
                with db_connection() as conn:
                    reranked = rerank_candidates(candidates, request.query, conn, k=request.k)
                    passage_ids = [pid for pid, _ in reranked]
                    details = get_passage_details(conn, passage_ids)
                    score_map = {pid: score for pid, score in reranked}
                    for d in details:
                        d["score"] = score_map.get(d["id"], 0.0)
                    return details

            passage_details = await asyncio.to_thread(_rerank_and_detail)

            # Send search results immediately
            results_payload = []
            for r in passage_details:
                text = r["text"]
                results_payload.append({
                    "passage_id": r["id"],
                    "title": r["title"],
                    "page": r["page"],
                    "snippet": text[:200] + "..." if len(text) > 200 else text,
                    "text": text,
                    "headings_path": r.get("headings_path") or [],
                    "score": r["score"],
                })

            yield _sse_event("search_results", {
                "query_type": qa["query_type"],
                "query_analysis": qa,
                "results": results_payload,
            })

            # Stream LLM answer tokens via queue bridge (sync generator → async yields)
            chunks = [
                {"text": r["text"], "page": r["page"], "title": r["title"]}
                for r in passage_details
            ]

            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _produce_tokens():
                try:
                    for token in stream_answer_with_llm(request.query, chunks):
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

            loop.run_in_executor(None, _produce_tokens)

            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield _sse_event("token", item)

            yield _sse_event("done", {})

        except Exception as e:
            logger.exception("Streaming search failed")
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/documents", response_model=List[DocumentInfo])
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document_detail(document_id: str):
    """Get detailed information about a single document."""
    try:
        with db_connection() as conn:
            stats = get_document_stats(conn, document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if stats is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return stats


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its passages, entities, and years."""
    try:
        uuid.UUID(document_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                # Get passage IDs for Meilisearch cleanup
                cur.execute("SELECT id FROM passages WHERE document_id = %s", (document_id,))
                passage_ids = [str(row[0]) for row in cur.fetchall()]

                # Delete from Postgres (cascades to passages, entities, years)
                cur.execute("DELETE FROM documents WHERE id = %s RETURNING id", (document_id,))
                deleted = cur.fetchone()
                if not deleted:
                    raise HTTPException(status_code=404, detail="Document not found")
            conn.commit()

        # Clean up Meilisearch index (best-effort)
        if passage_ids:
            try:
                client = get_meili_client()
                client.index("passages").delete_documents(passage_ids)
            except Exception as e:
                logger.warning("Meilisearch cleanup failed (non-fatal): %s", e)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "deleted", "document_id": document_id}


@router.post("/documents/upload", status_code=202, response_model=IngestTaskInfo)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    authors: str = Form(""),
    pub_year: Optional[int] = Form(None),
):
    """Upload a document for ingestion. Returns a task ID for status polling.

    Accepts PDF or TXT files. Ingestion runs in the background.
    """
    # Log upload requests to diagnose phantom "Notes" documents
    import traceback
    logger.warning(
        "UPLOAD REQUEST: filename=%r, title=%r, authors=%r, pub_year=%r, content_type=%r, size=%r",
        file.filename, title, authors, pub_year, file.content_type,
        file.size if hasattr(file, 'size') else 'unknown',
    )
    logger.warning("UPLOAD STACK:\n%s", "".join(traceback.format_stack()))

    # Validate file type
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are accepted")

    # Save to temp file
    tmp_dir = Path(tempfile.gettempdir()) / "codex_uploads"
    tmp_dir.mkdir(exist_ok=True)
    dest = tmp_dir / f"{uuid.uuid4()}{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _ingest_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "document_id": None,
        "filename": filename,
        "message": "Queued for ingestion",
        "created_at": now,
    }

    asyncio.create_task(_run_ingestion(task_id, str(dest), title, authors, pub_year))

    return _ingest_tasks[task_id]


@router.get("/documents/upload/{task_id}", response_model=IngestTaskInfo)
async def get_upload_status(task_id: str):
    """Poll the status of a document upload/ingestion task."""
    task = _ingest_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


async def _run_ingestion(
    task_id: str, file_path: str, title: str, authors: str, pub_year: Optional[int]
):
    """Background coroutine that runs ingest_document in a thread."""
    task = _ingest_tasks[task_id]
    task["status"] = "running"
    task["message"] = "Ingesting document..."

    try:
        from ingest import ingest_document

        doc_id = await asyncio.to_thread(
            ingest_document,
            file_path,
            title=title or None,
            authors=authors or None,
            pub_year=pub_year,
        )
        task["status"] = "completed"
        task["document_id"] = doc_id
        task["message"] = "Ingestion complete"
    except Exception as e:
        logger.exception("Ingestion failed for task %s", task_id)
        task["status"] = "failed"
        task["message"] = str(e)
    finally:
        # Clean up temp file
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        with db_connection() as conn:
            return get_system_stats(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

# ---------------------------------------------------------------------------
# Static file serving (SPA) — no-cache in development
# ---------------------------------------------------------------------------

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Add no-cache headers to static file responses for development."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

app.add_middleware(NoCacheStaticMiddleware)

_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for SPA routing."""
        return FileResponse(str(_static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Document upload/ingestion API routes."""

import asyncio
import contextlib
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anyio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from tome.models import IngestTaskInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

# Background ingestion task store
_ingest_tasks: dict[str, dict] = {}


@router.post("/documents/upload", status_code=202, response_model=IngestTaskInfo)
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
    title: str = Form(""),
    authors: str = Form(""),
    pub_year: int | None = Form(None),
):
    """Upload a document for ingestion. Returns a task ID for status polling.

    Accepts PDF or TXT files. Ingestion runs in the background.
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(
            status_code=400, detail="Only PDF and TXT files are accepted"
        )

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

    _background_task = asyncio.create_task(  # noqa: RUF006
        _run_ingestion(task_id, str(dest), title, authors, pub_year)
    )

    return _ingest_tasks[task_id]


@router.get("/documents/upload/{task_id}", response_model=IngestTaskInfo)
async def get_upload_status(task_id: str):
    """Poll the status of a document upload/ingestion task."""
    task = _ingest_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


async def _run_ingestion(
    task_id: str, file_path: str, title: str, authors: str, pub_year: int | None
):
    """Background coroutine that runs ingest_document in a thread."""
    task = _ingest_tasks[task_id]
    task["status"] = "running"
    task["message"] = "Ingesting document..."

    try:
        from tome.ingestion.ingest import ingest_document

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
        with contextlib.suppress(Exception):
            await anyio.Path(file_path).unlink(missing_ok=True)

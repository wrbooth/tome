"""
Codex API Service

FastAPI wrapper for the search functionality.
"""

import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from psycopg2.extras import RealDictCursor

from config import db_connection
from documents import get_system_stats
from search import search_codex
from models import PassageDetail, DocumentInfo

app = FastAPI(title="Codex Search API", version="1.0.0")


# API-specific request/response models that reference shared models.

class SearchRequest(BaseModel):
    query: str
    k: int = 20
    document_id: Optional[str] = None

class SearchResult(BaseModel):
    """API representation of a search result (includes snippet instead of full text)."""
    passage_id: str
    title: str
    page: int
    snippet: str
    score: float

class SearchResponse(BaseModel):
    query_type: str
    answer: str
    results: List[SearchResult]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search passages and generate an answer."""
    try:
        result = search_codex(request.query, k=request.k, document_id=request.document_id)

        search_results = []
        for r in result["results"]:
            text = r["text"]
            snippet = text[:200] + "..." if len(text) > 200 else text
            search_results.append(SearchResult(
                passage_id=r["id"],
                title=r["title"],
                page=r["page"],
                snippet=snippet,
                score=r["score"]
            ))

        return SearchResponse(
            query_type=result["query_type"],
            answer=result["answer"],
            results=search_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
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

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        with db_connection() as conn:
            return get_system_stats(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Codex API Service

FastAPI wrapper for the search functionality.
"""

import sys
sys.path.append('..')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from psycopg2.extras import RealDictCursor

from config import get_db_connection
from search import search_codex

app = FastAPI(title="Codex Search API", version="1.0.0")

class SearchRequest(BaseModel):
    query: str
    k: int = 20
    document_id: Optional[str] = None

class SearchResult(BaseModel):
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

@app.get("/documents")
async def list_documents():
    """List all documents in the system."""
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, authors, pub_year, source_path
                FROM documents ORDER BY title
            """)
            documents = cur.fetchall()
        conn.close()
        return [dict(doc) for doc in documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM passages")
            passage_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL")
            embedded_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM passage_entities")
            entity_count = cur.fetchone()[0]
        conn.close()

        return {
            "documents": doc_count,
            "passages": passage_count,
            "embedded_passages": embedded_count,
            "embedding_coverage": f"{(embedded_count/passage_count*100):.1f}%" if passage_count > 0 else "0%",
            "entities": entity_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

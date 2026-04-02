"""
Codex API Service

Simple FastAPI wrapper for the search functionality.
"""

import sys
sys.path.append('..')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from psycopg2.extras import RealDictCursor

from config import get_db_connection

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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """Search passages using the same logic as search.py."""
    try:
        from search import (
            hybrid_search,
            rerank_candidates,
            get_passage_details,
        )

        # Hybrid search (keyword + semantic in one Meilisearch call)
        candidates = hybrid_search(request.query, k=200, document_id=request.document_id)

        if not candidates:
            return []

        # Re-rank with cross-encoder
        conn = get_db_connection()
        reranked_candidates = rerank_candidates(candidates, request.query, conn, k=request.k)

        # Get passage details
        passage_ids = [pid for pid, _ in reranked_candidates]
        passage_details = get_passage_details(conn, passage_ids)

        # Create score mapping
        score_map = {pid: score for pid, score in reranked_candidates}
        
        # Format results
        results = []
        for result in passage_details:
            passage_id = result['id']
            score = score_map.get(passage_id, 0.0)
            
            # Truncate text for snippet
            text = result['text']
            snippet = text[:200] + "..." if len(text) > 200 else text
            
            results.append(SearchResult(
                passage_id=passage_id,
                title=result['title'],
                page=result['page'],
                snippet=snippet,
                score=score
            ))
        
        conn.close()
        return results
        
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
                FROM documents
                ORDER BY title
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
            # Document count
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
            
            # Passage count
            cur.execute("SELECT COUNT(*) FROM passages")
            passage_count = cur.fetchone()[0]
            
            # Embedded passages
            cur.execute("SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL")
            embedded_count = cur.fetchone()[0]
            
            # Entity count
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









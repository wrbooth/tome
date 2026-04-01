"""
Codex API Service

Simple FastAPI wrapper for the search functionality.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Codex Search API", version="1.0.0")

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "codex"),
        user=os.getenv("DB_USER", "codex"),
        password=os.getenv("DB_PASSWORD", "codex")
    )

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
        # Import search functions from the main search module
        import sys
        sys.path.append('..')
        
        from search import (
            get_meili_candidates,
            get_vector_candidates,
            rrf,
            apply_query_boosting,
            get_passage_details,
            analyze_query
        )

        # Analyze query (single LLM call)
        query_analysis = analyze_query(request.query)

        # Get candidates from both sources
        meili_candidates = get_meili_candidates(request.query, k=200, document_id=request.document_id)
        vector_candidates = get_vector_candidates(request.query, k=200, document_id=request.document_id)
        
        if not meili_candidates and not vector_candidates:
            return []
        
        # Apply RRF fusion
        from collections import defaultdict
        scores = defaultdict(float)
        
        for passage_id, rank in meili_candidates:
            scores[passage_id] += rrf(rank)
        
        for passage_id, rank in vector_candidates:
            scores[passage_id] += rrf(rank)
        
        # Sort by score and take top k
        top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:request.k]
        
        if not top_candidates:
            return []
        
        # Apply query-specific boosting
        conn = get_db_connection()
        boosted_candidates = apply_query_boosting(top_candidates, query_analysis, conn)
        
        # Get passage details
        passage_ids = [pid for pid, _ in boosted_candidates]
        passage_details = get_passage_details(conn, passage_ids)
        
        # Create score mapping
        score_map = {pid: score for pid, score in boosted_candidates}
        
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









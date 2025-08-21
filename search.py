#!/usr/bin/env python3
"""
Codex Search Script

Implements hybrid search with RRF fusion, query type detection, and specialized handling.
"""

import os
import sys
import click
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
import re
from typing import List, Dict, Any, Tuple, Optional

load_dotenv()

# RRF parameters
RRF_K = 60

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "codex"),
        user=os.getenv("DB_USER", "codex"),
        password=os.getenv("DB_PASSWORD", "codex")
    )

def rrf(rank: int) -> float:
    """Reciprocal rank fusion score."""
    return 1.0 / (RRF_K + rank)

def detect_query_type(query: str) -> str:
    """Detect query type: 'who', 'factoid', or 'general'."""
    query_lower = query.lower()
    
    if re.search(r'\bwho\s+(is|was)\b', query_lower):
        return 'who'
    elif re.search(r'\bwhat\b.*\b(taverns?|inns?|places?|buildings?)\b', query_lower):
        return 'factoid'
    else:
        return 'general'

def extract_person_from_query(query: str) -> Optional[str]:
    """Extract person name from 'who is/was X' queries."""
    match = re.search(r'\bwho\s+(is|was)\s+([^?]+)', query, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return None

def get_meili_candidates(query: str, k: int = 200, document_id: Optional[str] = None) -> List[Tuple[str, int]]:
    """Get BM25 candidates from Meilisearch."""
    try:
        from meilisearch import Client
        
        client = Client(
            os.getenv("MEILI_URL", "http://localhost:7700"),
            os.getenv("MEILI_MASTER_KEY", "")
        )
        
        index = client.index("passages")
        
        # Build search parameters
        search_params = {
            "q": query,
            "limit": k,
            "attributesToRetrieve": ["id"]
        }
        
        if document_id:
            search_params["filter"] = f"document_id = {document_id}"
        
        response = index.search(query, search_params)
        
        # Extract passage IDs and ranks
        candidates = []
        for i, hit in enumerate(response["hits"]):
            candidates.append((hit["id"], i + 1))  # rank starts at 1
        
        return candidates
        
    except ImportError:
        print("Warning: Meilisearch client not available")
        return []
    except Exception as e:
        print(f"Warning: Failed to get Meilisearch candidates: {e}")
        return []

def get_vector_candidates(query: str, k: int = 200, document_id: Optional[str] = None) -> List[Tuple[str, int]]:
    """Get vector similarity candidates from pgvector."""
    try:
        # Get query embedding
        query_embedding = get_query_embedding(query)
        if not query_embedding:
            return []
        
        conn = get_db_connection()
        
        # Build query
        sql = """
            SELECT id, 1 - (embedding <=> %s) as similarity
            FROM passages 
            WHERE embedding IS NOT NULL
        """
        params = [query_embedding]
        
        if document_id:
            sql += " AND document_id = %s"
            params.append(document_id)
        
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.extend([query_embedding, k])
        
        with conn.cursor() as cur:
            cur.execute(sql, params)
            results = cur.fetchall()
        
        conn.close()
        
        # Return passage IDs with ranks
        return [(row[0], i + 1) for i, row in enumerate(results)]
        
    except Exception as e:
        print(f"Warning: Failed to get vector candidates: {e}")
        return []

def get_query_embedding(query: str) -> Optional[List[float]]:
    """Get embedding for query text."""
    try:
        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=query,
                encoding_format="float"
            )
            return response.data[0].embedding
        
        # Fallback to local model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/e5-base-v2")
        embedding = model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()
        
    except Exception as e:
        print(f"Warning: Failed to get query embedding: {e}")
        return None

def apply_query_boosting(candidates: List[Tuple[str, float]], query: str, conn) -> List[Tuple[str, float]]:
    """Apply query-specific boosting."""
    query_type = detect_query_type(query)
    
    if query_type == 'who':
        person = extract_person_from_query(query)
        if person:
            # Boost passages with PERSON entity matching the query
            boosted_candidates = []
            
            for passage_id, score in candidates:
                # Check if passage has matching person entity
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 FROM passage_entities 
                        WHERE passage_id = %s 
                        AND ent_type = 'PERSON' 
                        AND norm_entity = %s
                    """, (passage_id, person.lower()))
                    
                    if cur.fetchone():
                        # Boost score for matching person
                        boosted_score = score * 1.5
                        boosted_candidates.append((passage_id, boosted_score))
                    else:
                        boosted_candidates.append((passage_id, score))
            
            return boosted_candidates
    
    return candidates

def get_passage_details(conn, passage_ids: List[str]) -> List[Dict[str, Any]]:
    """Get detailed passage information."""
    if not passage_ids:
        return []
    
    placeholders = ','.join(['%s'] * len(passage_ids))
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT p.id, p.text, p.page, d.title, p.headings_path
            FROM passages p
            JOIN documents d ON p.document_id = d.id
            WHERE p.id IN ({placeholders})
            ORDER BY p.id
        """, passage_ids)
        
        return cur.fetchall()

def format_results(results: List[Dict[str, Any]], scores: Dict[str, float]) -> str:
    """Format search results for display."""
    output = []
    
    for i, result in enumerate(results, 1):
        passage_id = result['id']
        score = scores.get(passage_id, 0.0)
        title = result['title']
        page = result['page']
        
        # Truncate text for snippet
        text = result['text']
        snippet = text[:200] + "..." if len(text) > 200 else text
        
        output.append(f"{i:2d}  {score:.3f}  {title}  p. {page}  \"{snippet}\"")
    
    return '\n'.join(output)

@click.command()
@click.option('--q', 'query', required=True, help='Search query')
@click.option('--k', default=20, type=int, help='Number of results to return')
@click.option('--doc', 'document_id', help='Filter by document ID')
def main(query: str, k: int, document_id: Optional[str]):
    """Search passages using hybrid retrieval with RRF fusion."""
    
    print(f"Searching for: {query}")
    print(f"Query type: {detect_query_type(query)}")
    
    # Get candidates from both sources
    print("Getting BM25 candidates...")
    meili_candidates = get_meili_candidates(query, k=200, document_id=document_id)
    print(f"Found {len(meili_candidates)} BM25 candidates")
    
    print("Getting vector candidates...")
    vector_candidates = get_vector_candidates(query, k=200, document_id=document_id)
    print(f"Found {len(vector_candidates)} vector candidates")
    
    if not meili_candidates and not vector_candidates:
        print("No candidates found")
        return
    
    # Apply RRF fusion
    print("Applying RRF fusion...")
    scores = defaultdict(float)
    
    for passage_id, rank in meili_candidates:
        scores[passage_id] += rrf(rank)
    
    for passage_id, rank in vector_candidates:
        scores[passage_id] += rrf(rank)
    
    # Sort by score and take top k
    top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    if not top_candidates:
        print("No results after fusion")
        return
    
    # Apply query-specific boosting
    conn = get_db_connection()
    boosted_candidates = apply_query_boosting(top_candidates, query, conn)
    
    # Get passage details
    passage_ids = [pid for pid, _ in boosted_candidates]
    passage_details = get_passage_details(conn, passage_ids)
    
    # Create score mapping
    score_map = {pid: score for pid, score in boosted_candidates}
    
    # Format and display results
    print(f"\nTop {len(passage_details)} results:")
    print("=" * 80)
    print(format_results(passage_details, score_map))
    
    conn.close()

if __name__ == "__main__":
    main()









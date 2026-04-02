#!/usr/bin/env python3
"""
Codex Embedding Generation Script

Generates embeddings for passages using OpenAI or local models.
"""

import os
import sys
import click
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Any, Optional
import tiktoken

from config import get_db_connection, get_openai_client, EMBEDDING_MODEL

def get_openai_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Get embeddings from OpenAI API."""
    try:
        client = get_openai_client()
        
        # OpenAI API has a limit of 8192 tokens per request
        # We'll batch requests if needed
        embeddings = []
        batch_size = 100  # Conservative batch size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float"
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return embeddings
        
    except ImportError:
        print("Error: OpenAI client not available. Install with: pip install openai")
        return []
    except Exception as e:
        print(f"Error getting OpenAI embeddings: {e}")
        return []

def get_local_embeddings(texts: List[str], model_name: str = "intfloat/e5-base-v2") -> List[List[float]]:
    """Get embeddings using local sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # Convert to list of lists
        return embeddings.tolist()
        
    except ImportError:
        print("Error: sentence-transformers not available. Install with: pip install sentence-transformers")
        return []
    except Exception as e:
        print(f"Error getting local embeddings: {e}")
        return []

def get_unembedded_passages(conn, limit: int = 1000, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get passages that don't have embeddings yet."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if document_id:
            cur.execute("""
                SELECT id, text FROM passages 
                WHERE embedding IS NULL AND document_id = %s
                ORDER BY id 
                LIMIT %s
            """, (document_id, limit))
        else:
            cur.execute("""
                SELECT id, text FROM passages 
                WHERE embedding IS NULL 
                ORDER BY id 
                LIMIT %s
            """, (limit,))
        return cur.fetchall()

def update_passage_embeddings(conn, passage_embeddings: List[tuple]):
    """Update passage embeddings in database."""
    with conn.cursor() as cur:
        for passage_id, embedding in passage_embeddings:
            cur.execute("""
                UPDATE passages 
                SET embedding = %s 
                WHERE id = %s
            """, (embedding, passage_id))
    
    conn.commit()

def validate_embedding_dimension(embedding: List[float], expected_dim: int = 1536) -> bool:
    """Validate that embedding has the expected dimension."""
    return len(embedding) == expected_dim

def run_embeddings(document_ids: Optional[List[str]] = None, provider: str = "openai",
                   model: str = EMBEDDING_MODEL, batch_size: int = 100,
                   limit: int = 1000) -> bool:
    """
    Generate embeddings for passages that don't have them yet.

    Can be called directly from Python (e.g. batch_reingest) or via the CLI.
    Returns True on success.
    """
    conn = get_db_connection()
    print("Connected to database")

    if document_ids:
        print(f"Processing documents: {', '.join(document_ids)}")
        all_passages = []
        for doc_id in document_ids:
            all_passages.extend(get_unembedded_passages(conn, limit, doc_id))
        passages = all_passages[:limit]
    else:
        passages = get_unembedded_passages(conn, limit)

    if not passages:
        print("No passages found without embeddings")
        conn.close()
        return True

    print(f"Found {len(passages)} passages without embeddings")

    texts = [p['text'] for p in passages]
    passage_ids = [p['id'] for p in passages]

    print(f"Generating embeddings using {provider} provider...")
    if provider == 'openai':
        embeddings = get_openai_embeddings(texts, model)
    else:
        embeddings = get_local_embeddings(texts, model)

    if not embeddings:
        print("Failed to generate embeddings")
        conn.close()
        return False

    print(f"Generated {len(embeddings)} embeddings")
    if embeddings:
        dim = len(embeddings[0])
        print(f"Embedding dimension: {dim}")
        if not validate_embedding_dimension(embeddings[0]):
            print(f"Warning: Expected dimension 1536, got {dim}")

    print("Updating database...")
    passage_embeddings = list(zip(passage_ids, embeddings))
    for i in range(0, len(passage_embeddings), batch_size):
        batch = passage_embeddings[i:i + batch_size]
        update_passage_embeddings(conn, batch)
        print(f"Updated batch {i//batch_size + 1}/{(len(passage_embeddings) + batch_size - 1)//batch_size}")

    remaining = get_unembedded_passages(conn, 1)
    if remaining:
        print(f"Note: {len(remaining)} passages still need embeddings")
    else:
        print("All passages now have embeddings!")

    conn.close()
    print("Embedding generation complete!")
    return True

@click.command()
@click.option('--provider', default='openai', type=click.Choice(['openai', 'local']),
              help='Embedding provider')
@click.option('--model', default=EMBEDDING_MODEL,
              help='Model name (for OpenAI) or model path (for local)')
@click.option('--batch-size', default=100, type=int,
              help='Batch size for processing')
@click.option('--limit', default=1000, type=int,
              help='Maximum number of passages to process')
@click.option('--doc', 'document_id', help='Process only specific document ID')
@click.option('--documents', help='Comma-separated list of document IDs')
def main(provider: str, model: str, batch_size: int, limit: int, document_id: Optional[str], documents: Optional[str]):
    """Generate embeddings for passages that don't have them yet."""
    doc_ids = None
    if document_id:
        doc_ids = [document_id]
    elif documents:
        doc_ids = [d.strip() for d in documents.split(',')]

    success = run_embeddings(doc_ids, provider, model, batch_size, limit)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()









#!/usr/bin/env python3
"""
Codex Embedding Generation Script

Generates embeddings for passages using OpenAI or local models.
"""

import os
import sys
import click
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any, Optional
import tiktoken

load_dotenv()

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "codex"),
        user=os.getenv("DB_USER", "codex"),
        password=os.getenv("DB_PASSWORD", "codex")
    )

def get_openai_embeddings(texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    """Get embeddings from OpenAI API."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
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

def validate_embedding_dimension(embedding: List[float], expected_dim: int = 1024) -> bool:
    """Validate that embedding has the expected dimension."""
    return len(embedding) == expected_dim

@click.command()
@click.option('--provider', default='openai', type=click.Choice(['openai', 'local']), 
              help='Embedding provider')
@click.option('--model', default='text-embedding-3-small', 
              help='Model name (for OpenAI) or model path (for local)')
@click.option('--batch-size', default=100, type=int, 
              help='Batch size for processing')
@click.option('--limit', default=1000, type=int, 
              help='Maximum number of passages to process')
@click.option('--doc', 'document_id', help='Process only specific document ID')
@click.option('--documents', help='Comma-separated list of document IDs')
def main(provider: str, model: str, batch_size: int, limit: int, document_id: Optional[str], documents: Optional[str]):
    """Generate embeddings for passages that don't have them yet."""
    
    # Connect to database
    try:
        conn = get_db_connection()
        print("Connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Handle document filtering
    target_document_ids = []
    if document_id:
        target_document_ids = [document_id]
    elif documents:
        target_document_ids = [doc_id.strip() for doc_id in documents.split(',')]
    
    # Get unembedded passages
    if target_document_ids:
        print(f"Processing documents: {', '.join(target_document_ids)}")
        all_passages = []
        for doc_id in target_document_ids:
            doc_passages = get_unembedded_passages(conn, limit, doc_id)
            all_passages.extend(doc_passages)
        passages = all_passages[:limit]  # Respect overall limit
    else:
        passages = get_unembedded_passages(conn, limit)
    
    if not passages:
        print("No passages found without embeddings")
        conn.close()
        return
    
    print(f"Found {len(passages)} passages without embeddings")
    
    # Extract text for embedding
    texts = [p['text'] for p in passages]
    passage_ids = [p['id'] for p in passages]
    
    # Generate embeddings
    print(f"Generating embeddings using {provider} provider...")
    
    if provider == 'openai':
        embeddings = get_openai_embeddings(texts, model)
    else:  # local
        embeddings = get_local_embeddings(texts, model)
    
    if not embeddings:
        print("Failed to generate embeddings")
        conn.close()
        sys.exit(1)
    
    # Validate embeddings
    print(f"Generated {len(embeddings)} embeddings")
    
    # Check dimension
    if embeddings:
        dim = len(embeddings[0])
        print(f"Embedding dimension: {dim}")
        
        if not validate_embedding_dimension(embeddings[0]):
            print(f"Warning: Expected dimension 1024, got {dim}")
    
    # Update database
    print("Updating database...")
    passage_embeddings = list(zip(passage_ids, embeddings))
    
    # Process in batches
    for i in range(0, len(passage_embeddings), batch_size):
        batch = passage_embeddings[i:i + batch_size]
        update_passage_embeddings(conn, batch)
        print(f"Updated batch {i//batch_size + 1}/{(len(passage_embeddings) + batch_size - 1)//batch_size}")
    
    # Verify completion
    remaining = get_unembedded_passages(conn, 1)
    if remaining:
        print(f"Note: {len(remaining)} passages still need embeddings")
    else:
        print("All passages now have embeddings!")
    
    conn.close()
    print("Embedding generation complete!")

if __name__ == "__main__":
    main()









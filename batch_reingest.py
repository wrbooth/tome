#!/usr/bin/env python3
"""
Codex Batch Re-ingestion Script

Handles re-ingestion of multiple documents with options to:
- Clear database and Meilisearch first
- Re-ingest specific documents
- Re-ingest all documents
- Selective clearing
"""

import os
import sys
import click
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import subprocess
from pathlib import Path

from config import get_db_connection, get_meili_client

def clear_database():
    """Clear all data from the database."""
    print("Clearing database...")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Clear passages first (due to foreign key constraint)
            cur.execute("DELETE FROM passages")
            print(f"  Deleted {cur.rowcount} passages")
            
            # Clear documents
            cur.execute("DELETE FROM documents")
            print(f"  Deleted {cur.rowcount} documents")
            
            conn.commit()
            print("Database cleared successfully")
    except Exception as e:
        print(f"Error clearing database: {e}")
        conn.rollback()
    finally:
        conn.close()

def clear_documents(documents: List[str]):
    """Clear specific documents from the database."""
    print(f"Clearing {len(documents)} documents from database...")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for doc_id in documents:
                # Delete passages first
                cur.execute("DELETE FROM passages WHERE document_id = %s", (doc_id,))
                passage_count = cur.rowcount
                
                # Delete document
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                doc_count = cur.rowcount
                
                print(f"  Deleted document {doc_id}: {doc_count} document, {passage_count} passages")
            
            conn.commit()
            print("Documents cleared successfully")
    except Exception as e:
        print(f"Error clearing documents: {e}")
        conn.rollback()
    finally:
        conn.close()

def clear_meilisearch():
    """Clear all data from Meilisearch."""
    print("Clearing Meilisearch...")
    
    try:
        client = get_meili_client()

        # Delete the index if it exists
        try:
            client.index("passages").delete()
            print("  Deleted Meilisearch index")
        except Exception as e:
            if "not found" in str(e).lower():
                print("  Meilisearch index already empty")
            else:
                print(f"  Error deleting Meilisearch index: {e}")
                
    except Exception as e:
        print(f"Error connecting to Meilisearch: {e}")

def get_document_info(conn, document_id: str) -> Optional[Dict[str, Any]]:
    """Get document information."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, title, source_path, authors, pub_year
            FROM documents 
            WHERE id = %s
        """, (document_id,))
        return cur.fetchone()

def list_all_documents() -> List[Dict[str, Any]]:
    """List all documents in the database."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, source_path, authors, pub_year
                FROM documents 
                ORDER BY title
            """)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

def reingest_document(file_path: str, title: str, authors: Optional[str] = None, 
                     pub_year: Optional[int] = None, debug: bool = False) -> bool:
    """Re-ingest a single document."""
    print(f"Re-ingesting document: {Path(file_path).name}")
    
    try:
        # Build command
        cmd = [
            sys.executable, "ingest.py", 
            file_path,
            "--title", title
        ]
        
        if authors:
            cmd.extend(["--authors", authors])
        
        if pub_year:
            cmd.extend(["--pub-year", str(pub_year)])
        
        if debug:
            cmd.append("--debug")
        
        # Run the ingest command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully re-ingested: {Path(file_path).name}")
            return True
        else:
            print(f"❌ Error re-ingesting {Path(file_path).name}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Exception re-ingesting {Path(file_path).name}: {e}")
        return False

def run_embedding(document_ids: Optional[List[str]] = None):
    """Run the embedding process."""
    print("Running embedding process...")
    
    try:
        # Build command
        cmd = [sys.executable, "embed.py"]
        
        if document_ids:
            # Process specific documents
            for doc_id in document_ids:
                print(f"  Generating embeddings for document: {doc_id}")
                doc_cmd = cmd + ["--doc", doc_id]
                result = subprocess.run(doc_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Embeddings completed for {doc_id}")
                else:
                    print(f"❌ Error generating embeddings for {doc_id}:")
                    print(result.stderr)
        else:
            # Process all documents
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Embedding completed successfully")
            else:
                print("❌ Error during embedding:")
                print(result.stderr)
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error running embedding: {e}")
        return False

@click.command()
@click.argument('documents', nargs=-1)
@click.option('--all', is_flag=True, help='Re-ingest all documents')
@click.option('--clear-first', is_flag=True, help='Clear database and Meilisearch first')
@click.option('--clear-docs', help='Comma-separated list of document IDs to clear')
@click.option('--skip-embedding', is_flag=True, help='Skip embedding generation')
@click.option('--debug', is_flag=True, help='Show debug information')
def main(documents: List[str], all: bool, clear_first: bool, clear_docs: Optional[str], 
         skip_embedding: bool, debug: bool):
    """Batch re-ingest documents."""
    
    print("=== Codex Batch Re-ingestion ===")
    
    # Determine which documents to process
    if all:
        print("Re-ingesting all documents...")
        all_docs = list_all_documents()
        if not all_docs:
            print("No documents found in database")
            return
        
        documents_to_process = all_docs
        print(f"Found {len(documents_to_process)} documents to re-ingest")
        
    elif documents:
        print(f"Re-ingesting {len(documents)} specific documents...")
        conn = get_db_connection()
        documents_to_process = []
        
        for doc_id in documents:
            doc_info = get_document_info(conn, doc_id)
            if doc_info:
                documents_to_process.append(doc_info)
            else:
                print(f"Warning: Document {doc_id} not found in database")
        
        conn.close()
        
        if not documents_to_process:
            print("No valid documents found")
            return
    else:
        print("Error: Must specify documents to re-ingest or use --all")
        return
    
    # Clear operations
    if clear_first:
        print("\nClearing database and Meilisearch...")
        clear_database()
        clear_meilisearch()
    elif clear_docs:
        doc_ids_to_clear = [doc_id.strip() for doc_id in clear_docs.split(',')]
        print(f"\nClearing specific documents: {', '.join(doc_ids_to_clear)}")
        clear_documents(doc_ids_to_clear)
    
    # Re-ingest documents
    print(f"\nRe-ingesting {len(documents_to_process)} documents...")
    successful = 0
    failed = 0
    
    for i, doc in enumerate(documents_to_process, 1):
        print(f"\n[{i}/{len(documents_to_process)}] Processing: {doc['title']}")
        
        # Check if source file exists
        if not os.path.exists(doc['source_path']):
            print(f"❌ Source file not found: {doc['source_path']}")
            failed += 1
            continue
        
        # Prepare metadata
        authors = ', '.join(doc['authors']) if doc['authors'] else None
        pub_year = doc['pub_year']
        
        # Re-ingest
        success = reingest_document(
            doc['source_path'], 
            doc['title'], 
            authors, 
            pub_year, 
            debug
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n=== Re-ingestion Summary ===")
    print(f"Total: {len(documents_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed documents:")
        for i, doc in enumerate(documents_to_process):
            if not os.path.exists(doc['source_path']):
                print(f"  - {doc['title']}: Source file not found")
    
    # Run embedding if requested
    if not skip_embedding and successful > 0:
        print(f"\nGenerating embeddings...")
        
        if all:
            # Generate embeddings for all documents
            embedding_success = run_embedding()
        else:
            # Generate embeddings for specific documents
            doc_ids = [doc['id'] for doc in documents_to_process]
            embedding_success = run_embedding(doc_ids)
        
        if embedding_success:
            print("✅ Embedding generation completed")
        else:
            print("❌ Embedding generation failed")
    
    print(f"\n=== Re-ingestion Complete ===")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()





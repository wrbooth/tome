#!/usr/bin/env python3
"""
Codex Document Management CLI

Provides commands for managing documents in the system:
- List documents with stats
- Delete documents
- Re-index documents in Meilisearch
- Get document details
"""

import os
import sys
import click
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json
from tabulate import tabulate

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

def get_document_stats(conn, document_id: str) -> Dict[str, Any]:
    """Get detailed stats for a document."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Document info
        cur.execute("""
            SELECT id, title, authors, pub_year, source_path
            FROM documents 
            WHERE id = %s
        """, (document_id,))
        doc = cur.fetchone()
        
        if not doc:
            return None
        
        # Passage count
        cur.execute("""
            SELECT COUNT(*) as total_passages,
                   COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedded_passages
            FROM passages 
            WHERE document_id = %s
        """, (document_id,))
        passage_stats = cur.fetchone()
        
        # Entity count
        cur.execute("""
            SELECT COUNT(*) as entity_count
            FROM passage_entities pe
            JOIN passages p ON pe.passage_id = p.id
            WHERE p.document_id = %s
        """, (document_id,))
        entity_stats = cur.fetchone()
        
        # Year count
        cur.execute("""
            SELECT COUNT(*) as year_count
            FROM passage_years py
            JOIN passages p ON py.passage_id = p.id
            WHERE p.document_id = %s
        """, (document_id,))
        year_stats = cur.fetchone()
        
        # Page range
        cur.execute("""
            SELECT MIN(page) as min_page, MAX(page) as max_page
            FROM passages 
            WHERE document_id = %s
        """, (document_id,))
        page_stats = cur.fetchone()
        
        return {
            'document': dict(doc),
            'passages': dict(passage_stats),
            'entities': dict(entity_stats),
            'years': dict(year_stats),
            'pages': dict(page_stats)
        }

def list_documents(conn, format: str = 'table') -> List[Dict[str, Any]]:
    """List all documents with basic stats."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                d.id,
                d.title,
                d.authors,
                d.pub_year,
                COUNT(p.id) as passage_count,
                COUNT(CASE WHEN p.embedding IS NOT NULL THEN 1 END) as embedded_count,
                MIN(p.page) as min_page,
                MAX(p.page) as max_page
            FROM documents d
            LEFT JOIN passages p ON d.id = p.document_id
            GROUP BY d.id, d.title, d.authors, d.pub_year
            ORDER BY d.title
        """)
        return [dict(row) for row in cur.fetchall()]

def delete_document(conn, document_id: str, confirm: bool = True) -> bool:
    """Delete a document and all its passages."""
    if confirm:
        # Get document info for confirmation
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT title FROM documents WHERE id = %s", (document_id,))
            doc = cur.fetchone()
            if not doc:
                print(f"Document {document_id} not found")
                return False
            
            # Get passage count
            cur.execute("SELECT COUNT(*) FROM passages WHERE document_id = %s", (document_id,))
            passage_count = cur.fetchone()[0]
            
            print(f"About to delete document: {doc['title']}")
            print(f"This will also delete {passage_count} passages")
            
            if not click.confirm("Are you sure you want to continue?"):
                print("Deletion cancelled")
                return False
    
    try:
        with conn.cursor() as cur:
            # Delete passages first (cascade will handle related tables)
            cur.execute("DELETE FROM passages WHERE document_id = %s", (document_id,))
            passage_deleted = cur.rowcount
            
            # Delete document
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            doc_deleted = cur.rowcount
            
            conn.commit()
            
            print(f"Deleted {doc_deleted} document and {passage_deleted} passages")
            return True
            
    except Exception as e:
        conn.rollback()
        print(f"Error deleting document: {e}")
        return False

def reindex_document_meilisearch(conn, document_id: str) -> bool:
    """Re-index a document in Meilisearch."""
    try:
        from meilisearch import Client
        
        # Get document passages
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT p.id, p.text, p.page, p.headings_path, d.title
                FROM passages p
                JOIN documents d ON p.document_id = d.id
                WHERE p.document_id = %s
            """, (document_id,))
            passages = cur.fetchall()
        
        if not passages:
            print(f"No passages found for document {document_id}")
            return False
        
        # Prepare documents for indexing
        from ingest import extract_entities_and_years
        
        documents = []
        for passage in passages:
            # Extract entities and years
            _, years = extract_entities_and_years(passage['text'])
            entities, _ = extract_entities_and_years(passage['text'])
            
            person_entities = [e["entity"] for e in entities if e["ent_type"] == "PERSON"]
            place_entities = [e["entity"] for e in entities if e["ent_type"] in ["GPE", "FAC"]]
            
            documents.append({
                "id": passage['id'],
                "document_id": document_id,
                "text": passage['text'],
                "page": passage['page'],
                "headings_path": passage['headings_path'],
                "years": years,
                "entities_person": person_entities,
                "entities_place": place_entities
            })
        
        # Index in Meilisearch
        client = Client(os.getenv("MEILI_URL", "http://localhost:7700"))
        index = client.index("passages")
        index.add_documents(documents)
        
        print(f"Re-indexed {len(documents)} passages for document {document_id}")
        return True
        
    except ImportError:
        print("Warning: Meilisearch client not available")
        return False
    except Exception as e:
        print(f"Error re-indexing document: {e}")
        return False

@click.group()
def cli():
    """Codex Document Management CLI."""
    pass

@cli.command()
@click.option('--format', default='table', type=click.Choice(['table', 'json']), 
              help='Output format')
def list(format: str):
    """List all documents with stats."""
    try:
        conn = get_db_connection()
        documents = list_documents(conn, format)
        
        if format == 'json':
            print(json.dumps(documents, indent=2))
        else:
            if not documents:
                print("No documents found")
                return
            
            # Prepare table data
            table_data = []
            for doc in documents:
                authors = ', '.join(doc['authors']) if doc['authors'] else 'Unknown'
                embedding_pct = f"{(doc['embedded_count']/doc['passage_count']*100):.1f}%" if doc['passage_count'] > 0 else "0%"
                page_range = f"{doc['min_page']}-{doc['max_page']}" if doc['min_page'] and doc['max_page'] else "N/A"
                
                table_data.append([
                    doc['id'][:8] + '...',
                    doc['title'][:50] + ('...' if len(doc['title']) > 50 else ''),
                    authors[:30] + ('...' if len(authors) > 30 else ''),
                    doc['pub_year'] or 'Unknown',
                    doc['passage_count'],
                    f"{doc['embedded_count']} ({embedding_pct})",
                    page_range
                ])
            
            headers = ['ID', 'Title', 'Authors', 'Year', 'Passages', 'Embedded', 'Pages']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
            
    except Exception as e:
        print(f"Error listing documents: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

@cli.command()
@click.argument('document_id')
@click.option('--format', default='table', type=click.Choice(['table', 'json']), 
              help='Output format')
def info(document_id: str, format: str):
    """Get detailed information about a document."""
    try:
        conn = get_db_connection()
        stats = get_document_stats(conn, document_id)
        
        if not stats:
            print(f"Document {document_id} not found")
            sys.exit(1)
        
        if format == 'json':
            print(json.dumps(stats, indent=2))
        else:
            doc = stats['document']
            passages = stats['passages']
            entities = stats['entities']
            years = stats['years']
            pages = stats['pages']
            
            print(f"Document: {doc['title']}")
            print(f"ID: {doc['id']}")
            print(f"Authors: {', '.join(doc['authors']) if doc['authors'] else 'Unknown'}")
            print(f"Year: {doc['pub_year'] or 'Unknown'}")
            print(f"Source: {doc['source_path']}")
            print()
            
            print("Statistics:")
            print(f"  Passages: {passages['total_passages']} total, {passages['embedded_passages']} embedded")
            print(f"  Entities: {entities['entity_count']}")
            print(f"  Years: {years['year_count']}")
            print(f"  Pages: {pages['min_page']} - {pages['max_page']}")
            
            if passages['total_passages'] > 0:
                embedding_pct = (passages['embedded_passages'] / passages['total_passages']) * 100
                print(f"  Embedding coverage: {embedding_pct:.1f}%")
            
    except Exception as e:
        print(f"Error getting document info: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

@cli.command()
@click.argument('document_id')
@click.option('--force', is_flag=True, help='Skip confirmation')
def delete(document_id: str, force: bool):
    """Delete a document and all its passages."""
    try:
        conn = get_db_connection()
        success = delete_document(conn, document_id, confirm=not force)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error deleting document: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

@cli.command()
@click.argument('document_id')
def reindex(document_id: str):
    """Re-index a document in Meilisearch."""
    try:
        conn = get_db_connection()
        success = reindex_document_meilisearch(conn, document_id)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error re-indexing document: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

@cli.command()
def stats():
    """Get overall system statistics."""
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
            
            # Year count
            cur.execute("SELECT COUNT(*) FROM passage_years")
            year_count = cur.fetchone()[0]
        
        print("System Statistics:")
        print(f"  Documents: {doc_count}")
        print(f"  Passages: {passage_count}")
        print(f"  Embedded passages: {embedded_count}")
        print(f"  Entities: {entity_count}")
        print(f"  Years: {year_count}")
        
        if passage_count > 0:
            embedding_pct = (embedded_count / passage_count) * 100
            print(f"  Embedding coverage: {embedding_pct:.1f}%")
            
    except Exception as e:
        print(f"Error getting system stats: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    cli()





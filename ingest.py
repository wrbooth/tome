#!/usr/bin/env python3
"""
Codex Document Ingestion Script

Handles PDF/TXT ingestion, text extraction, chunking, and database storage.
"""

import os
import sys
import uuid
import click
import fitz  # PyMuPDF
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import spacy
import re
from typing import List, Dict, Any, Tuple
import json

load_dotenv()

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "codex"),
        user=os.getenv("DB_USER", "codex"),
        password=os.getenv("DB_PASSWORD", "codex")
    )

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page numbers."""
    pages = []
    doc = fitz.open(file_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pages.append({
            "page": page_num + 1,
            "text": text,
            "blocks": []  # TODO: Add block extraction if needed
        })
    
    doc.close()
    return pages

def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from TXT file, treating each line as a page."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines to simulate pages
    sections = content.split('\n\n')
    pages = []
    
    for i, section in enumerate(sections):
        if section.strip():
            pages.append({
                "page": i + 1,
                "text": section.strip(),
                "blocks": []
            })
    
    return pages

def chunk_text(pages: List[Dict[str, Any]], max_tokens: int = 500, overlap: float = 0.1) -> List[Dict[str, Any]]:
    """Chunk text into passages of approximately max_tokens, allowing cross-page chunks."""
    chunks = []
    
    # Collect all paragraphs with their page numbers
    all_paragraphs = []
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        for paragraph in paragraphs:
            all_paragraphs.append({
                "page": page_num,
                "text": paragraph,
                "tokens": len(paragraph.split()) * 1.3  # Rough token estimation
            })
    
    # Create chunks across pages
    current_chunk = []
    current_tokens = 0
    chunk_start_page = None
    
    for i, para in enumerate(all_paragraphs):
        # If adding this paragraph would exceed max_tokens, save current chunk
        if current_tokens + para["tokens"] > max_tokens and current_chunk:
            # Save current chunk
            chunk_text = ' '.join([p["text"] for p in current_chunk])
            chunks.append({
                "page": chunk_start_page,
                "text": chunk_text,
                "char_start": 0,
                "char_end": len(chunk_text),
                "headings_path": []
            })
            
            # Start new chunk with minimal overlap (just the last paragraph)
            if len(current_chunk) > 0:
                current_chunk = [current_chunk[-1]]
                current_tokens = current_chunk[0]["tokens"]
                chunk_start_page = current_chunk[0]["page"]
            else:
                current_chunk = []
                current_tokens = 0
                chunk_start_page = para["page"]
        
        if not current_chunk:
            chunk_start_page = para["page"]
        
        current_chunk.append(para)
        current_tokens += para["tokens"]
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join([p["text"] for p in current_chunk])
        chunks.append({
            "page": chunk_start_page,
            "text": chunk_text,
            "char_start": 0,
            "char_end": len(chunk_text),
            "headings_path": []
        })
    
    return chunks

def extract_entities_and_years(text: str) -> Tuple[List[Dict[str, str]], List[int]]:
    """Extract entities and years from text."""
    entities = []
    years = []
    
    # Extract years using regex
    year_pattern = r'\b(1[4-9]\d{2}|20\d{2})\b'
    years = [int(year) for year in re.findall(year_pattern, text) if year.strip()]
    
    # Extract entities using spaCy
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({
                "entity": ent.text,
                "ent_type": ent.label_,
                "norm_entity": ent.text.lower()
            })
    
    return entities, years

def store_document(conn, title: str, source_path: str, authors: List[str] = None, pub_year: int = None) -> str:
    """Store document in database and return document ID."""
    doc_id = str(uuid.uuid4())
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (id, title, authors, pub_year, source_path)
            VALUES (%s, %s, %s, %s, %s)
        """, (doc_id, title, authors, pub_year, source_path))
    
    conn.commit()
    return doc_id

def store_passages(conn, doc_id: str, chunks: List[Dict[str, Any]]) -> List[str]:
    """Store passages in database and return passage IDs."""
    passage_ids = []
    
    with conn.cursor() as cur:
        for chunk in chunks:
            passage_id = str(uuid.uuid4())
            passage_ids.append(passage_id)
            
            cur.execute("""
                INSERT INTO passages (id, document_id, page, text, headings_path, char_start, char_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                passage_id,
                doc_id,
                chunk["page"],
                chunk["text"],
                chunk["headings_path"],  # Pass the list directly
                chunk["char_start"],
                chunk["char_end"]
            ))
            
            # Extract and store entities and years
            entities, years = extract_entities_and_years(chunk["text"])
            
            for entity in entities:
                cur.execute("""
                    INSERT INTO passage_entities (passage_id, entity, ent_type, norm_entity)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (passage_id, entity) DO NOTHING
                """, (passage_id, entity["entity"], entity["ent_type"], entity["norm_entity"]))
            
            for year in years:
                cur.execute("""
                    INSERT INTO passage_years (passage_id, year)
                    VALUES (%s, %s)
                    ON CONFLICT (passage_id, year) DO NOTHING
                """, (passage_id, year))
    
    conn.commit()
    return passage_ids

def index_in_meilisearch(passages: List[Dict[str, Any]], passage_ids: List[str]):
    """Index passages in Meilisearch."""
    try:
        from meilisearch import Client
        
        client = Client(
            os.getenv("MEILI_URL", "http://localhost:7700")
        )
        
        # Prepare documents for indexing
        documents = []
        for passage, passage_id in zip(passages, passage_ids):
            # Extract years and entities for filtering
            _, years = extract_entities_and_years(passage["text"])
            entities, _ = extract_entities_and_years(passage["text"])
            
            person_entities = [e["entity"] for e in entities if e["ent_type"] == "PERSON"]
            place_entities = [e["entity"] for e in entities if e["ent_type"] in ["GPE", "FAC"]]
            
            documents.append({
                "id": passage_id,
                "text": passage["text"],
                "page": passage["page"],
                "headings_path": passage["headings_path"],
                "years": years,
                "entities_person": person_entities,
                "entities_place": place_entities
            })
        
        # Create or update index
        index = client.index("passages")
        index.add_documents(documents)
        
        print(f"Indexed {len(documents)} passages in Meilisearch")
        
    except ImportError:
        print("Warning: Meilisearch client not available")
    except Exception as e:
        print(f"Warning: Failed to index in Meilisearch: {e}")

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--title', help='Document title')
@click.option('--authors', help='Comma-separated list of authors')
@click.option('--pub-year', type=int, help='Publication year')
def main(file_path: str, title: str, authors: str, pub_year: int):
    """Ingest a document (PDF or TXT) into the Codex system."""
    
    # Determine file type and extract text
    if file_path.lower().endswith('.pdf'):
        print(f"Extracting text from PDF: {file_path}")
        pages = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        print(f"Extracting text from TXT: {file_path}")
        pages = extract_text_from_txt(file_path)
    else:
        print("Error: Unsupported file type. Use PDF or TXT files.")
        sys.exit(1)
    
    print(f"Extracted {len(pages)} pages")
    
    # Chunk the text
    print("Chunking text...")
    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} chunks")
    
    # Connect to database
    try:
        conn = get_db_connection()
        print("Connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Store document
    if not title:
        title = os.path.basename(file_path)
    
    author_list = [a.strip() for a in authors.split(',')] if authors else None
    
    doc_id = store_document(conn, title, file_path, author_list, pub_year)
    print(f"Stored document: {title} (ID: {doc_id})")
    
    # Store passages
    passage_ids = store_passages(conn, doc_id, chunks)
    print(f"Stored {len(passage_ids)} passages")
    
    # Index in Meilisearch
    index_in_meilisearch(chunks, passage_ids)
    
    conn.close()
    print("Ingestion complete!")

if __name__ == "__main__":
    main()









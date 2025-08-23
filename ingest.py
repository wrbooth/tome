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
from typing import List, Dict, Any, Tuple, Optional
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

def detect_heading_patterns(text: str, only_level_1: bool = False) -> List[Dict[str, Any]]:
    """Detect headings using regex patterns."""
    headings = []
    
    # Common heading patterns with their levels
    patterns = [
        # Level 1: Main chapters/sections
        (r'^CHAPTER\s+\d+[:\s]*(.+)$', 1),  # CHAPTER 1: Title
        (r'^Chapter\s+\d+[:\s]*(.+)$', 1),  # Chapter 1: Title
        (r'^PART\s+\d+[:\s]*(.+)$', 1),     # PART 1: Title
        (r'^Part\s+\d+[:\s]*(.+)$', 1),     # Part 1: Title
        (r'^BOOK\s+\d+[:\s]*(.+)$', 1),     # BOOK 1: Title
        (r'^Book\s+\d+[:\s]*(.+)$', 1),     # Book 1: Title
        (r'^APPENDIX\s+[A-Z][:\s]*(.+)$', 1),  # APPENDIX A: Title
        (r'^Appendix\s+[A-Z][:\s]*(.+)$', 1),  # Appendix A: Title
        (r'^[A-Z][A-Z\s]{8,}$', 1),              # Long ALL CAPS TITLE (8+ chars, major sections)
    ]
    
    # Add more detailed patterns only if not restricting to level 1
    if not only_level_1:
        patterns.extend([
            # Level 2: Subsections
            (r'^Section\s+\d+\.\d+[:\s]*(.+)$', 2),  # Section 1.1: Title
            (r'^SECTION\s+\d+\.\d+[:\s]*(.+)$', 2),  # SECTION 1.1: Title
            (r'^\d+\.\d+\s+([A-Z][^.]*)$', 2),       # 1.1 Title (numbered sections)
            (r'^[A-Z][A-Z\s]{5,7}$', 2),             # Medium ALL CAPS TITLE (5-7 chars)
            
            # Level 3: Subsections
            (r'^\d+\.\d+\.\d+\s+([A-Z][^.]*)$', 3),  # 1.1.1 Title
            (r'^[A-Z][A-Z\s]{3,4}$', 3),             # Short ALL CAPS TITLE (3-4 chars)
            
            # Level 4: Minor headings
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$', 4),  # Title Case
            (r'^\d+\.\s+([A-Z][^.]*)$', 4),              # 1. Title (numbered sections)
        ])
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        for pattern, level in patterns:
            match = re.match(pattern, line)
            if match:
                heading_text = match.group(1) if len(match.groups()) > 0 else line
                # Additional quality check for regex-detected headings
                if is_quality_heading(heading_text, 0, False):
                    headings.append({
                        'text': heading_text.strip(),
                        'line_number': i,
                        'level': level,
                        'full_text': line,
                        'detection_method': 'regex'
                    })
                break
    
    return headings

def extract_text_with_font_info(file_path: str) -> List[Dict[str, Any]]:
    """Extract text with font information to detect headings."""
    pages = []
    doc = fitz.open(file_path)
    
    # Analyze font sizes across all pages to determine heading thresholds
    all_font_sizes = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_font_sizes.append(span["size"])
    
    # Calculate font size thresholds - be more selective for PDFs
    if all_font_sizes:
        font_sizes = sorted(all_font_sizes)
        # Use 85th percentile as threshold for major headings only
        heading_threshold = font_sizes[int(len(font_sizes) * 0.85)]
        # Use 95th percentile for major headings
        major_heading_threshold = font_sizes[int(len(font_sizes) * 0.95)]
    else:
        heading_threshold = 14
        major_heading_threshold = 16
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        page_text = ""
        headings = []
        line_info = []  # Track line-by-line info for heading detection
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    line_font_sizes = []
                    line_is_bold = False
                    
                    for span in line["spans"]:
                        text = span["text"]
                        font_size = span["size"]
                        font_flags = span["flags"]
                        is_bold = bool(font_flags & 2**4)
                        
                        line_text += text
                        line_font_sizes.append(font_size)
                        if is_bold:
                            line_is_bold = True
                    
                    if line_text.strip():
                        line_info.append({
                            'text': line_text.strip(),
                            'font_sizes': line_font_sizes,
                            'is_bold': line_is_bold,
                            'max_font_size': max(line_font_sizes) if line_font_sizes else 0
                        })
                        page_text += line_text + "\n"
        
        # Detect headings based on font analysis with quality filters
        for i, line_data in enumerate(line_info):
            text = line_data['text']
            max_font_size = line_data['max_font_size']
            is_bold = line_data['is_bold']
            
            # Quality filters to reduce noise
            if not is_quality_heading(text, max_font_size, is_bold):
                continue
            
            # For PDFs, only detect level 1 headings to reduce noise
            level = None
            if max_font_size >= major_heading_threshold and is_bold:
                level = 1  # Only major headings
            # Skip all other levels for PDFs to reduce noise
            
            if level:
                headings.append({
                    'text': text,
                    'line_number': i,
                    'level': level,
                    'full_text': text,
                    'detection_method': 'font',
                    'font_size': max_font_size,
                    'is_bold': is_bold
                })
        
        pages.append({
            "page": page_num + 1,
            "text": page_text.strip(),
            "headings": headings,
            "font_thresholds": {
                "heading": heading_threshold,
                "major_heading": major_heading_threshold
            }
        })
    
    doc.close()
    return pages

def is_quality_heading(text: str, font_size: float, is_bold: bool) -> bool:
    """Check if text qualifies as a quality heading."""
    # Remove extra whitespace and normalize
    text = text.strip()
    
    # Skip if too short or too long - be more restrictive for level 1 headings
    if len(text) < 5 or len(text) > 200:
        return False
    
    # Skip if mostly symbols or numbers - be more restrictive
    symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if symbol_count > len(text) * 0.3:  # More than 30% symbols
        return False
    
    # Skip if mostly numbers - be more restrictive
    digit_count = sum(1 for c in text if c.isdigit())
    if digit_count > len(text) * 0.5:  # More than 50% digits
        return False
    
    # Skip common noise patterns
    noise_patterns = [
        r'^[^\w]*$',  # Only symbols
        r'^\d+$',     # Only numbers
        r'^[A-Z\s]+$',  # ALL CAPS with only spaces (likely formatting)
        r'^[^\w\s]*$',  # Only symbols and spaces
        r'^[A-Z]\s*$',  # Single capital letter
        r'^\s*[^\w\s]+\s*$',  # Only symbols with spaces
        r'^[A-Z][A-Z\s]{0,2}$',  # Very short ALL CAPS
        r'^[^\w]*\d+[^\w]*$',  # Numbers with symbols
        r'^[A-Z]\d+[A-Z]*$',  # Letter-number combinations
        r'^[^\w]*[A-Z][^\w]*$',  # Single letter with symbols
        r'^[^\w]*\d+[^\w]*$',  # Numbers with symbols
        r'^[A-Z][A-Z\s]*\d+[A-Z\s]*$',  # ALL CAPS with numbers
        r'^[^\w\s]*[A-Z][^\w\s]*$',  # Single letter surrounded by symbols
        r'^[A-Z][A-Z\s]*[^\w\s]+[A-Z\s]*$',  # ALL CAPS with symbols
        r'^[^\w\s]*\d+[^\w\s]*$',  # Numbers surrounded by symbols
        r'^[A-Z][A-Z\s]*[^\w\s]+$',  # ALL CAPS ending with symbols
        r'^[^\w\s]+[A-Z][A-Z\s]*$',  # ALL CAPS starting with symbols
        r'^[A-Z][A-Z\s]*[^\w\s]+[A-Z\s]*[^\w\s]*$',  # ALL CAPS with symbols throughout
        r'^[^\w\s]*[A-Z][A-Z\s]*[^\w\s]*$',  # ALL CAPS surrounded by symbols
        r'^[A-Z][A-Z\s]*[^\w\s]+[A-Z\s]*[^\w\s]+$',  # ALL CAPS with symbols at start and end
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, text):
            return False
    
    # Must have at least one letter
    if not any(c.isalpha() for c in text):
        return False
    
    # Skip very short text unless it's bold and looks like a heading
    if len(text) < 5 and not (is_bold and font_size > 12):
        return False
    
    # Additional checks for historical documents
    # Skip text that looks like OCR artifacts or formatting
    if re.search(r'[^\w\s]{3,}', text):  # 3+ consecutive symbols
        return False
    
    # Skip text that's mostly punctuation or special characters
    if len(re.findall(r'[^\w\s]', text)) > len(text) * 0.3:
        return False
    
    # Skip text that looks like coordinates or measurements
    if re.search(r'\d+[^\w\s]*[A-Z]', text) or re.search(r'[A-Z][^\w\s]*\d+', text):
        return False
    
    # Skip obvious table headers, lists, and fragments
    table_patterns = [
        r'^[A-Z][a-z]+,\s+[A-Z][a-z]+$',  # "Name, Location" pattern
        r'^\d+\s+[a-z]+',  # "1 horse" pattern
        r'^[A-Z][a-z]+\s+\d+',  # "Page 5" pattern
        r'^\d+\.\d+$',  # Just numbers like "3.00"
        r'^[A-Z]\s+[A-Z]$',  # Single letters like "A B"
        r'^[a-z]{1,3}$',  # Very short lowercase
        r'^\w+\s*\d+\s*\w*$',  # Pattern like "Item 1 A"
    ]
    
    for pattern in table_patterns:
        if re.match(pattern, text):
            return False
    
    # Must have at least 2 words for a meaningful heading
    if len(text.split()) < 2:
        return False
    
    return True

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with heading detection."""
    return extract_text_with_font_info(file_path)

def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from TXT file with heading detection."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines to simulate pages
    sections = content.split('\n\n')
    pages = []
    
    for i, section in enumerate(sections):
        if section.strip():
            # Detect headings using regex patterns (allow all levels for TXT)
            headings = detect_heading_patterns(section, only_level_1=False)
            
            pages.append({
                "page": i + 1,
                "text": section.strip(),
                "headings": headings
            })
    
    return pages

def merge_heading_detection(pages: List[Dict[str, Any]], is_pdf: bool = True) -> List[Dict[str, Any]]:
    """Merge regex and font-based heading detection, preferring font detection."""
    total_headings = 0
    
    for page in pages:
        if 'headings' not in page:
            # For pages without font analysis, use regex (level 1 only for PDFs)
            page['headings'] = detect_heading_patterns(page['text'], only_level_1=is_pdf)
        else:
            # For PDF pages, combine font and regex detection (level 1 only)
            regex_headings = detect_heading_patterns(page['text'], only_level_1=is_pdf)
            
            # Create a map of line numbers to existing headings
            existing_headings = {h['line_number']: h for h in page['headings']}
            
            # Add regex headings that don't conflict with font headings
            for regex_heading in regex_headings:
                line_num = regex_heading['line_number']
                if line_num not in existing_headings:
                    page['headings'].append(regex_heading)
            
            # Filter to only level 1 headings for PDFs
            if is_pdf:
                page['headings'] = [h for h in page['headings'] if h.get('level', 1) == 1]
            
            # Sort by line number
            page['headings'].sort(key=lambda x: x['line_number'])
        
        total_headings += len(page['headings'])
    
    print(f"Detected {total_headings} total headings across all pages")
    return pages

def chunk_text_with_headings(pages: List[Dict[str, Any]], max_tokens: int = 300) -> List[Dict[str, Any]]:
    """Chunk text while preserving heading hierarchy across pages."""
    chunks = []
    current_headings = []  # Track current heading path across pages
    
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        headings = page_data.get("headings", [])
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        current_chunk = []
        current_tokens = 0
        chunk_headings = current_headings.copy()  # Headings for current chunk
        
        for i, paragraph in enumerate(paragraphs):
            # Check if this paragraph is a heading
            is_heading = any(h['line_number'] == i for h in headings)
            
            if is_heading:
                # Update current heading path
                heading = next(h for h in headings if h['line_number'] == i)
                level = heading['level']
                
                # Trim heading path to current level and add new heading
                current_headings = current_headings[:level-1]
                current_headings.append(heading['text'])
                chunk_headings = current_headings.copy()
                
                # Debug output for heading detection
                if len(current_headings) <= 3:  # Only show first few levels to avoid spam
                    print(f"  Page {page_num}: Heading level {level}: {heading['text']}")
            
            para_tokens = len(paragraph.split()) * 1.3  # Rough token estimation
            
            # If adding this paragraph would exceed max_tokens, save current chunk
            if current_tokens + para_tokens > max_tokens and current_chunk:
                # Create heading prefix
                heading_prefix = ""
                if chunk_headings:
                    heading_prefix = " | ".join(chunk_headings) + " | "
                
                # Save current chunk with heading prefix
                chunk_text = ' '.join(current_chunk)
                full_text = heading_prefix + chunk_text
                
                chunks.append({
                    "page": page_num,
                    "text": full_text,
                    "original_text": chunk_text,  # Keep original text for reference
                    "char_start": 0,
                    "char_end": len(chunk_text),
                    "headings_path": chunk_headings.copy()
                })
                
                # Start new chunk with minimal overlap (just the last paragraph)
                if len(current_chunk) > 0:
                    current_chunk = [current_chunk[-1]]
                    current_tokens = len(current_chunk[0].split()) * 1.3
                else:
                    current_chunk = []
                    current_tokens = 0
                
                # Update chunk headings for next chunk
                chunk_headings = current_headings.copy()
            
            current_chunk.append(paragraph)
            current_tokens += para_tokens
        
        # Add final chunk for this page
        if current_chunk:
            # Create heading prefix
            heading_prefix = ""
            if chunk_headings:
                heading_prefix = " | ".join(chunk_headings) + " | "
            
            chunk_text = ' '.join(current_chunk)
            full_text = heading_prefix + chunk_text
            
            chunks.append({
                "page": page_num,
                "text": full_text,
                "original_text": chunk_text,
                "char_start": 0,
                "char_end": len(chunk_text),
                "headings_path": chunk_headings.copy()
            })
    
    # Print summary of chunks with headings
    chunks_with_headings = sum(1 for chunk in chunks if chunk['headings_path'])
    print(f"Created {len(chunks)} chunks, {chunks_with_headings} with headings")
    
    return chunks

def extract_entities_and_years(text: str) -> Tuple[List[Dict[str, str]], List[int]]:
    """Extract entities and years from text."""
    entities = []
    years = []
    
    # Extract years using regex
    year_pattern = r'\b(1[4-9]\d{2}|20\d{2})\b'
    years = [int(year) for year in re.findall(year_pattern, text) if year.strip()]
    
    # Also extract decade references like "1940s", "1950s", etc.
    decade_pattern = r'\b(1[4-9]\d{2}s|20\d{2}s)\b'
    decade_matches = re.findall(decade_pattern, text)
    for decade in decade_matches:
        # Convert "1940s" to 1940, "1950s" to 1950, etc.
        base_year = int(decade[:-1])  # Remove 's' and convert to int
        years.append(base_year)
    
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
            
            # Extract and store entities and years from original text (not prefixed)
            text_for_entities = chunk.get("original_text", chunk["text"])
            entities, years = extract_entities_and_years(text_for_entities)
            
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
            # Extract years and entities from original text (not prefixed)
            text_for_entities = passage.get("original_text", passage["text"])
            _, years = extract_entities_and_years(text_for_entities)
            entities, _ = extract_entities_and_years(text_for_entities)
            
            person_entities = [e["entity"] for e in entities if e["ent_type"] == "PERSON"]
            place_entities = [e["entity"] for e in entities if e["ent_type"] in ["GPE", "FAC"]]
            
            documents.append({
                "id": passage_id,
                "text": passage["text"],  # Use prefixed text for search
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

def debug_headings(pages: List[Dict[str, Any]], max_pages: int = 3):
    """Debug function to show heading detection results."""
    print("\n=== HEADING DETECTION DEBUG ===")
    for i, page in enumerate(pages[:max_pages]):
        print(f"\nPage {page['page']}:")
        if page.get('headings'):
            for heading in page['headings']:
                method = heading.get('detection_method', 'unknown')
                level = heading.get('level', '?')
                text = heading.get('text', '')
                print(f"  Level {level} ({method}): {text}")
        else:
            print("  No headings detected")
    print("=== END DEBUG ===\n")

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--title', help='Document title')
@click.option('--authors', help='Comma-separated list of authors')
@click.option('--pub-year', type=int, help='Publication year')
@click.option('--debug', is_flag=True, help='Show debug information for heading detection')
def main(file_path: str, title: str, authors: str, pub_year: int, debug: bool):
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
    
    # Merge heading detection across pages
    print("Merging heading detection...")
    is_pdf = file_path.lower().endswith('.pdf')
    pages = merge_heading_detection(pages, is_pdf=is_pdf)
    
    # Debug output if requested
    if debug:
        debug_headings(pages)
    
    # Chunk the text
    print("Chunking text...")
    chunks = chunk_text_with_headings(pages)
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
    
    author_list = [a.strip() for a in authors.split(',') if a.strip()] if authors else None
    
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









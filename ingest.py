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
from psycopg2.extras import RealDictCursor
import spacy
import re
from typing import List, Dict, Any, Tuple, Optional
import json
from collections import Counter

import tiktoken

from config import get_db_connection, get_meili_client

_tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base tokenizer."""
    return len(_tokenizer.encode(text))

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

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
        (r'^APPENDIX\s+[A-Z][-\d]*[:\s]*(.+)$', 1),  # APPENDIX A: Title, APPENDIX E-4: Title
        (r'^Appendix\s+[A-Z][-\d]*[:\s]*(.+)$', 1),  # Appendix A: Title, Appendix E-4: Title
        (r'.*APPENDIX\s+[A-Z][-\d]*[:\s]*([^|]+)', 1),  # APPENDIX E-4: Title (anywhere in line)
        (r'.*APPENDIX\s+[A-Z][-\d]*\s+([^|]+?)(?:\s+Damage|\s+Claimant|$)', 1),  # APPENDIX E-4 MORGAN'S RAID CLAIMS
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
    """Extract text with font information and advanced heading detection."""
    pages = []
    doc = fitz.open(file_path)
    
    # Use advanced heading detection methods
    all_headings = merge_heading_detection_methods(file_path)
    
    # Group headings by page
    headings_by_page = {}
    for heading in all_headings:
        page_num = heading['page']
        if page_num not in headings_by_page:
            headings_by_page[page_num] = []
        headings_by_page[page_num].append(heading)
    
    # Analyze font sizes across all pages to determine heading thresholds (for backward compatibility)
    all_font_sizes = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_font_sizes.append(span["size"])
    
    # Calculate font size thresholds
    if all_font_sizes:
        font_sizes = sorted(all_font_sizes)
        heading_threshold = font_sizes[int(len(font_sizes) * 0.75)]
        major_heading_threshold = font_sizes[int(len(font_sizes) * 0.85)]
    else:
        heading_threshold = 12
        major_heading_threshold = 14
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        page_text = ""
        headings = headings_by_page.get(page_num, [])
        
        # Convert advanced headings to the expected format
        formatted_headings = []
        for heading in headings:
            title = heading.get('title', heading.get('text', ''))
            formatted_headings.append({
                'text': title,
                'line_number': 0,  # Will be updated below
                'level': heading['level'],
                'full_text': title,
                'detection_method': heading['detection_method'],
                'confidence': heading.get('confidence', 0.5)
            })
        
        # Extract text and find line numbers for headings
        line_info = []
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
        
        # Match headings to line numbers
        for heading in formatted_headings:
            for i, line_data in enumerate(line_info):
                if (heading['text'].lower() in line_data['text'].lower() or 
                    line_data['text'].lower() in heading['text'].lower()):
                    heading['line_number'] = i
                    break
        
        pages.append({
            "page": page_num + 1,
            "text": page_text.strip(),
            "headings": formatted_headings,
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
    
    # Special case: Allow appendix sections even if they don't meet normal criteria
    if 'APPENDIX' in text.upper() and 'MORGAN' in text.upper() and 'CLAIMS' in text.upper():
        return True
    
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
            # For pages without font analysis, use regex (level 1 and 2 for PDFs)
            page['headings'] = detect_heading_patterns(page['text'], only_level_1=False)
        else:
            # For PDF pages, combine font and regex detection (level 1 and 2)
            regex_headings = detect_heading_patterns(page['text'], only_level_1=False)
            
            # Create a map of line numbers to existing headings
            existing_headings = {h['line_number']: h for h in page['headings']}
            
            # Add regex headings that don't conflict with font headings
            for regex_heading in regex_headings:
                line_num = regex_heading['line_number']
                if line_num not in existing_headings:
                    page['headings'].append(regex_heading)
            
            # Filter to level 1 and 2 headings for PDFs (allow appendix sections)
            if is_pdf:
                page['headings'] = [h for h in page['headings'] if h.get('level', 1) in [1, 2]]
            
            # Sort by line number
            page['headings'].sort(key=lambda x: x['line_number'])
        
        total_headings += len(page['headings'])
    
    print(f"Detected {total_headings} total headings across all pages")
    return pages

def chunk_text_with_headings(pages: List[Dict[str, Any]], max_tokens: int = 300, document_title: str = None) -> List[Dict[str, Any]]:
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
                heading_text = heading.get('text', heading.get('title', ''))
                current_headings.append(heading_text)
                chunk_headings = current_headings.copy()
                
                # Debug output for heading detection
                if len(current_headings) <= 3:  # Only show first few levels to avoid spam
                    print(f"  Page {page_num}: Heading level {level}: {heading_text}")
            
            para_tokens = count_tokens(paragraph)
            
            # If adding this paragraph would exceed max_tokens, save current chunk
            if current_tokens + para_tokens > max_tokens and current_chunk:
                # Create document and heading prefix
                prefix_parts = []
                if document_title:
                    prefix_parts.append(document_title)
                if chunk_headings:
                    prefix_parts.extend(chunk_headings)
                
                prefix = " | ".join(prefix_parts) + " | " if prefix_parts else ""
                
                # Save current chunk with document and heading prefix
                chunk_text = ' '.join(current_chunk)
                full_text = prefix + chunk_text
                
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
                    current_tokens = count_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_tokens = 0
                
                # Update chunk headings for next chunk
                chunk_headings = current_headings.copy()
            
            current_chunk.append(paragraph)
            current_tokens += para_tokens
        
        # Add final chunk for this page
        if current_chunk:
            # Create document and heading prefix
            prefix_parts = []
            if document_title:
                prefix_parts.append(document_title)
            if chunk_headings:
                prefix_parts.extend(chunk_headings)
            
            prefix = " | ".join(prefix_parts) + " | " if prefix_parts else ""
            
            chunk_text = ' '.join(current_chunk)
            full_text = prefix + chunk_text
            
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

def index_in_meilisearch(passages: List[Dict[str, Any]], passage_ids: List[str], doc_id: str):
    """Index passages in Meilisearch."""
    try:
        client = get_meili_client()

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
                "document_id": doc_id,  # Add document_id for filtering
                "text": passage["text"],  # Use prefixed text for search
                "page": passage["page"],
                "headings_path": passage["headings_path"],
                "years": years,
                "entities_person": person_entities,
                "entities_place": place_entities
            })
        
        # Create or update index
        index = client.index("passages")

        # Check if index exists; if not, configure embedder and filterable attributes
        try:
            settings = index.get_settings()
            needs_setup = not settings.get("embedders")
        except Exception:
            needs_setup = True

        if needs_setup:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                task = index.update_embedders({
                    "default": {
                        "source": "openAi",
                        "apiKey": openai_key,
                        "model": "text-embedding-3-small",
                        "documentTemplate": "{{doc.text}}"
                    }
                })
                client.wait_for_task(task.task_uid, timeout_in_ms=300000)

            task = index.update_filterable_attributes(["document_id"])
            client.wait_for_task(task.task_uid, timeout_in_ms=60000)

        index.add_documents(documents, primary_key="id")

        print(f"Indexed {len(documents)} passages in Meilisearch for document {doc_id}")
        
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
                text = heading.get('text', heading.get('title', ''))
                print(f"  Level {level} ({method}): {text}")
        else:
            print("  No headings detected")
    print("=== END DEBUG ===\n")

def extract_headings_from_outline(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract headings from PDF's built-in outline/bookmarks (fastest method).
    
    Returns:
        List of heading dictionaries with level, title, and page info
    """
    try:
        doc = fitz.open(file_path)
        toc = doc.get_toc(simple=True)  # Returns [[level, title, page], ...]
        
        headings = []
        for level, title, page in toc:
            title = title.strip()
            if not title:  # Skip empty titles
                continue
                
            # Filter out page number headings like "Page 1", "Page 2", etc.
            if re.match(r'^Page\s+\d+$', title, re.IGNORECASE):
                continue
                
            headings.append({
                "level": level,
                "title": title,
                "page": page - 1,  # Convert to 0-based indexing
                "detection_method": "outline"
            })
        
        doc.close()
        return headings
    except Exception as e:
        print(f"Warning: Could not extract outline from {file_path}: {e}")
        return []

def extract_headings_from_toc_pages(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse TABLE OF CONTENTS pages to extract headings.
    
    This method looks for TOC pages and parses the format where titles and page numbers
    are on separate lines.
    """
    try:
        doc = fitz.open(file_path)
        toc_pages = []
        
        # Find pages containing "TABLE OF CONTENTS"
        TOC_TITLE = "TABLE OF CONTENTS"
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text()
            if TOC_TITLE in text.upper():
                toc_pages.append(i)
                # Also check next page if it looks TOC-ish (lots of lines ending in digits)
                if i + 1 < len(doc):
                    next_page = doc.load_page(i + 1)
                    next_text = next_page.get_text()
                    lines_with_digits = sum(1 for line in next_text.splitlines() 
                                          if re.search(r'\d{1,4}\s*$', line.strip()))
                    if lines_with_digits > 5:  # If more than 5 lines end with digits
                        toc_pages.append(i + 1)
        
        items = []
        
        for page_num in toc_pages:
            page = doc.load_page(page_num)
            text = page.get_text()
            lines = text.splitlines()
            
            # Parse TOC format: title on one line, page number on next line
            i = 0
            while i < len(lines) - 1:
                title_line = lines[i].strip()
                page_line = lines[i + 1].strip()
                
                # Skip if title line is empty or contains TOC header
                if not title_line or 'TABLE OF CONTENTS' in title_line.upper():
                    i += 1
                    continue
                
                # Check if next line is a page number
                page_match = re.match(r'^(\d{1,4})\s*$', page_line)
                if page_match:
                    page_num = int(page_match.group(1))
                    title = title_line.strip()
                    
                    # Clean up title
                    title = re.sub(r'\s{2,}', ' ', title)  # Normalize whitespace
                    title = title.strip('–-·•\t ')  # Remove common TOC artifacts
                    
                    # Skip very short titles or page numbers
                    if len(title) > 2 and not re.match(r'^\d+$', title):
                        items.append({
                            "level": 1,
                            "title": title,
                            "page": page_num - 1,  # Convert to 0-based indexing
                            "detection_method": "toc_parsing"
                        })
                    
                    i += 2  # Skip both title and page number lines
                else:
                    i += 1  # Move to next line
        
        doc.close()
        return items
    except Exception as e:
        print(f"Warning: Could not parse TOC from {file_path}: {e}")
        return []

def extract_headings_by_typography(file_path: str) -> List[Dict[str, Any]]:
    """
    Enhanced typography/layout detection for headings.
    
    Uses font size analysis, uppercase ratio, and positioning to detect headings.
    """
    try:
        doc = fitz.open(file_path)
        spans = []
        
        # Collect all text spans with their metadata
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text:
                                continue
                            
                            spans.append({
                                'page': page_num,
                                'text': text,
                                'size': span["size"],
                                'flags': span.get("flags", 0),
                                'origin': span.get("origin", (0, 0)),
                                'bbox': span.get("bbox", (0, 0, 0, 0))
                            })
        
        # Determine font size thresholds
        sizes = [round(span['size'], 1) for span in spans]
        size_counts = Counter(sizes)
        top_sizes = [size for size, _ in size_counts.most_common(5)]
        big_sizes = set(top_sizes[:2])  # Top 1-2 font sizes
        
        headings = []
        for span in spans:
            text = span['text']
            size = round(span['size'], 1)
            
            # Basic size filter
            if size not in big_sizes:
                continue
                
            # Length filter
            if not (4 <= len(text) <= 80):
                continue
            
            # Uppercase ratio analysis
            alpha_chars = [c for c in text if c.isalpha()]
            if not alpha_chars:
                continue
                
            uppercase_count = sum(1 for c in alpha_chars if c.isupper())
            uppercase_ratio = uppercase_count / len(alpha_chars)
            
            # Heading heuristics
            is_heading = False
            
            # High uppercase ratio (>60%) or title case
            if uppercase_ratio > 0.6 or text.istitle():
                # Avoid lines ending with punctuation (likely body text)
                if not text.endswith((".", ":", ";", ",")):
                    is_heading = True
            
            # Check for bold formatting
            is_bold = bool(span['flags'] & 2**4)
            if is_bold and len(text) > 3:
                is_heading = True
            
            # Check for common heading patterns
            heading_patterns = [
                r'^CHAPTER\s+\d+',
                r'^Chapter\s+\d+',
                r'^PART\s+\d+',
                r'^APPENDIX\s+[A-Z]',
                r'^Appendix\s+[A-Z]',
                r'^[A-Z][A-Z\s]{8,}$',  # Long ALL CAPS
            ]
            
            for pattern in heading_patterns:
                if re.match(pattern, text):
                    is_heading = True
                    break
            
            if is_heading:
                # For PDFs, only use level 1 headings to reduce noise
                level = 1
                
                headings.append({
                    "level": level,
                    "title": " ".join(text.split()),  # Normalize whitespace
                    "page": span['page'],
                    "detection_method": "typography",
                    "font_size": size,
                    "uppercase_ratio": uppercase_ratio,
                    "is_bold": is_bold
                })
        
        # De-duplicate consecutive duplicates
        deduped = []
        seen = set()
        for heading in headings:
            key = (heading['page'], heading['title'].lower())
            if key not in seen:
                seen.add(key)
                deduped.append(heading)
        
        doc.close()
        return deduped
    except Exception as e:
        print(f"Warning: Could not extract headings by typography from {file_path}: {e}")
        return []

def filter_appendices(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter and enhance appendix headings specifically.
    
    Historical PDFs often have clear appendix patterns that should be preserved.
    """
    appendix_patterns = [
        r'^APPENDIX\s+[A-Z][-\d]*',
        r'^Appendix\s+[A-Z][-\d]*',
        r'.*APPENDIX\s+[A-Z][-\d]*\s+([^|]+?)(?:\s+Damage|\s+Claimant|$)',
    ]
    
    appendix_headings = []
    for heading in headings:
        # Handle different heading formats
        title = heading.get('title', heading.get('text', ''))
        if not title:
            continue
            
        for pattern in appendix_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                appendix_headings.append(heading)
                break
    
    return appendix_headings

def merge_heading_detection_methods(file_path: str) -> List[Dict[str, Any]]:
    """
    Combine all heading detection methods and merge results intelligently.
    
    Returns a unified list of headings with confidence scores.
    """
    all_headings = []
    
    # Method 1: PDF Outline (fastest, most reliable)
    outline_headings = extract_headings_from_outline(file_path)
    for heading in outline_headings:
        heading['confidence'] = 0.9  # High confidence for outline
        all_headings.append(heading)
    
    # Method 2: TOC Parsing
    toc_headings = extract_headings_from_toc_pages(file_path)
    for heading in toc_headings:
        heading['confidence'] = 0.8  # High confidence for TOC
        all_headings.append(heading)
    
    # Method 3: Typography Detection
    typography_headings = extract_headings_by_typography(file_path)
    for heading in typography_headings:
        heading['confidence'] = 0.6  # Medium confidence for typography
        all_headings.append(heading)
    
    # Method 4: Regex Patterns (existing method)
    regex_headings = detect_heading_patterns_from_file(file_path)
    for heading in regex_headings:
        heading['confidence'] = 0.5  # Lower confidence for regex
        all_headings.append(heading)
    
    # Merge and deduplicate
    merged = merge_heading_results(all_headings)
    
    # Filter to only level 1 headings for PDFs to reduce noise
    merged = [h for h in merged if h.get('level', 1) == 1]
    
    # Add appendix-specific filtering
    appendix_headings = filter_appendices(merged)
    if appendix_headings:
        # Ensure appendices are included even if they have lower confidence
        for appendix in appendix_headings:
            appendix_title = appendix.get('title', appendix.get('text', ''))
            appendix_page = appendix.get('page', 0)
            
            if not any(h.get('title', h.get('text', '')).lower() == appendix_title.lower() 
                      and h.get('page', 0) == appendix_page for h in merged):
                appendix['confidence'] = max(appendix.get('confidence', 0), 0.7)
                merged.append(appendix)
    
    return merged

def detect_heading_patterns_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract headings using regex patterns from a file.
    """
    try:
        doc = fitz.open(file_path)
        headings = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Use existing regex detection
            page_headings = detect_heading_patterns(text, only_level_1=False)
            
            for heading in page_headings:
                heading['page'] = page_num
                heading['detection_method'] = 'regex'
                headings.append(heading)
        
        doc.close()
        return headings
    except Exception as e:
        print(f"Warning: Could not extract headings by regex from {file_path}: {e}")
        return []

def merge_heading_results(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge heading results from multiple detection methods.
    
    Handles duplicates by keeping the highest confidence detection.
    """
    # Group by page and title (case-insensitive)
    grouped = {}
    
    for heading in headings:
        # Handle different heading formats
        title = heading.get('title', heading.get('text', ''))
        page = heading.get('page', 0)
        
        if not title:  # Skip headings without title
            continue
            
        key = (page, title.lower())
        confidence = heading.get('confidence', 0)
        
        if key not in grouped or confidence > grouped[key]['confidence']:
            grouped[key] = heading
    
    # Convert back to list and sort by page, then by line number
    merged = list(grouped.values())
    merged.sort(key=lambda x: (x.get('page', 0), x.get('line_number', 0)))
    
    return merged

def rank_headings_by_relevance(query: str, headings: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Rank headings by relevance to a user query using fuzzy matching.
    
    Args:
        query: User's search query
        headings: List of heading dictionaries
        limit: Maximum number of results to return
    
    Returns:
        List of headings ranked by relevance score
    """
    try:
        # Try to import rapidfuzz for better fuzzy matching
        from rapidfuzz import process, fuzz
        use_rapidfuzz = True
    except ImportError:
        # Fallback to simple string matching
        use_rapidfuzz = False
        print("Warning: rapidfuzz not available, using simple string matching")
    
    if use_rapidfuzz:
        # Use rapidfuzz for sophisticated fuzzy matching
        choices = [h.get('title', h.get('text', '')) for h in headings]
        ranked = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=limit)
        
        # Attach scores to headings
        out = []
        for title, score, idx in ranked:
            h = headings[idx].copy()
            h['relevance_score'] = score / 100.0  # Normalize to 0-1
            out.append(h)
        
        return out
    else:
        # Simple fallback: exact and partial string matching
        query_lower = query.lower()
        scored_headings = []
        
        for heading in headings:
            title_lower = heading.get('title', heading.get('text', '')).lower()
            score = 0.0
            
            # Exact match
            if query_lower == title_lower:
                score = 1.0
            # Query is substring of title
            elif query_lower in title_lower:
                score = 0.8
            # Title is substring of query
            elif title_lower in query_lower:
                score = 0.7
            # Word overlap
            else:
                query_words = set(query_lower.split())
                title_words = set(title_lower.split())
                if query_words & title_words:  # Intersection
                    overlap = len(query_words & title_words)
                    score = overlap / max(len(query_words), len(title_words))
            
            if score > 0:
                h = heading.copy()
                h['relevance_score'] = score
                scored_headings.append(h)
        
        # Sort by score and return top results
        scored_headings.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_headings[:limit]

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
    chunks = chunk_text_with_headings(pages, document_title=title)
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
    index_in_meilisearch(chunks, passage_ids, doc_id)
    
    conn.close()
    print("Ingestion complete!")

if __name__ == "__main__":
    main()









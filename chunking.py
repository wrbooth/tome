"""
Text chunking for the Codex ingestion pipeline.

Splits extracted text into token-limited chunks while preserving
heading hierarchy across pages.
"""

import logging
import tiktoken
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base tokenizer."""
    return len(_tokenizer.encode(text))


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
                    logger.debug("  Page %d: Heading level %d: %s", page_num, level, heading_text)

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
                "headings_path": chunk_headings.copy()
            })

    # Log summary of chunks with headings
    chunks_with_headings = sum(1 for chunk in chunks if chunk['headings_path'])
    logger.info("Created %d chunks, %d with headings", len(chunks), chunks_with_headings)

    return chunks

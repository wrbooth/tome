#!/usr/bin/env python3
"""
Test script for heading detection functionality.
"""

import os
import sys
from ingest import (
    detect_heading_patterns, 
    extract_text_with_font_info, 
    extract_text_from_txt,
    merge_heading_detection,
    chunk_text_with_headings,
    debug_headings
)

def test_regex_detection():
    """Test regex-based heading detection."""
    print("=== Testing Regex Heading Detection ===")
    
    sample_text = """
CHAPTER 1: Introduction
This is the first chapter.

Section 1.1: Background
Here is some background information.

1.2 Methodology
This section describes the methodology.

PART 2: Analysis
This is part two.

Book 3: Conclusions
Final thoughts.
"""
    
    headings = detect_heading_patterns(sample_text)
    print(f"Detected {len(headings)} headings:")
    for heading in headings:
        print(f"  Level {heading['level']}: {heading['text']}")

def test_pdf_extraction(file_path):
    """Test PDF extraction with font analysis."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"=== Testing PDF Extraction: {file_path} ===")
    
    try:
        pages = extract_text_with_font_info(file_path)
        print(f"Extracted {len(pages)} pages")
        
        # Show font thresholds for first page
        if pages:
            thresholds = pages[0].get('font_thresholds', {})
            print(f"Font thresholds: {thresholds}")
            
            # Show headings from first few pages
            for i, page in enumerate(pages[:3]):
                print(f"\nPage {page['page']}:")
                headings = page.get('headings', [])
                print(f"  {len(headings)} headings detected")
                for heading in headings[:5]:  # Show first 5 headings
                    method = heading.get('detection_method', 'unknown')
                    level = heading.get('level', '?')
                    text = heading.get('text', '')
                    print(f"    Level {level} ({method}): {text}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")

def test_txt_extraction(file_path):
    """Test TXT extraction with regex detection."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"=== Testing TXT Extraction: {file_path} ===")
    
    try:
        pages = extract_text_from_txt(file_path)
        print(f"Extracted {len(pages)} pages")
        
        # Show headings from first few pages
        for i, page in enumerate(pages[:3]):
            print(f"\nPage {page['page']}:")
            headings = page.get('headings', [])
            print(f"  {len(headings)} headings detected")
            for heading in headings[:5]:  # Show first 5 headings
                method = heading.get('detection_method', 'unknown')
                level = heading.get('level', '?')
                text = heading.get('text', '')
                print(f"    Level {level} ({method}): {text}")
        
    except Exception as e:
        print(f"Error processing TXT: {e}")

def test_chunking(file_path):
    """Test the full chunking process."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"=== Testing Full Chunking Process: {file_path} ===")
    
    try:
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            pages = extract_text_with_font_info(file_path)
        else:
            pages = extract_text_from_txt(file_path)
        
        # Merge heading detection
        is_pdf = file_path.lower().endswith('.pdf')
        pages = merge_heading_detection(pages, is_pdf=is_pdf)
        
        # Show debug info
        debug_headings(pages, max_pages=2)
        
        # Chunk the text
        chunks = chunk_text_with_headings(pages, max_tokens=300)  # Smaller chunks for testing
        
        print(f"\nCreated {len(chunks)} chunks")
        
        # Show first few chunks with their headings
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  Page: {chunk['page']}")
            print(f"  Headings: {chunk['headings_path']}")
            print(f"  Text preview: {chunk['text'][:100]}...")
        
    except Exception as e:
        print(f"Error in chunking process: {e}")

def main():
    """Main test function."""
    print("Heading Detection Test Script")
    print("=" * 40)
    
    # Test regex detection
    test_regex_detection()
    
    # Test with provided files
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        if file_path.lower().endswith('.pdf'):
            test_pdf_extraction(file_path)
            test_chunking(file_path)
        elif file_path.lower().endswith('.txt'):
            test_txt_extraction(file_path)
            test_chunking(file_path)
        else:
            print("Please provide a PDF or TXT file")
    else:
        print("\nUsage: python test_heading_detection.py <file_path>")
        print("Example: python test_heading_detection.py sample.pdf")

if __name__ == "__main__":
    main()

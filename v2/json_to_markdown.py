#!/usr/bin/env python3
"""
Script to convert JSON files from OCR processing to a single markdown file.
Iterates through JSON files sequentially and extracts text content.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any


def extract_page_number(filename: str) -> int:
    """Extract page number from filename for sorting."""
    match = re.search(r'page_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def get_json_files(data_dir: str) -> List[str]:
    """Get all JSON files sorted by page number."""
    json_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # Sort by page number for sequential processing
    json_files.sort(key=lambda x: extract_page_number(os.path.basename(x)))
    
    return json_files


def process_json_file(file_path: str) -> str:
    """Process a single JSON file and return markdown content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return f"<!-- Error: Invalid JSON structure in {os.path.basename(file_path)} -->\n\n"
        
        markdown_content = []
        filename = os.path.basename(file_path)
        
        # Add page number as hidden comment
        page_num = extract_page_number(filename)
        markdown_content.append(f"<!-- Page {page_num} -->\n")
        
        for item in data:
            if isinstance(item, dict):
                category = item.get('category', 'Unknown')
                text = item.get('text', '')
                
                if text and text.strip():
                    # Handle different categories
                    if category == 'Title':
                        # Remove newlines from titles and clean up
                        cleaned_title = text.strip().replace('\n', ' ').replace('  ', ' ')
                        # Remove any # patterns at the beginning
                        cleaned_title = re.sub(r'^#+\s*', '', cleaned_title)
                        markdown_content.append(f"## {cleaned_title}\n\n")
                    elif category == 'Heading':
                        # Remove newlines from headings and clean up
                        cleaned_heading = text.strip().replace('\n', ' ').replace('  ', ' ')
                        markdown_content.append(f"#### {cleaned_heading}\n\n")
                    elif category == 'Text':
                        # Clean up text and preserve line breaks, but clean up excessive whitespace
                        cleaned_text = text.strip().replace('\n', '  \n').replace('  ', ' ')
                        markdown_content.append(f"{cleaned_text}\n\n")
                    elif category == 'Page-footer':
                        # Skip page numbers
                        continue
                    elif category == 'Picture':
                        markdown_content.append("*[Image content]*\n\n")
                    elif category == 'Section-header':
                        # Treat section-headers as level 3 headings and strip any # patterns
                        cleaned_heading = text.strip().replace('\n', ' ').replace('  ', ' ')
                        # Remove any # patterns at the beginning
                        cleaned_heading = re.sub(r'^#+\s*', '', cleaned_heading)
                        markdown_content.append(f"### {cleaned_heading}\n\n")
                    else:
                        # For other categories, just include the text
                        cleaned_text = text.strip().replace('\n', '  \n').replace('  ', ' ')
                        markdown_content.append(f"**{category}:** {cleaned_text}\n\n")
        
        return ''.join(markdown_content)
        
    except Exception as e:
        return f"<!-- Error processing {os.path.basename(file_path)}: {str(e)} -->\n\n"


def main():
    """Main function to process all JSON files and generate markdown."""
    import sys
    
    # Check if book title is provided as argument
    if len(sys.argv) < 2:
        print("Usage: python json_to_markdown.py \"Book Title\"")
        print("Example: python json_to_markdown.py \"A Brief History of Guernsey County, Ohio\"")
        return
    
    book_title = sys.argv[1]
    
    # Get the script directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    # Get all JSON files
    json_files = get_json_files(str(data_dir))
    
    if not json_files:
        print("No JSON files found in the data directory.")
        return
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Process each file and build markdown content
    markdown_content = []
    
    # Add document header
    markdown_content.append(f"# {book_title}\n\n")
    
    # Process each JSON file
    for i, json_file in enumerate(json_files, 1):
        print(f"Processing {i}/{len(json_files)}: {os.path.basename(json_file)}")
        file_content = process_json_file(json_file)
        markdown_content.append(file_content)
    
    # Write the combined markdown file
    output_file = script_dir / "combined_document.md"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(markdown_content)
        
        print(f"\nSuccessfully generated markdown file: {output_file}")
        print(f"Total files processed: {len(json_files)}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    main()

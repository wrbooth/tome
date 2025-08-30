#!/usr/bin/env python3
"""
Codex Batch Document Ingestion Script

Handles ingestion of multiple documents with support for:
- Directory processing (recursive)
- Metadata files (CSV/JSON)
- Parallel processing
- Progress tracking
"""

import os
import sys
import click
import json
import csv
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from dotenv import load_dotenv

load_dotenv()

def load_metadata_file(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    """Load document metadata from CSV or JSON file."""
    metadata = {}
    
    if metadata_path.lower().endswith('.json'):
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            for doc in data:
                if 'filename' in doc:
                    metadata[doc['filename']] = doc
    elif metadata_path.lower().endswith('.csv'):
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'filename' in row:
                    metadata[row['filename']] = row
    else:
        raise ValueError("Metadata file must be CSV or JSON")
    
    return metadata

def get_document_files(input_path: str, recursive: bool = False) -> List[str]:
    """Get list of document files to process."""
    path = Path(input_path)
    files = []
    
    if path.is_file():
        # Single file
        if path.suffix.lower() in ['.pdf', '.txt']:
            files.append(str(path))
    elif path.is_dir():
        # Directory
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt']:
                files.append(str(file_path))
    
    return sorted(files)

def extract_metadata_from_filename(file_path: str) -> Dict[str, Any]:
    """Extract basic metadata from filename."""
    filename = Path(file_path).stem
    return {
        'title': filename,
        'authors': None,
        'pub_year': None,
        'language': 'en'
    }

def ingest_single_document(file_path: str, metadata: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """Ingest a single document using the existing ingest.py script."""
    try:
        # Build command
        cmd = [
            sys.executable, "ingest.py",
            file_path,
            "--title", metadata.get('title', Path(file_path).stem),
        ]
        
        if metadata.get('authors'):
            # Convert list to comma-separated string if needed
            authors = metadata['authors']
            if isinstance(authors, list):
                authors = ','.join(authors)
            cmd.extend(["--authors", authors])
        
        if metadata.get('pub_year'):
            cmd.extend(["--pub-year", str(metadata['pub_year'])])
        
        if debug:
            cmd.append("--debug")
        
        # Run ingestion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return {
                'file_path': file_path,
                'status': 'success',
                'output': result.stdout
            }
        else:
            return {
                'file_path': file_path,
                'status': 'error',
                'error': result.stderr
            }
            
    except Exception as e:
        return {
            'file_path': file_path,
            'status': 'error',
            'error': str(e)
        }

def process_documents_parallel(files: List[str], metadata_dict: Dict[str, Dict[str, Any]], 
                             max_workers: int, debug: bool) -> List[Dict[str, Any]]:
    """Process documents in parallel."""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {}
        for file_path in files:
            filename = Path(file_path).name
            metadata = metadata_dict.get(filename, extract_metadata_from_filename(file_path))
            
            future = executor.submit(ingest_single_document, file_path, metadata, debug)
            future_to_file[future] = file_path
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"✅ Success: {Path(file_path).name}")
                else:
                    print(f"❌ Error: {Path(file_path).name} - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                results.append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': str(e)
                })
                print(f"❌ Exception: {Path(file_path).name} - {str(e)}")
    
    return results

def process_documents_sequential(files: List[str], metadata_dict: Dict[str, Dict[str, Any]], 
                               debug: bool) -> List[Dict[str, Any]]:
    """Process documents sequentially."""
    results = []
    
    for i, file_path in enumerate(files, 1):
        filename = Path(file_path).name
        metadata = metadata_dict.get(filename, extract_metadata_from_filename(file_path))
        
        print(f"Processing {i}/{len(files)}: {filename}")
        
        result = ingest_single_document(file_path, metadata, debug)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ Success: {filename}")
        else:
            print(f"❌ Error: {filename} - {result.get('error', 'Unknown error')}")
    
    return results

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--recursive', is_flag=True, help='Process directories recursively')
@click.option('--metadata', type=click.Path(), help='CSV/JSON file with document metadata')
@click.option('--parallel', default=1, help='Number of parallel processes (1 = sequential)')
@click.option('--debug', is_flag=True, help='Show debug information')
@click.option('--output', type=click.Path(), help='Save results to JSON file')
def main(input_path: str, recursive: bool, metadata: Optional[str], 
         parallel: int, debug: bool, output: Optional[str]):
    """Batch ingest multiple documents."""
    
    print(f"=== Codex Batch Ingestion ===")
    print(f"Input: {input_path}")
    print(f"Recursive: {recursive}")
    print(f"Parallel: {parallel}")
    print(f"Debug: {debug}")
    
    # Load metadata if provided
    metadata_dict = {}
    if metadata:
        print(f"Loading metadata from: {metadata}")
        metadata_dict = load_metadata_file(metadata)
        print(f"Loaded metadata for {len(metadata_dict)} documents")
    
    # Get document files
    files = get_document_files(input_path, recursive)
    if not files:
        print("No document files found!")
        return
    
    print(f"Found {len(files)} documents to process:")
    for file_path in files:
        print(f"  - {Path(file_path).name}")
    
    # Process documents
    print(f"\nStarting ingestion...")
    if parallel > 1:
        results = process_documents_parallel(files, metadata_dict, parallel, debug)
    else:
        results = process_documents_sequential(files, metadata_dict, debug)
    
    # Summary
    print(f"\n=== Ingestion Summary ===")
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed documents:")
        for result in results:
            if result['status'] == 'error':
                print(f"  - {Path(result['file_path']).name}: {result.get('error', 'Unknown error')}")
    
    # Save results if requested
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()





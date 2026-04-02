#!/usr/bin/env python3
"""
Script to clear database and MeiliSearch, then re-ingest documents with level 1 headings only.
"""

import os
import sys
import subprocess

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

def clear_meilisearch():
    """Clear all data from MeiliSearch."""
    print("Clearing MeiliSearch...")
    
    try:
        client = get_meili_client()

        # Delete the index if it exists
        try:
            client.index("passages").delete()
            print("  Deleted MeiliSearch index")
        except Exception as e:
            if "not found" in str(e).lower():
                print("  MeiliSearch index already empty")
            else:
                print(f"  Error deleting MeiliSearch index: {e}")
                
    except Exception as e:
        print(f"Error connecting to MeiliSearch: {e}")

def reingest_document(file_path):
    """Re-ingest the document."""
    print(f"Re-ingesting document: {file_path}")
    
    try:
        # Run the ingest command
        cmd = [
            "poetry", "run", "python", "ingest.py", 
            file_path,
            "--title", "A Brief History of Guernsey County",
            "--debug"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Document re-ingested successfully")
            print("Output:")
            print(result.stdout)
        else:
            print("Error during ingestion:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running ingestion: {e}")
        return False
    
    return True

def run_embedding():
    """Run the embedding process."""
    print("Running embedding process...")
    
    try:
        # Run the embed command
        cmd = ["poetry", "run", "python", "embed.py"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Embedding completed successfully")
            print("Output:")
            print(result.stdout)
        else:
            print("Error during embedding:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running embedding: {e}")
        return False
    
    return True

def main():
    """Main function to clear and re-ingest."""
    print("=== Clean Re-ingestion Process ===")
    
    file_path = "data/1998 A Brief History of Guernsey County.pdf"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Step 1: Clear database
    clear_database()
    
    # Step 2: Clear MeiliSearch
    clear_meilisearch()
    
    # Step 3: Re-ingest document
    if not reingest_document(file_path):
        print("Failed to re-ingest document")
        return
    
    # Step 4: Run embedding
    if not run_embedding():
        print("Failed to run embedding")
        return
    
    print("\n=== Re-ingestion Complete ===")
    print("Database and MeiliSearch have been cleared and repopulated with level 1 headings only.")

if __name__ == "__main__":
    main()

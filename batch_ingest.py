#!/usr/bin/env python3
"""
Codex Batch Document Ingestion Script

Handles ingestion of multiple documents with support for:
- Directory processing (recursive)
- Metadata files (CSV/JSON)
- Parallel processing
- Progress tracking
"""

import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from config import configure_logging
from ingest import ingest_document

logger = logging.getLogger(__name__)

load_dotenv()


def load_metadata_file(metadata_path: str) -> dict[str, dict[str, Any]]:
    """Load document metadata from CSV or JSON file."""
    metadata = {}

    if metadata_path.lower().endswith(".json"):
        with Path(metadata_path).open() as f:
            data = json.load(f)
            for doc in data:
                if "filename" in doc:
                    metadata[doc["filename"]] = doc
    elif metadata_path.lower().endswith(".csv"):
        with Path(metadata_path).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "filename" in row:
                    metadata[row["filename"]] = row
    else:
        msg = "Metadata file must be CSV or JSON"
        raise ValueError(msg)

    return metadata


def get_document_files(input_path: str, recursive: bool = False) -> list[str]:
    """Get list of document files to process."""
    path = Path(input_path)
    files = []

    if path.is_file():
        # Single file
        if path.suffix.lower() in [".pdf", ".txt"]:
            files.append(str(path))
    elif path.is_dir():
        # Directory
        pattern = "**/*" if recursive else "*"

        files.extend(
            str(file_path)
            for file_path in path.glob(pattern)
            if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".txt"]
        )

    return sorted(files)


def extract_metadata_from_filename(file_path: str) -> dict[str, Any]:
    """Extract basic metadata from filename."""
    filename = Path(file_path).stem
    return {"title": filename, "authors": None, "pub_year": None, "language": "en"}


def ingest_single_document(
    file_path: str, metadata: dict[str, Any], debug: bool = False
) -> dict[str, Any]:
    """Ingest a single document using direct import of ingest_document."""
    try:
        title = metadata.get("title", Path(file_path).stem)

        authors = metadata.get("authors")
        if isinstance(authors, list):
            authors = ",".join(authors)

        pub_year = metadata.get("pub_year")
        if pub_year is not None:
            pub_year = int(pub_year)

        doc_id = ingest_document(
            file_path,
            title=title,
            authors=authors,
            pub_year=pub_year,
            debug=debug,
        )

        return {
            "file_path": file_path,
            "status": "success",
            "output": f"Ingested document {doc_id}",
        }

    except Exception as e:
        return {"file_path": file_path, "status": "error", "error": str(e)}


def process_documents_parallel(
    files: list[str],
    metadata_dict: dict[str, dict[str, Any]],
    max_workers: int,
    debug: bool,
) -> list[dict[str, Any]]:
    """Process documents in parallel."""
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {}
        for file_path in files:
            filename = Path(file_path).name
            metadata = metadata_dict.get(
                filename, extract_metadata_from_filename(file_path)
            )

            future = executor.submit(ingest_single_document, file_path, metadata, debug)
            future_to_file[future] = file_path

        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    logger.info("Success: %s", Path(file_path).name)
                else:
                    logger.error(
                        "Error: %s - %s",
                        Path(file_path).name,
                        result.get("error", "Unknown error"),
                    )

            except Exception as e:
                results.append(
                    {"file_path": file_path, "status": "error", "error": str(e)}
                )
                logger.error("Exception: %s - %s", Path(file_path).name, str(e))

    return results


def process_documents_sequential(
    files: list[str], metadata_dict: dict[str, dict[str, Any]], debug: bool
) -> list[dict[str, Any]]:
    """Process documents sequentially."""
    results = []

    for i, file_path in enumerate(files, 1):
        filename = Path(file_path).name
        metadata = metadata_dict.get(
            filename, extract_metadata_from_filename(file_path)
        )

        logger.info("Processing %d/%d: %s", i, len(files), filename)

        result = ingest_single_document(file_path, metadata, debug)
        results.append(result)

        if result["status"] == "success":
            logger.info("Success: %s", filename)
        else:
            logger.error(
                "Error: %s - %s", filename, result.get("error", "Unknown error")
            )

    return results


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--recursive", is_flag=True, help="Process directories recursively")
@click.option(
    "--metadata", type=click.Path(), help="CSV/JSON file with document metadata"
)
@click.option(
    "--parallel", default=1, help="Number of parallel processes (1 = sequential)"
)
@click.option("--debug", is_flag=True, help="Show debug information")
@click.option("--output", type=click.Path(), help="Save results to JSON file")
def main(
    input_path: str,
    recursive: bool,
    metadata: str | None,
    parallel: int,
    debug: bool,
    output: str | None,
):
    """Batch ingest multiple documents."""
    configure_logging(logging.DEBUG if debug else logging.INFO)

    logger.info("=== Codex Batch Ingestion ===")
    logger.info("Input: %s", input_path)
    logger.info("Recursive: %s", recursive)
    logger.info("Parallel: %d", parallel)
    logger.info("Debug: %s", debug)

    # Load metadata if provided
    metadata_dict = {}
    if metadata:
        logger.info("Loading metadata from: %s", metadata)
        metadata_dict = load_metadata_file(metadata)
        logger.info("Loaded metadata for %d documents", len(metadata_dict))

    # Get document files
    files = get_document_files(input_path, recursive)
    if not files:
        logger.warning("No document files found!")
        return

    logger.info("Found %d documents to process:", len(files))
    for file_path in files:
        logger.info("  - %s", Path(file_path).name)

    # Process documents
    logger.info("Starting ingestion...")
    if parallel > 1:
        results = process_documents_parallel(files, metadata_dict, parallel, debug)
    else:
        results = process_documents_sequential(files, metadata_dict, debug)

    # Summary
    logger.info("=== Ingestion Summary ===")
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    logger.info("Total: %d", len(results))
    logger.info("Successful: %d", successful)
    logger.info("Failed: %d", failed)

    if failed > 0:
        logger.info("Failed documents:")
        for result in results:
            if result["status"] == "error":
                logger.info(
                    "  - %s: %s",
                    Path(result["file_path"]).name,
                    result.get("error", "Unknown error"),
                )

    # Save results if requested
    if output:
        with Path(output).open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to: %s", output)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

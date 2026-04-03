#!/usr/bin/env python3
"""
Test script for advanced heading detection functionality.
"""

import sys
from pathlib import Path

from codex.ingestion.ingest import (
    extract_headings_by_typography,
    extract_headings_from_outline,
    extract_headings_from_toc_pages,
    filter_appendices,
    merge_heading_detection_methods,
    rank_headings_by_relevance,
)


def test_outline_extraction(file_path):
    """Test PDF outline/bookmarks extraction."""
    print("=== Testing PDF Outline Extraction ===")

    headings = extract_headings_from_outline(file_path)
    print(f"Found {len(headings)} headings from outline")

    for heading in headings[:10]:  # Show first 10
        print(
            f"  Level {heading['level']}: {heading['title']} (p. {heading['page'] + 1})"
        )

    return headings


def test_toc_parsing(file_path):
    """Test Table of Contents parsing."""
    print("\n=== Testing TOC Parsing ===")

    headings = extract_headings_from_toc_pages(file_path)
    print(f"Found {len(headings)} headings from TOC")

    for heading in headings[:10]:  # Show first 10
        print(f"  {heading['title']} (p. {heading['page'] + 1})")

    return headings


def test_typography_detection(file_path):
    """Test typography-based heading detection."""
    print("\n=== Testing Typography Detection ===")

    headings = extract_headings_by_typography(file_path)
    print(f"Found {len(headings)} headings by typography")

    for heading in headings[:10]:  # Show first 10
        heading.get("detection_method", "typography")
        size = heading.get("font_size", "N/A")
        ratio = heading.get("uppercase_ratio", "N/A")
        bold = heading.get("is_bold", False)
        print(
            f"  Level {heading['level']}: {heading['title']} (p. {heading['page'] + 1})"
        )
        print(f"    Size: {size}, Uppercase: {ratio:.2f}, Bold: {bold}")

    return headings


def test_appendix_filtering(headings):
    """Test appendix-specific filtering."""
    print("\n=== Testing Appendix Filtering ===")

    appendix_headings = filter_appendices(headings)
    print(f"Found {len(appendix_headings)} appendix headings")

    for heading in appendix_headings:
        print(f"  {heading['title']} (p. {heading['page'] + 1})")

    return appendix_headings


def test_merged_detection(file_path):
    """Test the complete merged heading detection."""
    print("\n=== Testing Merged Heading Detection ===")

    all_headings = merge_heading_detection_methods(file_path)
    print(f"Found {len(all_headings)} total headings after merging")

    # Group by detection method
    by_method = {}
    for heading in all_headings:
        method = heading.get("detection_method", "unknown")
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(heading)

    print("\nHeadings by detection method:")
    for method, headings in by_method.items():
        print(f"  {method}: {len(headings)} headings")
        for heading in headings[:3]:  # Show first 3 from each method
            confidence = heading.get("confidence", "N/A")
            title = heading.get("title", heading.get("text", "Unknown"))
            print(f"    {title} (conf: {confidence})")

    return all_headings


def test_heading_ranking(headings, test_queries):
    """Test heading ranking by relevance."""
    print("\n=== Testing Heading Ranking ===")

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        ranked = rank_headings_by_relevance(query, headings, limit=5)

        for i, heading in enumerate(ranked, 1):
            score = heading.get("relevance_score", 0)
            title = heading.get("title", heading.get("text", "Unknown"))
            print(f"  {i}. {title} (score: {score:.3f})")


def test_with_guernsey_pdf():
    """Test with the Guernsey County PDF."""
    file_path = "data/1998 A Brief History of Guernsey County.pdf"

    if not Path(file_path).exists():
        print(f"Error: PDF file not found at {file_path}")
        return

    print("Advanced Heading Detection Test")
    print("=" * 50)
    print(f"Testing with: {file_path}")

    # Test individual methods
    test_outline_extraction(file_path)
    test_toc_parsing(file_path)
    test_typography_detection(file_path)

    # Test merged detection
    all_headings = test_merged_detection(file_path)

    # Test appendix filtering
    test_appendix_filtering(all_headings)

    # Test heading ranking with sample queries
    test_queries = [
        "Civil War",
        "Morgan's Raid",
        "Appendix",
        "Mound Builders",
        "Preface",
    ]
    test_heading_ranking(all_headings, test_queries)

    # Summary statistics
    print("\n=== Summary ===")
    print(f"Total headings detected: {len(all_headings)}")

    # Show some example headings by page
    print("\nSample headings by page:")
    by_page = {}
    for heading in all_headings:
        page = heading["page"]
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(heading)

    for page in sorted(by_page.keys())[:5]:  # Show first 5 pages
        print(f"\nPage {page + 1}:")
        for heading in by_page[page]:
            method = heading.get("detection_method", "unknown")
            title = heading.get("title", heading.get("text", "Unknown"))
            print(f"  {title} ({method})")


def main():
    """Main test function."""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            return

        print(f"Testing with: {file_path}")
        all_headings = test_merged_detection(file_path)

        # Test ranking with some sample queries
        test_queries = ["chapter", "section", "appendix"]
        test_heading_ranking(all_headings, test_queries)
    else:
        # Test with the Guernsey County PDF
        test_with_guernsey_pdf()


if __name__ == "__main__":
    main()

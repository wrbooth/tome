"""
PDF and text file extraction for the Codex ingestion pipeline.

Provides functions to extract text with font information from PDFs
and plain text from TXT files.
"""

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from heading_detection import (
    detect_heading_patterns,
    merge_heading_detection_methods,
)


def extract_text_with_font_info(file_path: str) -> list[dict[str, Any]]:  # noqa: C901
    """Extract text with font information and advanced heading detection."""
    pages = []
    doc = fitz.open(file_path)

    # Use advanced heading detection methods (reuse already-opened doc)
    all_headings = merge_heading_detection_methods(file_path, doc=doc)

    # Group headings by page
    headings_by_page = {}
    for heading in all_headings:
        page_num = heading["page"]
        if page_num not in headings_by_page:
            headings_by_page[page_num] = []
        headings_by_page[page_num].append(heading)

    # Analyze font sizes across all pages to determine heading
    # thresholds (for backward compatibility)
    all_font_sizes = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    all_font_sizes.extend(span["size"] for span in line["spans"])

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
            title = heading.get("title", heading.get("text", ""))
            formatted_headings.append(
                {
                    "text": title,
                    "line_number": 0,  # Will be updated below
                    "level": heading["level"],
                    "full_text": title,
                    "detection_method": heading["detection_method"],
                    "confidence": heading.get("confidence", 0.5),
                }
            )

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
                        line_info.append(
                            {
                                "text": line_text.strip(),
                                "font_sizes": line_font_sizes,
                                "is_bold": line_is_bold,
                                "max_font_size": max(line_font_sizes)
                                if line_font_sizes
                                else 0,
                            }
                        )
                        page_text += line_text + "\n"

        # Match headings to line numbers
        for heading in formatted_headings:
            for i, line_data in enumerate(line_info):
                if (
                    heading["text"].lower() in line_data["text"].lower()
                    or line_data["text"].lower() in heading["text"].lower()
                ):
                    heading["line_number"] = i
                    break

        pages.append(
            {
                "page": page_num + 1,
                "text": page_text.strip(),
                "headings": formatted_headings,
                "font_thresholds": {
                    "heading": heading_threshold,
                    "major_heading": major_heading_threshold,
                },
            }
        )

    doc.close()
    return pages


def extract_text_from_txt(file_path: str) -> list[dict[str, Any]]:
    """Extract text from TXT file with heading detection."""
    with Path(file_path).open(encoding="utf-8") as f:
        content = f.read()

    # Split by double newlines to simulate pages
    sections = content.split("\n\n")
    pages = []

    for i, section in enumerate(sections):
        if section.strip():
            # Detect headings using regex patterns (allow all levels for TXT)
            headings = detect_heading_patterns(section, only_level_1=False)

            pages.append({"page": i + 1, "text": section.strip(), "headings": headings})

    return pages

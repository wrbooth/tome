"""
Heading detection functions for the Tome ingestion pipeline.

Provides regex-based, outline-based, TOC-based, and typography-based
heading detection, plus merging and ranking utilities.
"""

import logging
import re
from collections import Counter
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def detect_heading_patterns(
    text: str, only_level_1: bool = False
) -> list[dict[str, Any]]:
    """Detect headings using regex patterns."""
    headings = []

    # Common heading patterns with their levels
    patterns = [
        # Level 1: Main chapters/sections
        (r"^CHAPTER\s+\d+[:\s]*(.+)$", 1),  # CHAPTER 1: Title
        (r"^Chapter\s+\d+[:\s]*(.+)$", 1),  # Chapter 1: Title
        (r"^PART\s+\d+[:\s]*(.+)$", 1),  # PART 1: Title
        (r"^Part\s+\d+[:\s]*(.+)$", 1),  # Part 1: Title
        (r"^BOOK\s+\d+[:\s]*(.+)$", 1),  # BOOK 1: Title
        (r"^Book\s+\d+[:\s]*(.+)$", 1),  # Book 1: Title
        (
            r"^APPENDIX\s+[A-Z][-\d]*[:\s]*(.+)$",
            1,
        ),  # APPENDIX A: Title, APPENDIX E-4: Title
        (
            r"^Appendix\s+[A-Z][-\d]*[:\s]*(.+)$",
            1,
        ),  # Appendix A: Title, Appendix E-4: Title
        (
            r".*APPENDIX\s+[A-Z][-\d]*[:\s]*([^|]+)",
            1,
        ),  # APPENDIX E-4: Title (anywhere in line)
        (
            r".*APPENDIX\s+[A-Z][-\d]*\s+([A-Z][^|]+?)$",
            1,
        ),  # APPENDIX E-4 TITLE WORDS (no colon separator)
        (r"^[A-Z][A-Z\s]{8,}$", 1),  # Long ALL CAPS TITLE (8+ chars, major sections)
    ]

    # Add more detailed patterns only if not restricting to level 1
    if not only_level_1:
        patterns.extend(
            [
                # Level 2: Subsections
                (r"^Section\s+\d+\.\d+[:\s]*(.+)$", 2),  # Section 1.1: Title
                (r"^SECTION\s+\d+\.\d+[:\s]*(.+)$", 2),  # SECTION 1.1: Title
                (r"^\d+\.\d+\s+([A-Z][^.]*)$", 2),  # 1.1 Title (numbered sections)
                (r"^[A-Z][A-Z\s]{5,7}$", 2),  # Medium ALL CAPS TITLE (5-7 chars)
                # Level 3: Subsections
                (r"^\d+\.\d+\.\d+\s+([A-Z][^.]*)$", 3),  # 1.1.1 Title
                (r"^[A-Z][A-Z\s]{3,4}$", 3),  # Short ALL CAPS TITLE (3-4 chars)
                # Level 4: Minor headings
                (r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$", 4),  # Title Case
                (r"^\d+\.\s+([A-Z][^.]*)$", 4),  # 1. Title (numbered sections)
            ]
        )

    lines = text.split("\n")
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
                    headings.append(
                        {
                            "text": heading_text.strip(),
                            "line_number": i,
                            "level": level,
                            "full_text": line,
                            "detection_method": "regex",
                        }
                    )
                break

    return headings


def is_quality_heading(text: str, font_size: float, is_bold: bool) -> bool:  # noqa: C901
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
        r"^[^\w]*$",  # Only symbols
        r"^\d+$",  # Only numbers
        # Short ALL CAPS (8 chars or fewer,
        # complements 8+ detection threshold)
        r"^[A-Z][A-Z\s]{0,7}$",
        r"^[^\w\s]*$",  # Only symbols and spaces
        r"^[A-Z]\s*$",  # Single capital letter
        r"^\s*[^\w\s]+\s*$",  # Only symbols with spaces
        r"^[A-Z][A-Z\s]{0,2}$",  # Very short ALL CAPS
        r"^[^\w]*\d+[^\w]*$",  # Numbers with symbols
        r"^[A-Z]\d+[A-Z]*$",  # Letter-number combinations
        r"^[^\w]*[A-Z][^\w]*$",  # Single letter with symbols
        r"^[A-Z][A-Z\s]*\d+[A-Z\s]*$",  # ALL CAPS with numbers
        r"^[^\w\s]*[A-Z][^\w\s]*$",  # Single letter surrounded by symbols
        r"^[A-Z][A-Z\s]*[^\w\s]+[A-Z\s]*$",  # ALL CAPS with symbols
        r"^[^\w\s]*\d+[^\w\s]*$",  # Numbers surrounded by symbols
        r"^[A-Z][A-Z\s]*[^\w\s]+$",  # ALL CAPS ending with symbols
        r"^[^\w\s]+[A-Z][A-Z\s]*$",  # ALL CAPS starting with symbols
        # ALL CAPS with symbols throughout
        r"^[A-Z][A-Z\s]*[^\w\s]+[A-Z\s]*[^\w\s]*$",
        # ALL CAPS surrounded by symbols
        r"^[^\w\s]*[A-Z][A-Z\s]*[^\w\s]*$",
        # ALL CAPS with symbols at start and end
        r"^[A-Z][A-Z\s]*[^\w\s]+[A-Z\s]*[^\w\s]+$",
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

    # Allow appendix headings early -- before coordinate/table
    # checks that would reject them
    if re.match(r"APPENDIX\s+[A-Z][-\d]*[\s:]", text.upper()):
        return True

    # Additional checks for historical documents
    # Skip text that looks like OCR artifacts or formatting
    if re.search(r"[^\w\s]{3,}", text):  # 3+ consecutive symbols
        return False

    # Skip text that's mostly punctuation or special characters
    if len(re.findall(r"[^\w\s]", text)) > len(text) * 0.3:
        return False

    # Skip text that looks like coordinates or measurements
    if re.search(r"\d+[^\w\s]*[A-Z]", text) or re.search(r"[A-Z][^\w\s]*\d+", text):
        return False

    # Skip obvious table headers, lists, and fragments
    table_patterns = [
        r"^[A-Z][a-z]+,\s+[A-Z][a-z]+$",  # "Name, Location" pattern
        r"^\d+\s+[a-z]+",  # "1 horse" pattern
        r"^[A-Z][a-z]+\s+\d+",  # "Page 5" pattern
        r"^\d+\.\d+$",  # Just numbers like "3.00"
        r"^[A-Z]\s+[A-Z]$",  # Single letters like "A B"
        r"^[a-z]{1,3}$",  # Very short lowercase
        r"^\w+\s*\d+\s*\w*$",  # Pattern like "Item 1 A"
    ]

    for pattern in table_patterns:
        if re.match(pattern, text):
            return False

    # Must have at least 2 words for a meaningful heading
    return len(text.split()) >= 2


def merge_heading_detection(
    pages: list[dict[str, Any]], is_pdf: bool = True
) -> list[dict[str, Any]]:
    """Merge regex and font-based heading detection, preferring font detection."""
    total_headings = 0

    for page in pages:
        if "headings" not in page:
            # For pages without font analysis, use regex (level 1 and 2 for PDFs)
            page["headings"] = detect_heading_patterns(page["text"], only_level_1=False)
        else:
            # For PDF pages, combine font and regex detection (level 1 and 2)
            regex_headings = detect_heading_patterns(page["text"], only_level_1=False)

            # Create a map of line numbers to existing headings
            existing_headings = {h["line_number"]: h for h in page["headings"]}

            # Add regex headings that don't conflict with font headings
            for regex_heading in regex_headings:
                line_num = regex_heading["line_number"]
                if line_num not in existing_headings:
                    page["headings"].append(regex_heading)

            # Filter to level 1 and 2 headings for PDFs (allow appendix sections)
            if is_pdf:
                page["headings"] = [
                    h for h in page["headings"] if h.get("level", 1) in [1, 2]
                ]

            # Sort by line number
            page["headings"].sort(key=lambda x: x["line_number"])

        total_headings += len(page["headings"])

    logger.info("Detected %d total headings across all pages", total_headings)
    return pages


def debug_headings(pages: list[dict[str, Any]], max_pages: int = 3):
    """Debug function to show heading detection results."""
    logger.debug("=== HEADING DETECTION DEBUG ===")
    for _i, page in enumerate(pages[:max_pages]):
        logger.debug("Page %s:", page["page"])
        if page.get("headings"):
            for heading in page["headings"]:
                method = heading.get("detection_method", "unknown")
                level = heading.get("level", "?")
                text = heading.get("text", heading.get("title", ""))
                logger.debug("  Level %s (%s): %s", level, method, text)
        else:
            logger.debug("  No headings detected")
    logger.debug("=== END DEBUG ===")


def extract_headings_from_outline(
    file_path: str, doc: "fitz.Document | None" = None
) -> list[dict[str, Any]]:
    """
    Extract headings from PDF's built-in outline/bookmarks (fastest method).

    Returns:
        List of heading dictionaries with level, title, and page info
    """
    try:
        owned_doc = doc is None
        if owned_doc:
            doc = fitz.open(file_path)
        assert doc is not None  # noqa: S101  # noqa: S101
        toc = doc.get_toc(simple=True)  # Returns [[level, title, page], ...]

        headings = []
        for level, title, page in toc:
            title = title.strip()
            if not title:  # Skip empty titles
                continue

            # Filter out page number headings like "Page 1", "Page 2", etc.
            if re.match(r"^Page\s+\d+$", title, re.IGNORECASE):
                continue

            headings.append(
                {
                    "level": level,
                    "title": title,
                    "page": page - 1,  # Convert to 0-based indexing
                    "detection_method": "outline",
                }
            )

        if owned_doc:
            doc.close()
        return headings
    except Exception as e:
        logger.warning("Could not extract outline from %s: %s", file_path, e)
        return []


def extract_headings_from_toc_pages(  # noqa: C901
    file_path: str, doc: "fitz.Document | None" = None
) -> list[dict[str, Any]]:
    """
    Parse TABLE OF CONTENTS pages to extract headings.

    This method looks for TOC pages and parses the format where titles and page numbers
    are on separate lines.
    """
    try:
        owned_doc = doc is None
        if owned_doc:
            doc = fitz.open(file_path)
        assert doc is not None  # noqa: S101  # noqa: S101
        toc_pages = []

        # Find pages containing "TABLE OF CONTENTS"
        TOC_TITLE = "TABLE OF CONTENTS"
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text()
            if TOC_TITLE in text.upper():
                toc_pages.append(i)
                # Also check next page if it looks TOC-ish
                # (lots of lines ending in digits)
                if i + 1 < len(doc):
                    next_page = doc.load_page(i + 1)
                    next_text = next_page.get_text()
                    lines_with_digits = sum(
                        1
                        for line in next_text.splitlines()
                        if re.search(r"\d{1,4}\s*$", line.strip())
                    )
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
                if not title_line or "TABLE OF CONTENTS" in title_line.upper():
                    i += 1
                    continue

                # Check if next line is a page number
                page_match = re.match(r"^(\d{1,4})\s*$", page_line)
                if page_match:
                    page_num = int(page_match.group(1))
                    title = title_line.strip()

                    # Clean up title
                    title = re.sub(r"\s{2,}", " ", title)  # Normalize whitespace
                    title = title.strip("–-·•\t ")  # noqa: RUF001

                    # Skip very short titles or page numbers
                    if len(title) > 2 and not re.match(r"^\d+$", title):
                        items.append(
                            {
                                "level": 1,
                                "title": title,
                                "page": page_num - 1,  # Convert to 0-based indexing
                                "detection_method": "toc_parsing",
                            }
                        )

                    i += 2  # Skip both title and page number lines
                else:
                    i += 1  # Move to next line

        if owned_doc:
            doc.close()
        return items
    except Exception as e:
        logger.warning("Could not parse TOC from %s: %s", file_path, e)
        return []


def extract_headings_by_typography(  # noqa: C901
    file_path: str, doc: "fitz.Document | None" = None
) -> list[dict[str, Any]]:
    """
    Enhanced typography/layout detection for headings.

    Uses font size analysis, uppercase ratio, and positioning to detect headings.
    """
    try:
        owned_doc = doc is None
        if owned_doc:
            doc = fitz.open(file_path)
        assert doc is not None  # noqa: S101  # noqa: S101
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

                            spans.append(
                                {
                                    "page": page_num,
                                    "text": text,
                                    "size": span["size"],
                                    "flags": span.get("flags", 0),
                                    "origin": span.get("origin", (0, 0)),
                                    "bbox": span.get("bbox", (0, 0, 0, 0)),
                                }
                            )

        # Determine font size thresholds
        sizes = [round(span["size"], 1) for span in spans]
        size_counts = Counter(sizes)
        top_sizes = [size for size, _ in size_counts.most_common(5)]
        big_sizes = set(top_sizes[:2])  # Top 1-2 font sizes

        headings = []
        for span in spans:
            text = span["text"]
            size = round(span["size"], 1)

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
            if (uppercase_ratio > 0.6 or text.istitle()) and not text.endswith(
                (".", ":", ";", ",")
            ):
                is_heading = True

            # Check for bold formatting
            is_bold = bool(span["flags"] & 2**4)
            if is_bold and len(text) > 3:
                is_heading = True

            # Check for common heading patterns
            heading_patterns = [
                r"^CHAPTER\s+\d+",
                r"^Chapter\s+\d+",
                r"^PART\s+\d+",
                r"^APPENDIX\s+[A-Z]",
                r"^Appendix\s+[A-Z]",
                r"^[A-Z][A-Z\s]{8,}$",  # Long ALL CAPS
            ]

            for pattern in heading_patterns:
                if re.match(pattern, text):
                    is_heading = True
                    break

            if is_heading:
                # For PDFs, only use level 1 headings to reduce noise
                level = 1

                headings.append(
                    {
                        "level": level,
                        "title": " ".join(text.split()),  # Normalize whitespace
                        "page": span["page"],
                        "detection_method": "typography",
                        "font_size": size,
                        "uppercase_ratio": uppercase_ratio,
                        "is_bold": is_bold,
                    }
                )

        # De-duplicate consecutive duplicates
        deduped = []
        seen = set()
        for heading in headings:
            key = (heading["page"], heading["title"].lower())
            if key not in seen:
                seen.add(key)
                deduped.append(heading)

        if owned_doc:
            doc.close()
        return deduped
    except Exception as e:
        logger.warning(
            "Could not extract headings by typography from %s: %s", file_path, e
        )
        return []


def filter_appendices(headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter and enhance appendix headings specifically.

    Historical PDFs often have clear appendix patterns that should be preserved.
    """
    appendix_patterns = [
        r"^APPENDIX\s+[A-Z][-\d]*",
        r"^Appendix\s+[A-Z][-\d]*",
        r".*APPENDIX\s+[A-Z][-\d]*\s+([^|]+?)(?:\s+Damage|\s+Claimant|$)",
    ]

    appendix_headings = []
    for heading in headings:
        # Handle different heading formats
        title = heading.get("title", heading.get("text", ""))
        if not title:
            continue

        for pattern in appendix_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                appendix_headings.append(heading)
                break

    return appendix_headings


def merge_heading_detection_methods(
    file_path: str, doc: "fitz.Document | None" = None
) -> list[dict[str, Any]]:
    """
    Combine all heading detection methods and merge results intelligently.

    Returns a unified list of headings with confidence scores.
    """
    owned_doc = doc is None
    if owned_doc:
        doc = fitz.open(file_path)
    assert doc is not None  # noqa: S101

    all_headings = []

    # Method 1: PDF Outline (fastest, most reliable)
    outline_headings = extract_headings_from_outline(file_path, doc=doc)
    for heading in outline_headings:
        heading["confidence"] = 0.9  # High confidence for outline
        all_headings.append(heading)

    # Method 2: TOC Parsing
    toc_headings = extract_headings_from_toc_pages(file_path, doc=doc)
    for heading in toc_headings:
        heading["confidence"] = 0.8  # High confidence for TOC
        all_headings.append(heading)

    # Method 3: Typography Detection
    typography_headings = extract_headings_by_typography(file_path, doc=doc)
    for heading in typography_headings:
        heading["confidence"] = 0.6  # Medium confidence for typography
        all_headings.append(heading)

    # Method 4: Regex Patterns (existing method)
    regex_headings = detect_heading_patterns_from_file(file_path, doc=doc)
    for heading in regex_headings:
        heading["confidence"] = 0.5  # Lower confidence for regex
        all_headings.append(heading)

    # Merge and deduplicate
    merged = merge_heading_results(all_headings)

    # Filter to only level 1 headings for PDFs to reduce noise
    merged = [h for h in merged if h.get("level", 1) == 1]

    # Add appendix-specific filtering
    appendix_headings = filter_appendices(merged)
    if appendix_headings:
        # Ensure appendices are included even if they have lower confidence
        for appendix in appendix_headings:
            appendix_title = appendix.get("title", appendix.get("text", ""))
            appendix_page = appendix.get("page", 0)

            if not any(
                h.get("title", h.get("text", "")).lower() == appendix_title.lower()
                and h.get("page", 0) == appendix_page
                for h in merged
            ):
                appendix["confidence"] = max(appendix.get("confidence", 0), 0.7)
                merged.append(appendix)

    if owned_doc:
        doc.close()
    return merged


def detect_heading_patterns_from_file(
    file_path: str, doc: "fitz.Document | None" = None
) -> list[dict[str, Any]]:
    """
    Extract headings using regex patterns from a file.
    """
    try:
        owned_doc = doc is None
        if owned_doc:
            doc = fitz.open(file_path)
        assert doc is not None  # noqa: S101  # noqa: S101
        headings = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()

            # Use existing regex detection
            page_headings = detect_heading_patterns(text, only_level_1=False)

            for heading in page_headings:
                heading["page"] = page_num
                heading["detection_method"] = "regex"
                headings.append(heading)

        if owned_doc:
            doc.close()
        return headings
    except Exception as e:
        logger.warning("Could not extract headings by regex from %s: %s", file_path, e)
        return []


def merge_heading_results(headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge heading results from multiple detection methods.

    Handles duplicates by keeping the highest confidence detection.
    """
    # Group by page and title (case-insensitive)
    grouped: dict[tuple[int, str], dict[str, Any]] = {}

    for heading in headings:
        # Handle different heading formats
        title = heading.get("title", heading.get("text", ""))
        page = heading.get("page", 0)

        if not title:  # Skip headings without title
            continue

        key = (page, title.lower())
        confidence = heading.get("confidence", 0)

        if key not in grouped or confidence > grouped[key].get("confidence", 0):
            grouped[key] = heading

    # Convert back to list and sort by page, then by line number
    merged = list(grouped.values())
    merged.sort(key=lambda x: (x.get("page", 0), x.get("line_number", 0)))

    return merged


def rank_headings_by_relevance(
    query: str, headings: list[dict[str, Any]], limit: int = 10
) -> list[dict[str, Any]]:
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
        from rapidfuzz import fuzz, process

        use_rapidfuzz = True
    except ImportError:
        # Fallback to simple string matching
        use_rapidfuzz = False
        logger.warning("rapidfuzz not available, using simple string matching")

    if use_rapidfuzz:
        # Use rapidfuzz for sophisticated fuzzy matching
        choices = [h.get("title", h.get("text", "")) for h in headings]
        ranked = process.extract(
            query, choices, scorer=fuzz.token_set_ratio, limit=limit
        )

        # Attach scores to headings
        out = []
        for _title, score, idx in ranked:
            h = headings[idx].copy()
            h["relevance_score"] = score / 100.0  # Normalize to 0-1
            out.append(h)

        return out
    # Simple fallback: exact and partial string matching
    query_lower = query.lower()
    scored_headings = []

    for heading in headings:
        title_lower = heading.get("title", heading.get("text", "")).lower()
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
            h["relevance_score"] = score
            scored_headings.append(h)

    # Sort by score and return top results
    scored_headings.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored_headings[:limit]

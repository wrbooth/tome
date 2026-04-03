"""Tests for chunk_text_with_headings in ingest.py."""

from tome.ingestion.ingest import chunk_text_with_headings


class TestChunkTextWithHeadings:
    def test_single_page_single_chunk(self):
        """Short text produces exactly one chunk."""
        pages = [{"page": 1, "text": "Short paragraph here.", "headings": []}]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0]["page"] == 1

    def test_text_exceeding_max_tokens_splits(self):
        """Text over max_tokens produces multiple chunks (needs paragraph breaks)."""
        paragraphs = [
            f"Paragraph {i} with some extra words to pad it out." for i in range(30)
        ]
        long_text = "\n".join(paragraphs)
        pages = [{"page": 1, "text": long_text, "headings": []}]
        chunks = chunk_text_with_headings(pages, max_tokens=30)
        assert len(chunks) > 1

    def test_document_title_prefix(self):
        """Document title appears in chunk text prefix."""
        pages = [{"page": 1, "text": "Some content here.", "headings": []}]
        chunks = chunk_text_with_headings(
            pages, max_tokens=500, document_title="My Document"
        )
        assert chunks[0]["text"].startswith("My Document |")

    def test_no_title_no_headings_no_prefix(self):
        """Without title or headings, text has no prefix."""
        pages = [{"page": 1, "text": "Plain content.", "headings": []}]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        assert chunks[0]["text"] == "Plain content."

    def test_heading_hierarchy_level1(self):
        """Level 1 heading sets the headings_path."""
        pages = [
            {
                "page": 1,
                "text": "CHAPTER 1: Intro\nBody text paragraph here.",
                "headings": [{"text": "Intro", "line_number": 0, "level": 1}],
            }
        ]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        assert "Intro" in chunks[0]["headings_path"]

    def test_heading_hierarchy_level2_appends(self):
        """Level 2 heading appends to path under level 1."""
        pages = [
            {
                "page": 1,
                "text": "Chapter Heading\nSection Heading\nBody text.",
                "headings": [
                    {"text": "Chapter", "line_number": 0, "level": 1},
                    {"text": "Section", "line_number": 1, "level": 2},
                ],
            }
        ]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        assert chunks[0]["headings_path"] == ["Chapter", "Section"]

    def test_new_level1_resets_path(self):
        """A new level 1 heading resets the heading path."""
        # Build text that's long enough to force multiple chunks
        body = "\n".join(["Paragraph text. " * 20] * 3)
        text = f"First Chapter\n{body}\nSecond Chapter\n{body}"
        pages = [
            {
                "page": 1,
                "text": text,
                "headings": [
                    {"text": "First Chapter", "line_number": 0, "level": 1},
                    {"text": "Second Chapter", "line_number": 4, "level": 1},
                ],
            }
        ]
        chunks = chunk_text_with_headings(pages, max_tokens=50)
        # Find chunks after "Second Chapter" heading
        second_chapter_chunks = [
            c for c in chunks if "Second Chapter" in c["headings_path"]
        ]
        for c in second_chapter_chunks:
            assert "First Chapter" not in c["headings_path"]

    def test_title_and_headings_prefix_format(self):
        """Prefix is 'Title | H1 | H2 | text'."""
        pages = [
            {
                "page": 1,
                "text": "Main Heading\nSub Heading\nBody content here.",
                "headings": [
                    {"text": "Main", "line_number": 0, "level": 1},
                    {"text": "Sub", "line_number": 1, "level": 2},
                ],
            }
        ]
        chunks = chunk_text_with_headings(pages, max_tokens=500, document_title="Doc")
        assert chunks[0]["text"].startswith("Doc | Main | Sub |")

    def test_original_text_excludes_prefix(self):
        """original_text field has no prefix."""
        pages = [
            {
                "page": 1,
                "text": "Heading Line\nSome content.",
                "headings": [{"text": "Heading", "line_number": 0, "level": 1}],
            }
        ]
        chunks = chunk_text_with_headings(pages, max_tokens=500, document_title="Title")
        assert not chunks[0]["original_text"].startswith("Title")

    def test_multi_page_headings_carry_across(self):
        """Headings from page 1 carry into page 2 chunks."""
        pages = [
            {
                "page": 1,
                "text": "Chapter One\nPage one content.",
                "headings": [{"text": "Chapter One", "line_number": 0, "level": 1}],
            },
            {"page": 2, "text": "More content on page two.", "headings": []},
        ]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        # Page 2 chunk should still have "Chapter One" in heading path
        page2_chunks = [c for c in chunks if c["page"] == 2]
        assert len(page2_chunks) >= 1, "Expected at least one chunk from page 2"
        assert "Chapter One" in page2_chunks[0]["headings_path"]

    def test_empty_paragraphs_skipped(self):
        """Empty lines don't create empty chunks."""
        pages = [{"page": 1, "text": "Content.\n\n\nMore content.", "headings": []}]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        for chunk in chunks:
            assert chunk["original_text"].strip() != ""

    def test_overlap_last_paragraph_carried(self):
        """Last paragraph of chunk N appears as first of chunk N+1."""
        # Create text with distinct paragraphs that will force splitting
        paras = [f"Paragraph number {i} with some extra words." for i in range(20)]
        text = "\n".join(paras)
        pages = [{"page": 1, "text": text, "headings": []}]
        chunks = chunk_text_with_headings(pages, max_tokens=30)
        assert len(chunks) >= 2, "Expected at least 2 chunks for overlap test"
        # The last sentence/paragraph of chunk 0 should appear in chunk 1 (overlap)
        # Paragraphs are joined with spaces in output,
        # so split on ". " to find last paragraph
        chunk0_text = chunks[0]["original_text"].strip()
        # Find the last complete sentence (paragraph) in chunk 0
        sentences = [s.strip() for s in chunk0_text.split(". ") if s.strip()]
        last_sentence = sentences[-1].rstrip(".")
        assert last_sentence in chunks[1]["original_text"], (
            f"Expected last paragraph of chunk 0"
            f" ({last_sentence!r}) to appear in chunk 1"
        )

    def test_chunk_has_required_keys(self):
        """Each chunk has page, text, original_text, headings_path."""
        pages = [{"page": 1, "text": "Some content here.", "headings": []}]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        required = {"page", "text", "original_text", "headings_path"}
        for chunk in chunks:
            assert required.issubset(chunk.keys())

    def test_headings_path_reflects_state_at_creation(self):
        """headings_path is a snapshot, not a reference to mutable state."""
        pages = [
            {
                "page": 1,
                "text": "First Heading\n"
                + " ".join(["word"] * 200)
                + "\nSecond Heading\nMore text.",
                "headings": [
                    {"text": "First", "line_number": 0, "level": 1},
                    {"text": "Second", "line_number": 2, "level": 1},
                ],
            }
        ]
        chunks = chunk_text_with_headings(pages, max_tokens=50)
        # First chunk's headings_path should not be mutated by later headings
        first_chunks = [c for c in chunks if "First" in c.get("headings_path", [])]
        for c in first_chunks:
            assert "Second" not in c["headings_path"]

    def test_empty_pages_produce_no_chunks(self):
        """Pages with only whitespace produce no chunks."""
        pages = [{"page": 1, "text": "   \n\n   ", "headings": []}]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        assert len(chunks) == 0

    def test_page_number_correct(self):
        """Chunks have the correct page number from their source page."""
        pages = [
            {"page": 5, "text": "Page five content.", "headings": []},
            {"page": 6, "text": "Page six content.", "headings": []},
        ]
        chunks = chunk_text_with_headings(pages, max_tokens=500)
        assert any(c["page"] == 5 for c in chunks)
        assert any(c["page"] == 6 for c in chunks)

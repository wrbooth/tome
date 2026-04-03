"""Tests for impure (I/O-dependent) functions in ingest.py, using mocks."""

from unittest.mock import MagicMock, patch

import pytest

from codex.ingestion.ingest import (
    debug_headings,
    detect_heading_patterns_from_file,
    extract_headings_by_typography,
    extract_headings_from_outline,
    extract_headings_from_toc_pages,
    extract_text_from_txt,
    extract_text_with_font_info,
    index_in_meilisearch,
    ingest_document,
    merge_heading_detection_methods,
    store_document,
    store_passages,
)
from tests.unit.conftest import make_mock_db_connection

# ── extract_text_from_txt ─────────────────────────────────────────────────


class TestExtractTextFromTxt:
    def test_single_section(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("CHAPTER 1: The Beginning\nFirst paragraph content here.")
        pages = extract_text_from_txt(str(f))
        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert "First paragraph" in pages[0]["text"]

    def test_multiple_sections_split_by_double_newline(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Section one text.\n\nSection two text.\n\nSection three text.")
        pages = extract_text_from_txt(str(f))
        assert len(pages) == 3
        assert pages[0]["page"] == 1
        assert pages[2]["page"] == 3

    def test_empty_sections_skipped(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Content.\n\n\n\n\n\nMore content.")
        pages = extract_text_from_txt(str(f))
        # Empty sections between double newlines are skipped
        assert all(p["text"].strip() for p in pages)

    def test_headings_detected_in_txt(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(
            "CHAPTER 1: The Introduction\nSome body text here.\n\nNext section text."
        )
        pages = extract_text_from_txt(str(f))
        assert len(pages[0].get("headings", [])) >= 1

    def test_empty_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("")
        pages = extract_text_from_txt(str(f))
        assert pages == []


# ── debug_headings ────────────────────────────────────────────────────────


class TestDebugHeadings:
    def test_with_headings(self, caplog):
        pages = [
            {
                "page": 1,
                "headings": [
                    {"text": "Chapter One", "detection_method": "regex", "level": 1}
                ],
            }
        ]
        with caplog.at_level("DEBUG", logger="codex.ingestion.heading_detection"):
            debug_headings(pages)
        assert "HEADING DETECTION DEBUG" in caplog.text
        assert "Chapter One" in caplog.text
        assert "Level 1" in caplog.text

    def test_no_headings(self, caplog):
        pages = [{"page": 1, "headings": []}]
        with caplog.at_level("DEBUG", logger="codex.ingestion.heading_detection"):
            debug_headings(pages)
        assert "No headings detected" in caplog.text

    def test_respects_max_pages(self, caplog):
        pages = [{"page": i, "headings": []} for i in range(10)]
        with caplog.at_level("DEBUG", logger="codex.ingestion.heading_detection"):
            debug_headings(pages, max_pages=2)
        assert "Page 0" in caplog.text
        assert "Page 1" in caplog.text
        assert "Page 2" not in caplog.text

    def test_heading_with_title_key(self, caplog):
        pages = [{"page": 1, "headings": [{"title": "Appendix A", "level": 1}]}]
        with caplog.at_level("DEBUG", logger="codex.ingestion.heading_detection"):
            debug_headings(pages)
        assert "Appendix A" in caplog.text


# ── store_document ────────────────────────────────────────────────────────


class TestStoreDocument:
    def test_inserts_and_returns_doc_id(self, mock_db_conn):
        doc_id = store_document(mock_db_conn, "Test Doc", "/path/to/doc.pdf")
        assert isinstance(doc_id, str)
        assert len(doc_id) == 36  # UUID format
        mock_db_conn.commit.assert_called_once()

    def test_passes_authors_and_year(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        store_document(mock_db_conn, "Doc", "/path", authors=["A", "B"], pub_year=2000)
        call_args = cursor.execute.call_args[0]
        sql = call_args[0]
        params = call_args[1]
        assert "INSERT INTO documents" in sql
        assert params[1] == "Doc"
        assert params[2] == ["A", "B"]
        assert params[3] == 2000

    def test_none_authors_and_year(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        store_document(mock_db_conn, "Doc", "/path")
        params = cursor.execute.call_args[0][1]
        assert params[2] is None  # authors
        assert params[3] is None  # pub_year


# ── store_passages ────────────────────────────────────────────────────────


class TestStorePassages:
    def test_returns_passage_ids(self, mock_db_conn):
        chunks = [
            {
                "page": 1,
                "text": "Hello world.",
                "original_text": "Hello world.",
                "headings_path": ["Ch1"],
            },
        ]
        with patch("codex.ingestion.storage.extract_entities_and_years", return_value=([], [])):
            ids = store_passages(mock_db_conn, "doc-1", chunks)
            assert len(ids) == 1
            assert len(ids[0]) == 36

    def test_stores_entities_and_years(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        entities = [
            {"entity": "Washington", "ent_type": "PERSON", "norm_entity": "washington"}
        ]
        years = [1776]
        chunks = [
            {"page": 1, "text": "Text.", "original_text": "Text.", "headings_path": []},
        ]
        with patch(
            "codex.ingestion.storage.extract_entities_and_years", return_value=(entities, years)
        ):
            store_passages(mock_db_conn, "doc-1", chunks)
            # Should have INSERT for passage + entity + year = 3 execute calls
            assert cursor.execute.call_count == 3

    def test_multiple_chunks(self, mock_db_conn):
        chunks = [
            {
                "page": i,
                "text": f"Text {i}.",
                "original_text": f"Text {i}.",
                "headings_path": [],
            }
            for i in range(3)
        ]
        with patch("codex.ingestion.storage.extract_entities_and_years", return_value=([], [])):
            ids = store_passages(mock_db_conn, "doc-1", chunks)
            assert len(ids) == 3
            mock_db_conn.commit.assert_called_once()

    def test_uses_original_text_for_entities(self, mock_db_conn):
        chunks = [
            {
                "page": 1,
                "text": "Title | Prefixed text.",
                "original_text": "Unprefixed text.",
                "headings_path": ["Title"],
            },
        ]
        with patch(
            "codex.ingestion.storage.extract_entities_and_years", return_value=([], [])
        ) as mock_extract:
            store_passages(mock_db_conn, "doc-1", chunks)
            mock_extract.assert_called_with("Unprefixed text.")


# ── index_in_meilisearch ─────────────────────────────────────────────────


class TestIndexInMeilisearch:
    def test_successful_indexing(self, mock_meili_client):
        passages = [
            {"page": 1, "text": "Text.", "original_text": "Text.", "headings_path": []},
        ]
        passage_ids = ["p-1"]
        with (
            patch("codex.ingestion.storage.get_meili_client", return_value=mock_meili_client),
            patch("codex.ingestion.storage.extract_entities_and_years", return_value=([], [])),
        ):
            index_in_meilisearch(passages, passage_ids, "doc-1")
            mock_meili_client.index("passages").add_documents.assert_called_once()

    def test_sets_up_embedder_when_needed(self, mock_meili_client):
        mock_meili_client.index(
            "passages"
        ).get_settings.return_value = {}  # no embedders
        passages = [
            {"page": 1, "text": "Text.", "original_text": "Text.", "headings_path": []},
        ]
        with (
            patch("codex.ingestion.storage.get_meili_client", return_value=mock_meili_client),
            patch("codex.ingestion.storage.extract_entities_and_years", return_value=([], [])),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            index_in_meilisearch(passages, ["p-1"], "doc-1")
            mock_meili_client.index("passages").update_embedders.assert_called_once()
            mock_meili_client.index(
                "passages"
            ).update_filterable_attributes.assert_called_once()

    def test_skips_embedder_setup_when_exists(self, mock_meili_client):
        # get_settings returns embedders already configured (default from fixture)
        passages = [
            {"page": 1, "text": "Text.", "original_text": "Text.", "headings_path": []},
        ]
        with (
            patch("codex.ingestion.storage.get_meili_client", return_value=mock_meili_client),
            patch("codex.ingestion.storage.extract_entities_and_years", return_value=([], [])),
        ):
            index_in_meilisearch(passages, ["p-1"], "doc-1")
            mock_meili_client.index("passages").update_embedders.assert_not_called()

    def test_meilisearch_exception_handled(self, caplog):
        with patch(
            "codex.ingestion.storage.get_meili_client", side_effect=Exception("Connection refused")
        ):
            with caplog.at_level("WARNING", logger="codex.ingestion.storage"):
                index_in_meilisearch([], [], "doc-1")
            assert any("Meilisearch" in r.message for r in caplog.records)

    def test_import_error_handled(self, caplog):
        with patch("codex.ingestion.storage.get_meili_client", side_effect=ImportError("no module")):
            with caplog.at_level("WARNING", logger="codex.ingestion.storage"):
                index_in_meilisearch([], [], "doc-1")
            assert any("Meilisearch" in r.message for r in caplog.records)


# ── extract_headings_from_outline ─────────────────────────────────────────


class TestExtractHeadingsFromOutline:
    def test_extracts_toc_entries(self):
        mock_doc = MagicMock()
        mock_doc.get_toc.return_value = [
            [1, "Chapter 1", 5],
            [2, "Section 1.1", 6],
        ]
        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = extract_headings_from_outline("fake.pdf")
            assert len(headings) == 2
            assert headings[0]["title"] == "Chapter 1"
            assert headings[0]["page"] == 4  # 0-based
            assert headings[0]["level"] == 1
            assert headings[0]["detection_method"] == "outline"

    def test_skips_empty_titles(self):
        mock_doc = MagicMock()
        mock_doc.get_toc.return_value = [[1, "", 1], [1, "Valid", 2]]
        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = extract_headings_from_outline("fake.pdf")
            assert len(headings) == 1

    def test_skips_page_number_headings(self):
        mock_doc = MagicMock()
        mock_doc.get_toc.return_value = [[1, "Page 5", 5], [1, "Real Heading", 6]]
        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = extract_headings_from_outline("fake.pdf")
            assert len(headings) == 1
            assert headings[0]["title"] == "Real Heading"

    def test_exception_returns_empty(self):
        with patch(
            "codex.ingestion.heading_detection.fitz.open", side_effect=Exception("File not found")
        ):
            headings = extract_headings_from_outline("missing.pdf")
            assert headings == []


# ── extract_headings_from_toc_pages ───────────────────────────────────────


class TestExtractHeadingsFromTocPages:
    def test_parses_toc_format(self):
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 2

        toc_text = "TABLE OF CONTENTS\nChapter One\n5\nChapter Two\n10\n"
        body_text = "Regular body text content.\nMore text."

        def load_page(idx):
            page = MagicMock()
            page.get_text.return_value = toc_text if idx == 0 else body_text
            return page

        mock_doc.load_page.side_effect = load_page

        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = extract_headings_from_toc_pages("fake.pdf")
            assert len(headings) >= 1
            assert headings[0]["detection_method"] == "toc_parsing"

    def test_exception_returns_empty(self):
        with patch("codex.ingestion.heading_detection.fitz.open", side_effect=Exception("Error")):
            assert extract_headings_from_toc_pages("missing.pdf") == []


# ── extract_headings_by_typography ────────────────────────────────────────


class TestExtractHeadingsByTypography:
    def _make_mock_doc(self, spans_data):
        """Helper to build a mock fitz document with given spans."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        blocks = []
        for text, size, flags in spans_data:
            blocks.append(
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": text,
                                    "size": size,
                                    "flags": flags,
                                    "origin": (0, 0),
                                    "bbox": (0, 0, 100, 20),
                                }
                            ]
                        }
                    ]
                }
            )
        page.get_text.return_value = {"blocks": blocks}
        mock_doc.load_page.return_value = page
        return mock_doc

    def test_detects_large_uppercase_heading(self):
        # Large font, uppercase text
        spans = [
            ("INTRODUCTION", 16.0, 0),
            ("Body text here.", 12.0, 0),
            ("More body text.", 12.0, 0),
        ]
        mock_doc = self._make_mock_doc(spans)
        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = extract_headings_by_typography("fake.pdf")
            titles = [h["title"] for h in headings]
            assert "INTRODUCTION" in titles

    def test_detects_bold_heading(self):
        # Bold flag = 2^4 = 16
        spans = [
            ("Bold Heading", 12.0, 16),
            ("Normal text.", 12.0, 0),
        ]
        mock_doc = self._make_mock_doc(spans)
        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = extract_headings_by_typography("fake.pdf")
            titles = [h["title"] for h in headings]
            assert "Bold Heading" in titles

    def test_skips_short_text(self):
        spans = [
            ("Hi", 16.0, 0),
            ("Body text.", 12.0, 0),
        ]
        mock_doc = self._make_mock_doc(spans)
        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = extract_headings_by_typography("fake.pdf")
            titles = [h["title"] for h in headings]
            assert "Hi" not in titles

    def test_deduplicates(self):
        spans = [
            ("SAME HEADING", 16.0, 0),
            ("SAME HEADING", 16.0, 0),
        ]
        mock_doc = self._make_mock_doc(spans)
        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = extract_headings_by_typography("fake.pdf")
            titles = [h["title"] for h in headings]
            assert titles.count("SAME HEADING") <= 1

    def test_exception_returns_empty(self):
        with patch("codex.ingestion.heading_detection.fitz.open", side_effect=Exception("Error")):
            assert extract_headings_by_typography("missing.pdf") == []


# ── detect_heading_patterns_from_file ─────────────────────────────────────


class TestDetectHeadingPatternsFromFile:
    def test_detects_headings_per_page(self):
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        page0 = MagicMock()
        page0.get_text.return_value = "CHAPTER 1: First Chapter Title\nBody text."
        page1 = MagicMock()
        page1.get_text.return_value = "CHAPTER 2: Second Chapter Title\nMore text."
        mock_doc.load_page.side_effect = [page0, page1]

        with patch("codex.ingestion.heading_detection.fitz.open", return_value=mock_doc):
            headings = detect_heading_patterns_from_file("fake.pdf")
            assert len(headings) >= 2
            assert headings[0]["page"] == 0
            assert headings[0]["detection_method"] == "regex"

    def test_exception_returns_empty(self):
        with patch("codex.ingestion.heading_detection.fitz.open", side_effect=Exception("Error")):
            assert detect_heading_patterns_from_file("missing.pdf") == []


# ── merge_heading_detection_methods ───────────────────────────────────────


class TestMergeHeadingDetectionMethods:
    def test_combines_all_methods(self):
        with (
            patch("codex.ingestion.heading_detection.fitz.open", return_value=MagicMock()),
            patch(
                "codex.ingestion.heading_detection.extract_headings_from_outline",
                return_value=[
                    {
                        "level": 1,
                        "title": "Outline Heading",
                        "page": 0,
                        "detection_method": "outline",
                    }
                ],
            ),
            patch("codex.ingestion.heading_detection.extract_headings_from_toc_pages", return_value=[]),
            patch("codex.ingestion.heading_detection.extract_headings_by_typography", return_value=[]),
            patch(
                "codex.ingestion.heading_detection.detect_heading_patterns_from_file", return_value=[]
            ),
        ):
            headings = merge_heading_detection_methods("fake.pdf")
            assert len(headings) >= 1
            assert headings[0]["confidence"] == 0.9

    def test_assigns_correct_confidence_scores(self):
        with (
            patch("codex.ingestion.heading_detection.fitz.open", return_value=MagicMock()),
            patch(
                "codex.ingestion.heading_detection.extract_headings_from_outline",
                return_value=[
                    {"level": 1, "title": "A", "page": 0, "detection_method": "outline"}
                ],
            ),
            patch(
                "codex.ingestion.heading_detection.extract_headings_from_toc_pages",
                return_value=[
                    {
                        "level": 1,
                        "title": "B",
                        "page": 1,
                        "detection_method": "toc_parsing",
                    }
                ],
            ),
            patch(
                "codex.ingestion.heading_detection.extract_headings_by_typography",
                return_value=[
                    {
                        "level": 1,
                        "title": "C",
                        "page": 2,
                        "detection_method": "typography",
                    }
                ],
            ),
            patch(
                "codex.ingestion.heading_detection.detect_heading_patterns_from_file",
                return_value=[
                    {
                        "level": 1,
                        "title": "D",
                        "page": 3,
                        "detection_method": "regex",
                        "line_number": 0,
                    }
                ],
            ),
        ):
            headings = merge_heading_detection_methods("fake.pdf")
            confidence_by_title = {
                h.get("title", h.get("text", "")): h["confidence"] for h in headings
            }
            assert confidence_by_title.get("A") == 0.9
            assert confidence_by_title.get("B") == 0.8
            assert confidence_by_title.get("C") == 0.6
            assert confidence_by_title.get("D") == 0.5

    def test_filters_to_level_1(self):
        with (
            patch("codex.ingestion.heading_detection.fitz.open", return_value=MagicMock()),
            patch(
                "codex.ingestion.heading_detection.extract_headings_from_outline",
                return_value=[
                    {
                        "level": 1,
                        "title": "Level 1",
                        "page": 0,
                        "detection_method": "outline",
                    },
                    {
                        "level": 2,
                        "title": "Level 2",
                        "page": 0,
                        "detection_method": "outline",
                    },
                ],
            ),
            patch("codex.ingestion.heading_detection.extract_headings_from_toc_pages", return_value=[]),
            patch("codex.ingestion.heading_detection.extract_headings_by_typography", return_value=[]),
            patch(
                "codex.ingestion.heading_detection.detect_heading_patterns_from_file", return_value=[]
            ),
        ):
            headings = merge_heading_detection_methods("fake.pdf")
            levels = [h["level"] for h in headings]
            assert all(level == 1 for level in levels)

    def test_preserves_appendix_headings(self):
        with (
            patch("codex.ingestion.heading_detection.fitz.open", return_value=MagicMock()),
            patch("codex.ingestion.heading_detection.extract_headings_from_outline", return_value=[]),
            patch("codex.ingestion.heading_detection.extract_headings_from_toc_pages", return_value=[]),
            patch(
                "codex.ingestion.heading_detection.extract_headings_by_typography",
                return_value=[
                    {
                        "level": 1,
                        "title": "APPENDIX A: Data",
                        "page": 10,
                        "detection_method": "typography",
                    }
                ],
            ),
            patch(
                "codex.ingestion.heading_detection.detect_heading_patterns_from_file", return_value=[]
            ),
        ):
            headings = merge_heading_detection_methods("fake.pdf")
            titles = [h.get("title", h.get("text", "")) for h in headings]
            assert any("APPENDIX" in t for t in titles)


# ── extract_text_with_font_info ───────────────────────────────────────────


class TestExtractTextWithFontInfo:
    def test_basic_extraction(self):
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Hello World",
                                    "size": 12.0,
                                    "flags": 0,
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_doc.load_page.return_value = page

        with (
            patch("codex.ingestion.pdf_extraction.fitz.open", return_value=mock_doc),
            patch("codex.ingestion.pdf_extraction.merge_heading_detection_methods", return_value=[]),
        ):
            pages = extract_text_with_font_info("fake.pdf")
            assert len(pages) == 1
            assert pages[0]["page"] == 1
            assert "Hello World" in pages[0]["text"]

    def test_heading_matched_to_line(self):
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {
                    "lines": [
                        {"spans": [{"text": "Chapter One", "size": 16.0, "flags": 0}]},
                        {"spans": [{"text": "Body text.", "size": 12.0, "flags": 0}]},
                    ]
                }
            ]
        }
        mock_doc.load_page.return_value = page

        detected_headings = [
            {
                "level": 1,
                "title": "Chapter One",
                "page": 0,
                "detection_method": "outline",
                "confidence": 0.9,
            }
        ]

        with (
            patch("codex.ingestion.pdf_extraction.fitz.open", return_value=mock_doc),
            patch(
                "codex.ingestion.pdf_extraction.merge_heading_detection_methods",
                return_value=detected_headings,
            ),
        ):
            pages = extract_text_with_font_info("fake.pdf")
            assert len(pages[0]["headings"]) >= 1
            assert pages[0]["headings"][0]["text"] == "Chapter One"


# ── ingest_document ───────────────────────────────────────────────────────


class TestIngestDocument:
    def test_pdf_pipeline(self):
        mock_pages = [{"page": 1, "text": "Content.", "headings": []}]
        mock_chunks = [
            {
                "page": 1,
                "text": "Content.",
                "original_text": "Content.",
                "headings_path": [],
            }
        ]
        mock_conn = MagicMock()

        with (
            patch("codex.ingestion.ingest.extract_text_with_font_info", return_value=mock_pages),
            patch("codex.ingestion.ingest.merge_heading_detection", return_value=mock_pages),
            patch("codex.ingestion.ingest.chunk_text_with_headings", return_value=mock_chunks),
            patch("codex.ingestion.ingest.db_connection", make_mock_db_connection(mock_conn)),
            patch("codex.ingestion.ingest.store_document", return_value="doc-123"),
            patch("codex.ingestion.ingest.store_passages", return_value=["p-1"]),
            patch("codex.ingestion.ingest.index_in_meilisearch"),
        ):
            doc_id = ingest_document("test.pdf", title="Test")
            assert doc_id == "doc-123"

    def test_txt_pipeline(self):
        mock_pages = [{"page": 1, "text": "Content.", "headings": []}]
        mock_chunks = [
            {
                "page": 1,
                "text": "Content.",
                "original_text": "Content.",
                "headings_path": [],
            }
        ]
        mock_conn = MagicMock()

        with (
            patch("codex.ingestion.ingest.extract_text_from_txt", return_value=mock_pages),
            patch("codex.ingestion.ingest.merge_heading_detection", return_value=mock_pages),
            patch("codex.ingestion.ingest.chunk_text_with_headings", return_value=mock_chunks),
            patch("codex.ingestion.ingest.db_connection", make_mock_db_connection(mock_conn)),
            patch("codex.ingestion.ingest.store_document", return_value="doc-456"),
            patch("codex.ingestion.ingest.store_passages", return_value=["p-1"]),
            patch("codex.ingestion.ingest.index_in_meilisearch"),
        ):
            doc_id = ingest_document("test.txt", title="Test")
            assert doc_id == "doc-456"

    def test_unsupported_file_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_document("test.docx")

    def test_title_defaults_to_filename(self):
        mock_pages = [{"page": 1, "text": "Content.", "headings": []}]
        mock_chunks = [
            {
                "page": 1,
                "text": "Content.",
                "original_text": "Content.",
                "headings_path": [],
            }
        ]
        mock_conn = MagicMock()

        with (
            patch("codex.ingestion.ingest.extract_text_from_txt", return_value=mock_pages),
            patch("codex.ingestion.ingest.merge_heading_detection", return_value=mock_pages),
            patch("codex.ingestion.ingest.chunk_text_with_headings", return_value=mock_chunks),
            patch("codex.ingestion.ingest.db_connection", make_mock_db_connection(mock_conn)),
            patch("codex.ingestion.ingest.store_document", return_value="doc-1") as mock_store,
            patch("codex.ingestion.ingest.store_passages", return_value=["p-1"]),
            patch("codex.ingestion.ingest.index_in_meilisearch"),
        ):
            ingest_document("test.txt")  # no title
            call_args = mock_store.call_args
            assert call_args[0][1] == "test.txt"  # title = basename

    def test_authors_parsed_from_comma_string(self):
        mock_pages = [{"page": 1, "text": "Content.", "headings": []}]
        mock_chunks = [
            {
                "page": 1,
                "text": "Content.",
                "original_text": "Content.",
                "headings_path": [],
            }
        ]
        mock_conn = MagicMock()

        with (
            patch("codex.ingestion.ingest.extract_text_from_txt", return_value=mock_pages),
            patch("codex.ingestion.ingest.merge_heading_detection", return_value=mock_pages),
            patch("codex.ingestion.ingest.chunk_text_with_headings", return_value=mock_chunks),
            patch("codex.ingestion.ingest.db_connection", make_mock_db_connection(mock_conn)),
            patch("codex.ingestion.ingest.store_document", return_value="doc-1") as mock_store,
            patch("codex.ingestion.ingest.store_passages", return_value=["p-1"]),
            patch("codex.ingestion.ingest.index_in_meilisearch"),
        ):
            ingest_document("test.txt", title="Test", authors="Alice, Bob")
            call_args = mock_store.call_args
            assert call_args[0][3] == ["Alice", "Bob"]

    def test_debug_calls_debug_headings(self):
        mock_pages = [{"page": 1, "text": "Content.", "headings": []}]
        mock_chunks = [
            {
                "page": 1,
                "text": "Content.",
                "original_text": "Content.",
                "headings_path": [],
            }
        ]
        mock_conn = MagicMock()

        with (
            patch("codex.ingestion.ingest.extract_text_from_txt", return_value=mock_pages),
            patch("codex.ingestion.ingest.merge_heading_detection", return_value=mock_pages),
            patch("codex.ingestion.ingest.chunk_text_with_headings", return_value=mock_chunks),
            patch("codex.ingestion.ingest.db_connection", make_mock_db_connection(mock_conn)),
            patch("codex.ingestion.ingest.store_document", return_value="doc-1"),
            patch("codex.ingestion.ingest.store_passages", return_value=["p-1"]),
            patch("codex.ingestion.ingest.index_in_meilisearch"),
            patch("codex.ingestion.ingest.debug_headings") as mock_debug,
        ):
            ingest_document("test.txt", title="Test", debug=True)
            mock_debug.assert_called_once()

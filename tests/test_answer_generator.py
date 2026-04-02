"""Tests for answer_generator.py."""

import pytest
from unittest.mock import MagicMock, patch


# ── format_chunks_for_llm ────────────────────────────────────────────────

class TestFormatChunksForLlm:
    def test_empty_list(self):
        from answer_generator import format_chunks_for_llm
        assert format_chunks_for_llm([]) == ""

    def test_single_chunk(self):
        from answer_generator import format_chunks_for_llm
        chunks = [{"text": "Hello world.", "page": 5, "title": "My Doc"}]
        result = format_chunks_for_llm(chunks)
        assert "CHUNK 1" in result
        assert "My Doc" in result
        assert "Page 5" in result
        assert "Hello world." in result

    def test_multiple_chunks_numbered(self):
        from answer_generator import format_chunks_for_llm
        chunks = [
            {"text": "First.", "page": 1, "title": "Doc A"},
            {"text": "Second.", "page": 2, "title": "Doc B"},
        ]
        result = format_chunks_for_llm(chunks)
        assert "CHUNK 1" in result
        assert "CHUNK 2" in result

    def test_missing_title_defaults(self):
        from answer_generator import format_chunks_for_llm
        chunks = [{"text": "Some text.", "page": 1}]
        result = format_chunks_for_llm(chunks)
        assert "Unknown Document" in result

    def test_missing_page_defaults(self):
        from answer_generator import format_chunks_for_llm
        chunks = [{"text": "Some text.", "title": "Doc"}]
        result = format_chunks_for_llm(chunks)
        assert "Unknown" in result


# ── extract_source_pages ──────────────────────────────────────────────────

class TestExtractSourcePages:
    def test_parses_sources_format(self):
        from answer_generator import extract_source_pages
        answer = "The answer is yes. Sources: [History Book, Page 10; Other Book, Page 20]"
        chunks = []
        result = extract_source_pages(answer, chunks)
        assert len(result) == 2
        assert result[0]["title"] == "History Book"
        assert result[0]["page"] == 10
        assert result[1]["title"] == "Other Book"
        assert result[1]["page"] == 20

    def test_fallback_to_chunks_when_no_sources(self):
        from answer_generator import extract_source_pages
        answer = "The answer is yes."
        chunks = [
            {"title": "My Doc", "page": 5},
            {"title": "Other Doc", "page": 10},
        ]
        result = extract_source_pages(answer, chunks)
        assert len(result) == 2
        assert result[0]["page"] == 5

    def test_single_source(self):
        from answer_generator import extract_source_pages
        answer = "Answer. Sources: [Doc Title, Page 42]"
        result = extract_source_pages(answer, [])
        assert len(result) == 1
        assert result[0]["page"] == 42

    def test_malformed_source_skipped(self):
        from answer_generator import extract_source_pages
        answer = "Answer. Sources: [malformed entry without page]"
        chunks = [{"title": "Fallback", "page": 1}]
        result = extract_source_pages(answer, chunks)
        # Malformed entries are skipped, so result may be empty from parsing
        assert isinstance(result, list)

    def test_case_insensitive_sources(self):
        from answer_generator import extract_source_pages
        answer = "Answer. sources: [Doc, Page 5]"
        result = extract_source_pages(answer, [])
        assert len(result) == 1


# ── format_answer_with_sources ────────────────────────────────────────────

class TestFormatAnswerWithSources:
    def test_sources_already_in_answer(self):
        from answer_generator import format_answer_with_sources
        data = {"answer": "Answer text. Sources: [Doc, Page 5]", "sources": []}
        result = format_answer_with_sources(data)
        assert result == data["answer"]

    def test_lowercase_sources_in_answer(self):
        from answer_generator import format_answer_with_sources
        data = {"answer": "Answer. sources: listed here", "sources": []}
        result = format_answer_with_sources(data)
        assert result == data["answer"]

    def test_dict_format_sources_appended(self):
        from answer_generator import format_answer_with_sources
        data = {
            "answer": "The answer is yes.",
            "sources": [{"title": "Doc A", "page": 5}, {"title": "Doc B", "page": 10}]
        }
        result = format_answer_with_sources(data)
        assert "Sources:" in result
        assert "Doc A, Page 5" in result
        assert "Doc B, Page 10" in result

    def test_legacy_int_sources(self):
        from answer_generator import format_answer_with_sources
        data = {"answer": "The answer.", "sources": [5, 10, 3]}
        result = format_answer_with_sources(data)
        assert "Page 3" in result
        assert "Page 5" in result
        assert "Page 10" in result

    def test_empty_sources_no_suffix(self):
        from answer_generator import format_answer_with_sources
        data = {"answer": "The answer.", "sources": []}
        result = format_answer_with_sources(data)
        assert result == "The answer."

    def test_missing_answer_key(self):
        from answer_generator import format_answer_with_sources
        data = {"sources": []}
        result = format_answer_with_sources(data)
        assert result == ""


# ── generate_answer_with_llm ──────────────────────────────────────────────

class TestGenerateAnswerWithLlm:
    def test_empty_chunks_returns_no_answer(self):
        from answer_generator import generate_answer_with_llm
        result = generate_answer_with_llm("What happened?", [])
        assert "cannot" in result["answer"].lower() or "no relevant" in result["answer"].lower()
        assert result["confidence"] == "none"

    def test_successful_call(self):
        from answer_generator import generate_answer_with_llm
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is yes. Sources: [Doc, Page 5]"

        with patch("answer_generator._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = mock_response
            chunks = [{"text": "Test text.", "page": 5, "title": "Doc"}]
            result = generate_answer_with_llm("What?", chunks)
            assert result["answer"] == "The answer is yes. Sources: [Doc, Page 5]"
            assert result["confidence"] == "high"

    def test_api_exception_returns_error(self):
        from answer_generator import generate_answer_with_llm
        with patch("answer_generator._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.side_effect = Exception("API error")
            chunks = [{"text": "Test.", "page": 1, "title": "Doc"}]
            result = generate_answer_with_llm("What?", chunks)
            assert result["confidence"] == "error"
            assert "Error" in result["answer"]


# ── answer_query ──────────────────────────────────────────────────────────

class TestAnswerQuery:
    def test_pipeline_orchestration(self):
        from answer_generator import answer_query
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated answer. Sources: [Doc, Page 1]"

        with patch("answer_generator._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = mock_response
            search_results = [{"text": "Content.", "page": 1, "title": "Doc"}]
            result = answer_query("What?", search_results)
            assert isinstance(result, str)
            assert "Generated answer" in result

    def test_empty_search_results(self):
        from answer_generator import answer_query
        result = answer_query("What?", [])
        assert isinstance(result, str)
        assert "cannot" in result.lower() or "no relevant" in result.lower()

"""Tests for search.py."""

import json
from unittest.mock import MagicMock, patch

from search import format_results
from tests.conftest import make_mock_db_connection

# ── format_results ────────────────────────────────────────────────────────


class TestFormatResults:
    def test_empty_results(self):
        assert format_results([], {}) == ""

    def test_single_result(self):
        results = [{"id": "p1", "text": "Short text.", "page": 5, "title": "Doc A"}]
        scores = {"p1": 0.95}
        output = format_results(results, scores)
        assert "0.950" in output
        assert "Doc A" in output
        assert "p. 5" in output

    def test_text_truncated_at_200_chars(self):
        long_text = "x" * 300
        results = [{"id": "p1", "text": long_text, "page": 1, "title": "Doc"}]
        scores = {"p1": 0.5}
        output = format_results(results, scores)
        assert "..." in output
        assert len(long_text[:200]) == 200

    def test_multi_document_shows_brackets(self):
        results = [
            {"id": "p1", "text": "Text.", "page": 1, "title": "Doc A"},
            {"id": "p2", "text": "Text.", "page": 1, "title": "Doc B"},
        ]
        scores = {"p1": 0.9, "p2": 0.8}
        output = format_results(results, scores)
        assert "[Doc A]" in output
        assert "[Doc B]" in output

    def test_single_document_no_brackets(self):
        results = [
            {"id": "p1", "text": "Text.", "page": 1, "title": "Doc A"},
            {"id": "p2", "text": "Text.", "page": 2, "title": "Doc A"},
        ]
        scores = {"p1": 0.9, "p2": 0.8}
        output = format_results(results, scores)
        assert "[Doc A]" not in output
        assert "Doc A" in output

    def test_missing_score_defaults_to_zero(self):
        results = [{"id": "p1", "text": "Text.", "page": 1, "title": "Doc"}]
        scores = {}  # no score for p1
        output = format_results(results, scores)
        assert "0.000" in output


# ── analyze_query ─────────────────────────────────────────────────────────


class TestAnalyzeQuery:
    def test_successful_analysis(self, mock_openai_client):
        from search import analyze_query

        with patch("search.get_openai_client", return_value=mock_openai_client):
            result = analyze_query("Who founded Cambridge?")
            assert "query_type" in result
            assert "expansions" in result

    def test_api_error_returns_defaults(self):
        from search import analyze_query

        with patch("search.get_openai_client", side_effect=Exception("API down")):
            result = analyze_query("test query")
            assert result["query_type"] == "general"
            assert result["person"] is None
            assert "test query" in result["expansions"]

    def test_person_normalization_empty_string(self, mock_openai_client):
        from search import analyze_query

        # Set person to empty string in response
        response_data = json.loads(
            mock_openai_client.chat.completions.create.return_value.choices[
                0
            ].message.content
        )
        response_data["person"] = ""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(response_data)
        with patch("search.get_openai_client", return_value=mock_openai_client):
            result = analyze_query("What happened?")
            assert result["person"] is None

    def test_person_normalization_none_string(self, mock_openai_client):
        from search import analyze_query

        response_data = json.loads(
            mock_openai_client.chat.completions.create.return_value.choices[
                0
            ].message.content
        )
        response_data["person"] = "none"
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(response_data)
        with patch("search.get_openai_client", return_value=mock_openai_client):
            result = analyze_query("What?")
            assert result["person"] is None

    def test_original_query_prepended_to_expansions(self, mock_openai_client):
        from search import analyze_query

        with patch("search.get_openai_client", return_value=mock_openai_client):
            result = analyze_query("Who was George Washington?")
            assert result["expansions"][0] == "Who was George Washington?"

    def test_expansions_limited_to_15(self, mock_openai_client):
        from search import analyze_query

        response_data = json.loads(
            mock_openai_client.chat.completions.create.return_value.choices[
                0
            ].message.content
        )
        response_data["expansions"] = [f"term{i}" for i in range(20)]
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(response_data)
        with patch("search.get_openai_client", return_value=mock_openai_client):
            result = analyze_query("test")
            assert len(result["expansions"]) <= 15


# ── hybrid_search ─────────────────────────────────────────────────────────


class TestHybridSearch:
    def test_returns_id_score_tuples(self, mock_meili_client):
        from search import hybrid_search

        with patch("search.get_meili_client", return_value=mock_meili_client):
            results = hybrid_search("test query")
            assert len(results) == 2
            assert results[0] == ("passage-1", 0.95)

    def test_with_document_id_filter(self, mock_meili_client):
        from search import hybrid_search

        with patch("search.get_meili_client", return_value=mock_meili_client):
            test_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            hybrid_search("test", document_id=test_uuid)
            mock_meili_client.index("passages").search.assert_called()
            call_args = mock_meili_client.index("passages").search.call_args
            search_params = (
                call_args[0][1]
                if len(call_args[0]) > 1
                else call_args[1].get("opt_params", {})
            )
            assert "filter" in search_params, "Expected 'filter' key in search params"
            assert test_uuid in search_params["filter"], (
                "Expected document_id in filter string"
            )

    def test_error_returns_empty_list(self):
        from search import hybrid_search

        with patch(
            "search.get_meili_client", side_effect=Exception("Connection refused")
        ):
            results = hybrid_search("test")
            assert results == []


# ── rerank_candidates ─────────────────────────────────────────────────────


class TestRerankCandidates:
    def test_empty_candidates_returns_empty(self, mock_db_conn):
        from search import rerank_candidates

        result = rerank_candidates([], "query", mock_db_conn)
        assert result == []

    def test_reranks_and_returns_top_k(self, mock_db_conn):
        from search import rerank_candidates

        mock_cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = [
            {"id": "p1", "text": "First passage."},
            {"id": "p2", "text": "Second passage."},
        ]

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.3]

        with patch("search._get_reranker", return_value=mock_reranker):
            candidates = [("p1", 0.5), ("p2", 0.5)]
            result = rerank_candidates(candidates, "query", mock_db_conn, k=1)
            assert len(result) == 1
            assert result[0][0] == "p1"  # highest score


# ── get_passage_details ───────────────────────────────────────────────────


class TestGetPassageDetails:
    def test_empty_list(self, mock_db_conn):
        from search import get_passage_details

        result = get_passage_details(mock_db_conn, [])
        assert result == []

    def test_preserves_order(self, mock_db_conn):
        from search import get_passage_details

        mock_cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = [
            {
                "id": "p2",
                "text": "Second",
                "page": 2,
                "title": "Doc",
                "headings_path": [],
            },
            {
                "id": "p1",
                "text": "First",
                "page": 1,
                "title": "Doc",
                "headings_path": [],
            },
        ]
        result = get_passage_details(mock_db_conn, ["p1", "p2"])
        assert result[0]["id"] == "p1"
        assert result[1]["id"] == "p2"


# ── search_codex ──────────────────────────────────────────────────────────


class TestSearchCodex:
    def test_no_candidates_returns_empty(self):
        from search import search_codex

        with (
            patch(
                "search.analyze_query",
                return_value={"query_type": "general", "expansions": ["q"]},
            ),
            patch("search.hybrid_search", return_value=[]),
        ):
            result = search_codex("test query")
            assert result["results"] == []
            assert "No candidates" in result["answer"]

    def test_full_pipeline(self, mock_db_conn):
        from search import search_codex

        with (
            patch(
                "search.analyze_query",
                return_value={"query_type": "factoid", "expansions": ["q"]},
            ),
            patch("search.hybrid_search", return_value=[("p1", 0.9)]),
            patch("search.db_connection", make_mock_db_connection(mock_db_conn)),
            patch("search.rerank_candidates", return_value=[("p1", 0.95)]),
            patch(
                "search.get_passage_details",
                return_value=[
                    {
                        "id": "p1",
                        "text": "Answer text.",
                        "page": 1,
                        "title": "Doc",
                        "headings_path": [],
                    }
                ],
            ),
            patch("search.generate_answer", return_value="The answer."),
        ):
            result = search_codex("What happened?")
            assert result["query_type"] == "factoid"
            assert result["answer"] == "The answer."
            assert len(result["results"]) == 1
            assert result["results"][0]["score"] == 0.95

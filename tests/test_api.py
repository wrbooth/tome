"""Tests for api/main.py endpoints."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    """FastAPI test client with mocked dependencies."""
    with patch("api.main.get_db_connection") as mock_conn, \
         patch("api.main.search_codex") as mock_search:
        from api.main import app
        from fastapi.testclient import TestClient
        yield TestClient(app), mock_conn, mock_search


# ── GET /health ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_healthy(self):
        from api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# ── POST /search ──────────────────────────────────────────────────────────

class TestSearchEndpoint:
    def test_successful_search(self):
        from api.main import app
        from fastapi.testclient import TestClient
        mock_result = {
            "query_type": "factoid",
            "answer": "The answer is yes.",
            "results": [
                {"id": "p1", "text": "Short text.", "page": 5, "title": "Doc A",
                 "headings_path": [], "score": 0.95}
            ]
        }
        with patch("api.main.search_codex", return_value=mock_result):
            client = TestClient(app)
            response = client.post("/search", json={"query": "What happened?"})
            assert response.status_code == 200
            data = response.json()
            assert data["query_type"] == "factoid"
            assert data["answer"] == "The answer is yes."
            assert len(data["results"]) == 1
            assert data["results"][0]["passage_id"] == "p1"

    def test_snippet_truncation(self):
        from api.main import app
        from fastapi.testclient import TestClient
        long_text = "x" * 300
        mock_result = {
            "query_type": "general",
            "answer": "Answer.",
            "results": [
                {"id": "p1", "text": long_text, "page": 1, "title": "Doc",
                 "headings_path": [], "score": 0.5}
            ]
        }
        with patch("api.main.search_codex", return_value=mock_result):
            client = TestClient(app)
            response = client.post("/search", json={"query": "test"})
            snippet = response.json()["results"][0]["snippet"]
            assert snippet.endswith("...")
            assert len(snippet) == 203  # 200 + "..."

    def test_search_error_returns_500(self):
        from api.main import app
        from fastapi.testclient import TestClient
        with patch("api.main.search_codex", side_effect=Exception("Search failed")):
            client = TestClient(app)
            response = client.post("/search", json={"query": "test"})
            assert response.status_code == 500


# ── GET /documents ────────────────────────────────────────────────────────

class TestDocumentsEndpoint:
    def test_list_documents(self):
        from api.main import app
        from fastapi.testclient import TestClient
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {"id": "d1", "title": "Doc A", "authors": None, "pub_year": 2000, "source_path": "/path"},
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("api.main.get_db_connection", return_value=mock_conn):
            client = TestClient(app)
            response = client.get("/documents")
            assert response.status_code == 200
            assert len(response.json()) == 1

    def test_documents_error_returns_500(self):
        from api.main import app
        from fastapi.testclient import TestClient
        with patch("api.main.get_db_connection", side_effect=Exception("DB down")):
            client = TestClient(app)
            response = client.get("/documents")
            assert response.status_code == 500


# ── GET /stats ────────────────────────────────────────────────────────────

class TestStatsEndpoint:
    def test_returns_stats(self):
        from api.main import app
        from fastapi.testclient import TestClient
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.side_effect = [
            (10,),   # doc count
            (100,),  # passage count
            (80,),   # embedded count
            (500,),  # entity count
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("api.main.get_db_connection", return_value=mock_conn):
            client = TestClient(app)
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["documents"] == 10
            assert data["passages"] == 100
            assert data["embedded_passages"] == 80
            assert data["embedding_coverage"] == "80.0%"

    def test_zero_passages_coverage(self):
        from api.main import app
        from fastapi.testclient import TestClient
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.side_effect = [
            (0,),  # doc count
            (0,),  # passage count
            (0,),  # embedded count
            (0,),  # entity count
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("api.main.get_db_connection", return_value=mock_conn):
            client = TestClient(app)
            response = client.get("/stats")
            assert response.status_code == 200
            assert response.json()["embedding_coverage"] == "0%"

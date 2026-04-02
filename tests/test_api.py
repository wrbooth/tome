"""Tests for api/main.py endpoints."""

import io
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from tests.conftest import make_mock_db_connection


def _make_search_result(**overrides):
    """Helper to build a mock search_codex return value."""
    defaults = {
        "query_type": "factoid",
        "query_analysis": {
            "query_type": "factoid",
            "person": None,
            "entities": {
                "persons": [], "places": [], "events": [], "dates": [],
                "families": [], "companies": [], "industries": [], "settlement_terms": [],
            },
            "expansions": ["test query"],
        },
        "answer": "The answer is yes.",
        "results": [
            {"id": "p1", "text": "Short text.", "page": 5, "title": "Doc A",
             "headings_path": ["Chapter 1"], "score": 0.95}
        ],
    }
    defaults.update(overrides)
    return defaults


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
        mock_result = _make_search_result()
        with patch("api.main.search_codex", return_value=mock_result):
            client = TestClient(app)
            response = client.post("/search", json={"query": "What happened?"})
            assert response.status_code == 200
            data = response.json()
            assert data["query_type"] == "factoid"
            assert data["answer"] == "The answer is yes."
            assert len(data["results"]) == 1
            assert data["results"][0]["passage_id"] == "p1"

    def test_search_returns_full_text_and_headings(self):
        from api.main import app
        from fastapi.testclient import TestClient
        mock_result = _make_search_result()
        with patch("api.main.search_codex", return_value=mock_result):
            client = TestClient(app)
            response = client.post("/search", json={"query": "test"})
            result = response.json()["results"][0]
            assert result["text"] == "Short text."
            assert result["headings_path"] == ["Chapter 1"]

    def test_search_returns_query_analysis(self):
        from api.main import app
        from fastapi.testclient import TestClient
        mock_result = _make_search_result()
        mock_result["query_analysis"]["person"] = "John Smith"
        mock_result["query_analysis"]["expansions"] = ["who is John Smith", "John Smith history"]
        with patch("api.main.search_codex", return_value=mock_result):
            client = TestClient(app)
            response = client.post("/search", json={"query": "Who is John Smith?"})
            data = response.json()
            qa = data["query_analysis"]
            assert qa["query_type"] == "factoid"
            assert qa["person"] == "John Smith"
            assert "who is John Smith" in qa["expansions"]

    def test_snippet_truncation(self):
        from api.main import app
        from fastapi.testclient import TestClient
        long_text = "x" * 300
        mock_result = _make_search_result(
            results=[{"id": "p1", "text": long_text, "page": 1, "title": "Doc",
                      "headings_path": [], "score": 0.5}]
        )
        with patch("api.main.search_codex", return_value=mock_result):
            client = TestClient(app)
            response = client.post("/search", json={"query": "test"})
            result = response.json()["results"][0]
            assert result["snippet"].endswith("...")
            assert len(result["snippet"]) == 203  # 200 + "..."
            assert result["text"] == long_text  # full text preserved

    def test_search_error_returns_500(self):
        from api.main import app
        from fastapi.testclient import TestClient
        with patch("api.main.search_codex", side_effect=Exception("Search failed")):
            client = TestClient(app)
            response = client.post("/search", json={"query": "test"})
            assert response.status_code == 500

    def test_search_with_missing_query_analysis(self):
        """Backwards compat: search_codex result without query_analysis key."""
        from api.main import app
        from fastapi.testclient import TestClient
        mock_result = {
            "query_type": "general",
            "answer": "Answer.",
            "results": [
                {"id": "p1", "text": "Text.", "page": 1, "title": "Doc",
                 "headings_path": [], "score": 0.5}
            ],
        }
        with patch("api.main.search_codex", return_value=mock_result):
            client = TestClient(app)
            response = client.post("/search", json={"query": "test"})
            assert response.status_code == 200
            assert response.json()["query_analysis"]["query_type"] == "general"


# ── POST /search/stream ──────────────────────────────────────────────────

class TestSearchStreamEndpoint:
    def test_stream_returns_sse_events(self):
        from api.main import app
        from fastapi.testclient import TestClient

        mock_analysis = {
            "query_type": "general",
            "person": None,
            "entities": {"persons": [], "places": [], "events": [], "dates": [],
                         "families": [], "companies": [], "industries": [], "settlement_terms": []},
            "expansions": ["test"],
        }
        mock_details = [
            {"id": "p1", "text": "Passage text.", "page": 1, "title": "Doc A",
             "headings_path": [], "score": 0.9}
        ]

        with patch("api.main.analyze_query", return_value=mock_analysis), \
             patch("api.main.hybrid_search", return_value=[("p1", 0.9)]), \
             patch("api.main.rerank_candidates", return_value=[("p1", 0.9)]), \
             patch("api.main.get_passage_details", return_value=mock_details), \
             patch("api.main.stream_answer_with_llm", return_value=iter(["Hello", " world"])), \
             patch("api.main.db_connection", make_mock_db_connection(MagicMock())):

            client = TestClient(app)
            response = client.post(
                "/search/stream",
                json={"query": "test"},
                headers={"Accept": "text/event-stream"},
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            # Parse SSE events from response body
            events = _parse_sse(response.text)
            event_types = [e["event"] for e in events if e.get("event")]
            assert "search_results" in event_types
            assert "token" in event_types
            assert "done" in event_types

    def test_stream_no_candidates(self):
        from api.main import app
        from fastapi.testclient import TestClient

        mock_analysis = {
            "query_type": "general", "person": None,
            "entities": {"persons": [], "places": [], "events": [], "dates": [],
                         "families": [], "companies": [], "industries": [], "settlement_terms": []},
            "expansions": [],
        }

        with patch("api.main.analyze_query", return_value=mock_analysis), \
             patch("api.main.hybrid_search", return_value=[]):

            client = TestClient(app)
            response = client.post(
                "/search/stream",
                json={"query": "nonexistent"},
                headers={"Accept": "text/event-stream"},
            )
            assert response.status_code == 200
            events = _parse_sse(response.text)
            event_types = [e["event"] for e in events if e.get("event")]
            assert "search_results" in event_types
            assert "done" in event_types


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

        with patch("api.main.db_connection", make_mock_db_connection(mock_conn)):
            client = TestClient(app)
            response = client.get("/documents")
            assert response.status_code == 200
            assert len(response.json()) == 1

    def test_documents_error_returns_500(self):
        from api.main import app
        from fastapi.testclient import TestClient
        with patch("api.main.db_connection", side_effect=Exception("DB down")):
            client = TestClient(app)
            response = client.get("/documents")
            assert response.status_code == 500


# ── GET /documents/{document_id} ─────────────────────────────────────────

class TestDocumentDetailEndpoint:
    def test_returns_document_stats(self):
        from api.main import app
        from fastapi.testclient import TestClient
        mock_stats = {
            "document": {"id": "d1", "title": "Doc A", "authors": None, "pub_year": 2000, "source_path": "/path"},
            "passages": {"total_passages": 50, "embedded_passages": 45},
            "entities": {"entity_count": 120},
            "years": {"year_count": 15},
            "pages": {"min_page": 1, "max_page": 80},
        }
        with patch("api.main.db_connection", make_mock_db_connection(MagicMock())), \
             patch("api.main.get_document_stats", return_value=mock_stats):
            client = TestClient(app)
            response = client.get("/documents/d1")
            assert response.status_code == 200
            data = response.json()
            assert data["document"]["title"] == "Doc A"
            assert data["passages"]["total_passages"] == 50
            assert data["pages"]["max_page"] == 80

    def test_document_not_found(self):
        from api.main import app
        from fastapi.testclient import TestClient
        with patch("api.main.db_connection", make_mock_db_connection(MagicMock())), \
             patch("api.main.get_document_stats", return_value=None):
            client = TestClient(app)
            response = client.get("/documents/nonexistent")
            assert response.status_code == 404

    def test_document_detail_db_error(self):
        from api.main import app
        from fastapi.testclient import TestClient
        with patch("api.main.db_connection", side_effect=Exception("DB error")):
            client = TestClient(app)
            response = client.get("/documents/d1")
            assert response.status_code == 500


# ── POST /documents/upload ───────────────────────────────────────────────

class TestUploadEndpoint:
    def test_upload_pdf_returns_202(self):
        from api.main import app, _ingest_tasks
        from fastapi.testclient import TestClient

        _ingest_tasks.clear()
        client = TestClient(app)
        file_content = b"%PDF-1.4 fake content"
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            data={"title": "Test Doc", "authors": "Author A", "pub_year": "2020"},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"
        assert data["filename"] == "test.pdf"
        assert data["task_id"]

    def test_upload_txt_accepted(self):
        from api.main import app, _ingest_tasks
        from fastapi.testclient import TestClient

        _ingest_tasks.clear()
        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", io.BytesIO(b"Some text"), "text/plain")},
            data={"title": "Notes"},
        )
        assert response.status_code == 202

    def test_upload_rejects_unsupported_type(self):
        from api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("image.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert response.status_code == 400
        assert "PDF and TXT" in response.json()["detail"]


# ── GET /documents/upload/{task_id} ──────────────────────────────────────

class TestUploadStatusEndpoint:
    def test_returns_task_status(self):
        from api.main import app, _ingest_tasks
        from fastapi.testclient import TestClient

        _ingest_tasks.clear()
        _ingest_tasks["task-123"] = {
            "task_id": "task-123",
            "status": "running",
            "document_id": None,
            "filename": "doc.pdf",
            "message": "Ingesting...",
            "created_at": "2026-04-02T00:00:00+00:00",
        }
        client = TestClient(app)
        response = client.get("/documents/upload/task-123")
        assert response.status_code == 200
        assert response.json()["status"] == "running"

    def test_task_not_found(self):
        from api.main import app, _ingest_tasks
        from fastapi.testclient import TestClient

        _ingest_tasks.clear()
        client = TestClient(app)
        response = client.get("/documents/upload/nonexistent")
        assert response.status_code == 404


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
            (50,),   # year count
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("api.main.db_connection", make_mock_db_connection(mock_conn)):
            client = TestClient(app)
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["documents"] == 10
            assert data["passages"] == 100
            assert data["embedded_passages"] == 80
            assert data["embedding_coverage"] == "80.0%"
            assert data["years"] == 50

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
            (0,),  # year count
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("api.main.db_connection", make_mock_db_connection(mock_conn)):
            client = TestClient(app)
            response = client.get("/stats")
            assert response.status_code == 200
            assert response.json()["embedding_coverage"] == "0%"


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_sse(text: str) -> list:
    """Parse raw SSE text into a list of event dicts."""
    events = []
    current = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            raw = line[len("data:"):].strip()
            try:
                current["data"] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                current["data"] = raw
    if current:
        events.append(current)
    return events

"""Tests for batch_reingest.py."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tests.unit.conftest import make_mock_db_connection
from tome.ingestion.batch_reingest import (
    clear_database,
    clear_documents,
    clear_meilisearch,
    get_document_info,
    list_all_documents,
    main,
    reingest_document,
)

# ── clear_database ────────────────────────────────────────────────────────


class TestClearDatabase:
    def test_deletes_passages_then_documents(self):
        mock_conn = MagicMock()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 10
        with patch(
            "tome.ingestion.batch_reingest.db_connection",
            make_mock_db_connection(mock_conn),
        ):
            clear_database()
            calls = cursor.execute.call_args_list
            # First delete passages, then documents
            assert "DELETE FROM passages" in calls[0][0][0]
            assert "DELETE FROM documents" in calls[1][0][0]

    def test_rollback_on_error(self):
        mock_conn = MagicMock()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = Exception("DB error")
        with (
            patch(
                "tome.ingestion.batch_reingest.db_connection",
                make_mock_db_connection(mock_conn),
            ),
            pytest.raises(Exception, match="DB error"),
        ):
            clear_database()


# ── clear_documents ───────────────────────────────────────────────────────


class TestClearDocuments:
    def test_deletes_specific_documents(self):
        mock_conn = MagicMock()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1
        with patch(
            "tome.ingestion.batch_reingest.db_connection",
            make_mock_db_connection(mock_conn),
        ):
            clear_documents(["doc-1", "doc-2"])
            # 2 docs x 2 deletes each (passages + document)
            assert cursor.execute.call_count == 4

    def test_rollback_on_error(self):
        mock_conn = MagicMock()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = Exception("DB error")
        with (
            patch(
                "tome.ingestion.batch_reingest.db_connection",
                make_mock_db_connection(mock_conn),
            ),
            pytest.raises(Exception, match="DB error"),
        ):
            clear_documents(["doc-1"])

    def test_empty_list(self):
        mock_conn = MagicMock()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        with patch(
            "tome.ingestion.batch_reingest.db_connection",
            make_mock_db_connection(mock_conn),
        ):
            clear_documents([])
            cursor.execute.assert_not_called()


# ── clear_meilisearch ─────────────────────────────────────────────────────


class TestClearMeilisearch:
    def test_deletes_index(self, mock_meili_client):
        with patch(
            "tome.ingestion.batch_reingest.get_meili_client",
            return_value=mock_meili_client,
        ):
            clear_meilisearch()
            mock_meili_client.index("passages").delete.assert_called_once()

    def test_handles_not_found(self, mock_meili_client, caplog):
        mock_meili_client.index("passages").delete.side_effect = Exception(
            "index not found"
        )
        with patch(
            "tome.ingestion.batch_reingest.get_meili_client",
            return_value=mock_meili_client,
        ):
            with caplog.at_level("INFO", logger="tome.ingestion.batch_reingest"):
                clear_meilisearch()
            assert "already empty" in caplog.text

    def test_handles_other_delete_error(self, mock_meili_client, caplog):
        mock_meili_client.index("passages").delete.side_effect = Exception(
            "timeout error"
        )
        with patch(
            "tome.ingestion.batch_reingest.get_meili_client",
            return_value=mock_meili_client,
        ):
            with caplog.at_level("ERROR", logger="tome.ingestion.batch_reingest"):
                clear_meilisearch()
            assert "Error deleting" in caplog.text

    def test_handles_connection_error(self, caplog):
        with patch(
            "tome.ingestion.batch_reingest.get_meili_client",
            side_effect=Exception("Connection refused"),
        ):
            with caplog.at_level("ERROR", logger="tome.ingestion.batch_reingest"):
                clear_meilisearch()
            assert "Error connecting" in caplog.text


# ── get_document_info ─────────────────────────────────────────────────────


class TestGetDocumentInfo:
    def test_returns_document(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {
            "id": "doc-1",
            "title": "Test",
            "source_path": "/path",
            "authors": None,
            "pub_year": 2000,
        }
        result = get_document_info(mock_db_conn, "doc-1")
        assert result["title"] == "Test"

    def test_returns_none_when_not_found(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = None
        result = get_document_info(mock_db_conn, "nonexistent")
        assert result is None


# ── list_all_documents ────────────────────────────────────────────────────


class TestListAllDocuments:
    def test_returns_documents(self):
        mock_conn = MagicMock()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            {
                "id": "d1",
                "title": "Doc A",
                "source_path": "/a",
                "authors": None,
                "pub_year": 2000,
            },
            {
                "id": "d2",
                "title": "Doc B",
                "source_path": "/b",
                "authors": ["X"],
                "pub_year": 2001,
            },
        ]
        with patch(
            "tome.ingestion.batch_reingest.db_connection",
            make_mock_db_connection(mock_conn),
        ):
            result = list_all_documents()
            assert len(result) == 2

    def test_empty_database(self):
        mock_conn = MagicMock()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        with patch(
            "tome.ingestion.batch_reingest.db_connection",
            make_mock_db_connection(mock_conn),
        ):
            result = list_all_documents()
            assert result == []


# ── reingest_document ─────────────────────────────────────────────────────


class TestReingestDocument:
    def test_success(self):
        with patch("tome.ingestion.ingest.ingest_document") as mock_ingest:
            result = reingest_document("/path/doc.pdf", "Doc Title")
            assert result is True
            mock_ingest.assert_called_once_with(
                "/path/doc.pdf", "Doc Title", None, None, False
            )

    def test_with_metadata(self):
        with patch("tome.ingestion.ingest.ingest_document") as mock_ingest:
            result = reingest_document(
                "/path/doc.pdf", "Doc", authors="Alice", pub_year=2000, debug=True
            )
            assert result is True
            mock_ingest.assert_called_once_with(
                "/path/doc.pdf", "Doc", "Alice", 2000, True
            )

    def test_failure(self):
        with patch(
            "tome.ingestion.ingest.ingest_document",
            side_effect=Exception("Ingest failed"),
        ):
            result = reingest_document("/path/doc.pdf", "Doc")
            assert result is False


# ── CLI main ──────────────────────────────────────────────────────────────


class TestMainCli:
    def test_all_flag_no_documents(self, caplog):
        runner = CliRunner()
        with (
            caplog.at_level("INFO", logger="tome.ingestion.batch_reingest"),
            patch("tome.ingestion.batch_reingest.list_all_documents", return_value=[]),
        ):
            runner.invoke(main, ["--all"])
        assert "No documents found" in caplog.text

    def test_all_flag_with_documents(self, caplog):
        docs = [
            {
                "id": "d1",
                "title": "Doc A",
                "source_path": "/tmp/test.pdf",
                "authors": None,
                "pub_year": 2000,
            },
        ]
        runner = CliRunner()
        with (
            caplog.at_level("INFO", logger="tome.ingestion.batch_reingest"),
            patch(
                "tome.ingestion.batch_reingest.list_all_documents", return_value=docs
            ),
            patch("tome.ingestion.batch_reingest.reingest_document", return_value=True),
            patch("pathlib.Path.exists", return_value=True),
        ):
            runner.invoke(main, ["--all"])
        assert "Successful: 1" in caplog.text

    def test_specific_documents(self, caplog):
        runner = CliRunner()
        mock_conn = MagicMock()
        doc_info = {
            "id": "d1",
            "title": "Doc",
            "source_path": "/tmp/test.pdf",
            "authors": ["A"],
            "pub_year": 2000,
        }
        with (
            caplog.at_level("INFO", logger="tome.ingestion.batch_reingest"),
            patch(
                "tome.ingestion.batch_reingest.db_connection",
                make_mock_db_connection(mock_conn),
            ),
            patch(
                "tome.ingestion.batch_reingest.get_document_info", return_value=doc_info
            ),
            patch("tome.ingestion.batch_reingest.reingest_document", return_value=True),
            patch("pathlib.Path.exists", return_value=True),
        ):
            runner.invoke(main, ["d1"])
        assert "Successful: 1" in caplog.text

    def test_specific_document_not_found(self, caplog):
        runner = CliRunner()
        mock_conn = MagicMock()
        with (
            caplog.at_level("WARNING", logger="tome.ingestion.batch_reingest"),
            patch(
                "tome.ingestion.batch_reingest.db_connection",
                make_mock_db_connection(mock_conn),
            ),
            patch("tome.ingestion.batch_reingest.get_document_info", return_value=None),
        ):
            runner.invoke(main, ["nonexistent"])
        assert "not found" in caplog.text

    def test_no_documents_specified(self, caplog):
        runner = CliRunner()
        with caplog.at_level("ERROR", logger="tome.ingestion.batch_reingest"):
            runner.invoke(main, [])
        assert "Must specify" in caplog.text

    def test_clear_first_flag(self):
        docs = [
            {
                "id": "d1",
                "title": "Doc",
                "source_path": "/tmp/test.pdf",
                "authors": None,
                "pub_year": None,
            }
        ]
        runner = CliRunner()
        with (
            patch(
                "tome.ingestion.batch_reingest.list_all_documents", return_value=docs
            ),
            patch("tome.ingestion.batch_reingest.clear_database") as mock_clear_db,
            patch("tome.ingestion.batch_reingest.clear_meilisearch") as mock_clear_ms,
            patch("tome.ingestion.batch_reingest.reingest_document", return_value=True),
            patch("pathlib.Path.exists", return_value=True),
        ):
            runner.invoke(main, ["--all", "--clear-first"])
            mock_clear_db.assert_called_once()
            mock_clear_ms.assert_called_once()

    def test_clear_docs_flag(self):
        docs = [
            {
                "id": "d1",
                "title": "Doc",
                "source_path": "/tmp/test.pdf",
                "authors": None,
                "pub_year": None,
            }
        ]
        runner = CliRunner()
        with (
            patch(
                "tome.ingestion.batch_reingest.list_all_documents", return_value=docs
            ),
            patch("tome.ingestion.batch_reingest.clear_documents") as mock_clear,
            patch("tome.ingestion.batch_reingest.reingest_document", return_value=True),
            patch("pathlib.Path.exists", return_value=True),
        ):
            runner.invoke(main, ["--all", "--clear-docs", "d1,d2"])
            mock_clear.assert_called_once_with(["d1", "d2"])

    def test_source_file_not_found(self, caplog):
        docs = [
            {
                "id": "d1",
                "title": "Doc",
                "source_path": "/nonexistent/file.pdf",
                "authors": None,
                "pub_year": None,
            }
        ]
        runner = CliRunner()
        with (
            caplog.at_level("INFO", logger="tome.ingestion.batch_reingest"),
            patch(
                "tome.ingestion.batch_reingest.list_all_documents", return_value=docs
            ),
            patch("pathlib.Path.exists", return_value=False),
        ):
            runner.invoke(main, ["--all"])
        assert "Source file not found" in caplog.text
        assert "Failed: 1" in caplog.text

    def test_reingest_failure_tracked(self, caplog):
        docs = [
            {
                "id": "d1",
                "title": "Doc",
                "source_path": "/tmp/test.pdf",
                "authors": None,
                "pub_year": None,
            }
        ]
        runner = CliRunner()
        with (
            caplog.at_level("INFO", logger="tome.ingestion.batch_reingest"),
            patch(
                "tome.ingestion.batch_reingest.list_all_documents", return_value=docs
            ),
            patch(
                "tome.ingestion.batch_reingest.reingest_document", return_value=False
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = runner.invoke(main, ["--all"])
        assert "Failed: 1" in caplog.text
        assert result.exit_code == 1

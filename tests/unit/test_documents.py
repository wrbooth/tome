"""Tests for documents.py."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tests.unit.conftest import make_mock_db_connection
from tome.documents import (
    cli,
    delete_document,
    get_document_stats,
    list_documents,
    reindex_document_meilisearch,
)

# ── get_document_stats ────────────────────────────────────────────────────


class TestGetDocumentStats:
    def test_document_not_found(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = None
        result = get_document_stats(mock_db_conn, "nonexistent-id")
        assert result is None

    def test_returns_nested_stats(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            # doc query
            {
                "id": "doc-1",
                "title": "Test Doc",
                "authors": ["Author"],
                "pub_year": 2000,
                "source_path": "/path",
            },
            # passage stats
            {"total_passages": 10},
            # entity stats
            {"entity_count": 50},
            # year stats
            {"year_count": 15},
            # page stats
            {"min_page": 1, "max_page": 20},
        ]
        result = get_document_stats(mock_db_conn, "doc-1")
        assert result is not None
        assert "document" in result
        assert "passages" in result
        assert "entities" in result
        assert "years" in result
        assert "pages" in result
        assert result["passages"]["total_passages"] == 10

    def test_queries_correct_document_id(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = None
        get_document_stats(mock_db_conn, "specific-id")
        first_call = cursor.execute.call_args_list[0]
        assert "specific-id" in first_call[0][1]


# ── list_documents ────────────────────────────────────────────────────────


class TestListDocuments:
    def test_returns_list_of_dicts(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            {
                "id": "d1",
                "title": "Doc A",
                "authors": None,
                "pub_year": 2000,
                "passage_count": 5,
                "min_page": 1,
                "max_page": 10,
            },
        ]
        result = list_documents(mock_db_conn)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Doc A"

    def test_empty_database(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        result = list_documents(mock_db_conn)
        assert result == []


# ── delete_document ───────────────────────────────────────────────────────


class TestDeleteDocument:
    def test_confirm_false_deletes_directly(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1
        result = delete_document(mock_db_conn, "doc-1", confirm=False)
        assert result is True
        mock_db_conn.commit.assert_called_once()

    def test_exception_causes_rollback(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = Exception("DB error")
        result = delete_document(mock_db_conn, "doc-1", confirm=False)
        assert result is False
        mock_db_conn.rollback.assert_called_once()

    def test_confirm_true_not_found(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = None
        result = delete_document(mock_db_conn, "nonexistent", confirm=True)
        assert result is False

    def test_confirm_true_user_cancels(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            {"title": "Doc"},  # doc info
            (5,),  # passage count (fetchone()[0])
        ]
        with patch("tome.documents.click.confirm", return_value=False):
            result = delete_document(mock_db_conn, "doc-1", confirm=True)
            assert result is False

    def test_confirm_true_user_accepts(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        # First two fetchone calls for confirm, then execute calls for delete
        cursor.fetchone.side_effect = [
            {"title": "Doc"},  # doc info
            (5,),  # passage count
        ]
        cursor.rowcount = 1
        with patch("tome.documents.click.confirm", return_value=True):
            result = delete_document(mock_db_conn, "doc-1", confirm=True)
            assert result is True


# ── reindex_document_meilisearch ──────────────────────────────────────────


class TestReindexDocumentMeilisearch:
    def test_no_passages_returns_false(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        result = reindex_document_meilisearch(mock_db_conn, "doc-1")
        assert result is False

    def test_successful_reindex(self, mock_db_conn, mock_meili_client):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            {
                "id": "p1",
                "text": "Test text from 1865.",
                "page": 1,
                "headings_path": ["Chapter 1"],
                "title": "Doc",
            },
        ]
        with (
            patch("tome.documents.get_meili_client", return_value=mock_meili_client),
            patch(
                "tome.ingestion.ingest.extract_entities_and_years",
                return_value=([], [1865]),
            ),
        ):
            result = reindex_document_meilisearch(mock_db_conn, "doc-1")
            assert result is True
            mock_meili_client.index("passages").add_documents.assert_called_once()

    def test_meilisearch_error_returns_false(self, mock_db_conn):
        cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            {
                "id": "p1",
                "text": "Test.",
                "page": 1,
                "headings_path": [],
                "title": "Doc",
            },
        ]
        with (
            patch(
                "tome.documents.get_meili_client",
                side_effect=Exception("Connection error"),
            ),
            patch(
                "tome.ingestion.ingest.extract_entities_and_years",
                return_value=([], []),
            ),
        ):
            result = reindex_document_meilisearch(mock_db_conn, "doc-1")
            assert result is False


# ── CLI: list command ─────────────────────────────────────────────────────


class TestListCliCommand:
    def _mock_conn_with_docs(self, docs):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = docs
        mock_conn.cursor.return_value = cursor
        return mock_conn

    def test_list_json_format(self):
        docs = [
            {
                "id": "d1",
                "title": "Doc A",
                "authors": None,
                "pub_year": 2000,
                "passage_count": 5,
                "min_page": 1,
                "max_page": 10,
            }
        ]
        mock_conn = self._mock_conn_with_docs(docs)
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["list", "--format", "json"])
            assert result.exit_code == 0
            assert "Doc A" in result.output

    def test_list_table_format(self):
        docs = [
            {
                "id": "d1234567-abcd-efgh-ijkl-mnopqrstuvwx",
                "title": "Doc A",
                "authors": ["Author X"],
                "pub_year": 2000,
                "passage_count": 5,
                "min_page": 1,
                "max_page": 10,
            }
        ]
        mock_conn = self._mock_conn_with_docs(docs)
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 0
            assert "Doc A" in result.output

    def test_list_empty(self):
        mock_conn = self._mock_conn_with_docs([])
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["list"])
            assert "No documents found" in result.output

    def test_list_error(self):
        runner = CliRunner()
        with patch("tome.documents.db_connection", side_effect=Exception("DB error")):
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 1


# ── CLI: info command ─────────────────────────────────────────────────────


class TestInfoCliCommand:
    def test_info_json(self):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.side_effect = [
            {
                "id": "d1",
                "title": "Doc",
                "authors": ["A"],
                "pub_year": 2000,
                "source_path": "/path",
            },
            {"total_passages": 10},
            {"entity_count": 50},
            {"year_count": 15},
            {"min_page": 1, "max_page": 20},
        ]
        mock_conn.cursor.return_value = cursor
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["info", "d1", "--format", "json"])
            assert result.exit_code == 0
            assert "Doc" in result.output

    def test_info_table(self):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.side_effect = [
            {
                "id": "d1",
                "title": "Doc Title",
                "authors": ["Author A"],
                "pub_year": 2000,
                "source_path": "/path",
            },
            {"total_passages": 10},
            {"entity_count": 50},
            {"year_count": 15},
            {"min_page": 1, "max_page": 20},
        ]
        mock_conn.cursor.return_value = cursor
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["info", "d1"])
            assert result.exit_code == 0
            assert "Doc Title" in result.output

    def test_info_not_found(self):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = cursor
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["info", "nonexistent"])
            assert result.exit_code == 1


# ── CLI: delete command ───────────────────────────────────────────────────


class TestDeleteCliCommand:
    def test_delete_with_force(self):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.rowcount = 1
        mock_conn.cursor.return_value = cursor
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["delete", "doc-1", "--force"])
            assert result.exit_code == 0

    def test_delete_failure_exits_1(self):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.execute.side_effect = Exception("DB error")
        mock_conn.cursor.return_value = cursor
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["delete", "doc-1", "--force"])
            assert result.exit_code == 1


# ── CLI: reindex command ──────────────────────────────────────────────────


class TestReindexCliCommand:
    def test_reindex_success(self):
        mock_conn = MagicMock()
        runner = CliRunner()
        with (
            patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)),
            patch("tome.documents.reindex_document_meilisearch", return_value=True),
        ):
            result = runner.invoke(cli, ["reindex", "doc-1"])
            assert result.exit_code == 0

    def test_reindex_failure(self):
        mock_conn = MagicMock()
        runner = CliRunner()
        with (
            patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)),
            patch("tome.documents.reindex_document_meilisearch", return_value=False),
        ):
            result = runner.invoke(cli, ["reindex", "doc-1"])
            assert result.exit_code == 1


# ── CLI: stats command ────────────────────────────────────────────────────


class TestStatsCliCommand:
    def test_stats_output(self):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.side_effect = [
            (10,),  # documents
            (100,),  # passages
            (500,),  # entities
            (50,),  # years
        ]
        mock_conn.cursor.return_value = cursor
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "Documents: 10" in result.output
            assert "Passages: 100" in result.output

    def test_stats_zero_passages(self):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.side_effect = [(0,), (0,), (0,), (0,)]
        mock_conn.cursor.return_value = cursor
        runner = CliRunner()
        with patch("tome.documents.db_connection", make_mock_db_connection(mock_conn)):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0

    def test_stats_error(self):
        runner = CliRunner()
        with patch("tome.documents.db_connection", side_effect=Exception("DB error")):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 1

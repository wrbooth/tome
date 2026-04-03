"""Tests for embed.py."""

from unittest.mock import MagicMock, patch

from embed import validate_embedding_dimension
from tests.conftest import make_mock_db_connection

# ── validate_embedding_dimension ──────────────────────────────────────────


class TestValidateEmbeddingDimension:
    def test_correct_dimension(self):
        assert validate_embedding_dimension([0.1] * 1536) is True

    def test_wrong_dimension(self):
        assert validate_embedding_dimension([0.1] * 768) is False

    def test_empty_list(self):
        assert validate_embedding_dimension([]) is False

    def test_custom_expected_dim(self):
        assert validate_embedding_dimension([0.1] * 768, expected_dim=768) is True


# ── get_openai_embeddings ─────────────────────────────────────────────────


class TestGetOpenaiEmbeddings:
    def test_single_batch(self, mock_openai_client):
        from embed import get_openai_embeddings

        with patch("embed.get_openai_client", return_value=mock_openai_client):
            texts = ["Hello world"]
            result = get_openai_embeddings(texts)
            assert len(result) == 1
            assert len(result[0]) == 1536

    def test_batching_over_100(self, mock_openai_client):
        from embed import get_openai_embeddings

        # Create mock that returns different batch sizes
        def make_response(batch):
            resp = MagicMock()
            data_items = []
            for _ in batch:
                item = MagicMock()
                item.embedding = [0.1] * 1536
                data_items.append(item)
            resp.data = data_items
            return resp

        mock_openai_client.embeddings.create.side_effect = lambda **kwargs: (
            make_response(kwargs["input"])
        )

        with patch("embed.get_openai_client", return_value=mock_openai_client):
            texts = [f"text {i}" for i in range(150)]
            result = get_openai_embeddings(texts)
            assert len(result) == 150
            # Should have been called twice: batch of 100 + batch of 50
            assert mock_openai_client.embeddings.create.call_count == 2

    def test_api_exception_returns_empty(self):
        from embed import get_openai_embeddings

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API error")
        with patch("embed.get_openai_client", return_value=mock_client):
            result = get_openai_embeddings(["test"])
            assert result == []


# ── get_local_embeddings ──────────────────────────────────────────────────


class TestGetLocalEmbeddings:
    def test_returns_embeddings(self):
        import numpy as np

        from embed import get_local_embeddings

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        with (
            patch("embed.SentenceTransformer", return_value=mock_model)
            if hasattr(__import__("embed"), "SentenceTransformer")
            else patch.dict("sys.modules", {"sentence_transformers": MagicMock()})
        ):
            # The function imports SentenceTransformer inside itself
            mock_st = MagicMock()
            mock_model_instance = MagicMock()
            mock_model_instance.encode.return_value = MagicMock(
                tolist=lambda: [[0.1, 0.2, 0.3]]
            )
            mock_st.SentenceTransformer.return_value = mock_model_instance

            with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
                result = get_local_embeddings(["test"])
                assert isinstance(result, list)
                assert len(result) > 0, "Expected at least one embedding returned"
                assert isinstance(result[0], list), (
                    "Expected each embedding to be a list"
                )


# ── get_unembedded_passages ───────────────────────────────────────────────


class TestGetUnembeddedPassages:
    def test_without_document_id(self, mock_db_conn):
        from embed import get_unembedded_passages

        mock_cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = [
            {"id": "p1", "text": "Text 1"},
            {"id": "p2", "text": "Text 2"},
        ]
        result = get_unembedded_passages(mock_db_conn, limit=100)
        assert len(result) == 2
        mock_cursor.execute.assert_called_once()
        # Verify no document_id in query args
        call_args = mock_cursor.execute.call_args
        assert len(call_args[0][1]) == 1  # only limit param

    def test_with_document_id(self, mock_db_conn):
        from embed import get_unembedded_passages

        mock_cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = [{"id": "p1", "text": "Text"}]
        result = get_unembedded_passages(mock_db_conn, limit=100, document_id="doc-1")
        assert len(result) == 1
        call_args = mock_cursor.execute.call_args
        assert len(call_args[0][1]) == 2  # document_id + limit


# ── update_passage_embeddings ─────────────────────────────────────────────


class TestUpdatePassageEmbeddings:
    def test_updates_and_commits(self, mock_db_conn):
        from embed import update_passage_embeddings

        mock_cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        embeddings = [("p1", [0.1] * 10), ("p2", [0.2] * 10)]
        update_passage_embeddings(mock_db_conn, embeddings)
        assert mock_cursor.execute.call_count == 2
        mock_db_conn.commit.assert_called_once()


# ── run_embeddings ────────────────────────────────────────────────────────


class TestRunEmbeddings:
    def test_no_passages_returns_true(self):
        from embed import run_embeddings

        mock_conn = MagicMock()
        with (
            patch("embed.db_connection", make_mock_db_connection(mock_conn)),
            patch("embed.get_unembedded_passages", return_value=[]),
        ):
            result = run_embeddings()
            assert result is True

    def test_failed_embedding_returns_false(self):
        from embed import run_embeddings

        mock_conn = MagicMock()
        passages = [{"id": "p1", "text": "Test text"}]
        with (
            patch("embed.db_connection", make_mock_db_connection(mock_conn)),
            patch("embed.get_unembedded_passages", return_value=passages),
            patch("embed.get_openai_embeddings", return_value=[]),
        ):
            result = run_embeddings(provider="openai")
            assert result is False

    def test_successful_embedding(self):
        from embed import run_embeddings

        mock_conn = MagicMock()
        passages = [{"id": "p1", "text": "Test text"}]
        embeddings = [[0.1] * 1536]
        with (
            patch("embed.db_connection", make_mock_db_connection(mock_conn)),
            patch("embed.get_unembedded_passages") as mock_get,
        ):
            # First call returns passages, second (remaining check) returns empty
            mock_get.side_effect = [passages, []]
            with (
                patch("embed.get_openai_embeddings", return_value=embeddings),
                patch("embed.update_passage_embeddings"),
            ):
                result = run_embeddings(provider="openai")
                assert result is True

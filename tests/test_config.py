"""Tests for config.py."""

import pytest
from unittest.mock import patch, MagicMock


class TestModelConstants:
    def test_default_query_analysis_model(self):
        with patch.dict("os.environ", {}, clear=True):
            # Re-import to pick up env changes
            import importlib
            import config
            importlib.reload(config)
            assert config.QUERY_ANALYSIS_MODEL == "gpt-4o-mini"

    def test_default_answer_model(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert config.ANSWER_MODEL == "gpt-5-mini-2025-08-07"

    def test_default_embedding_model(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert config.EMBEDDING_MODEL == "text-embedding-3-small"

    def test_default_reranker_model(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert config.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_env_override(self):
        with patch.dict("os.environ", {"QUERY_ANALYSIS_MODEL": "custom-model"}):
            import importlib
            import config
            importlib.reload(config)
            assert config.QUERY_ANALYSIS_MODEL == "custom-model"


class TestGetDbConnection:
    def test_calls_psycopg2_connect(self):
        from config import get_db_connection
        with patch("config.psycopg2.connect", return_value=MagicMock()) as mock_connect:
            get_db_connection()
            mock_connect.assert_called_once()

    def test_uses_env_vars(self):
        from config import get_db_connection
        with patch.dict("os.environ", {"DB_HOST": "myhost", "DB_PORT": "5433"}), \
             patch("config.psycopg2.connect", return_value=MagicMock()) as mock_connect:
            get_db_connection()
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["host"] == "myhost"
            assert call_kwargs["port"] == "5433"


class TestGetMeiliClient:
    def test_returns_client(self):
        from config import get_meili_client
        client = get_meili_client()
        assert client is not None


class TestGetOpenaiClient:
    def test_returns_client(self):
        from config import get_openai_client
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = get_openai_client()
            assert client is not None

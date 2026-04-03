"""Tests for config.py."""

from unittest.mock import MagicMock, patch


class TestSettings:
    def test_default_values(self):
        from tome.config import Settings

        with patch.dict("os.environ", {}, clear=True):
            s = Settings(_env_file=None)
            assert s.query_analysis_model == "gpt-4o-mini"
            assert s.answer_model == "gpt-5-mini-2025-08-07"
            assert s.embedding_model == "text-embedding-3-small"
            assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
            assert s.db_host == "localhost"
            assert s.db_port == "5432"
            assert s.meili_url == "http://localhost:7700"

    def test_env_override(self):
        from tome.config import Settings

        with patch.dict(
            "os.environ",
            {"QUERY_ANALYSIS_MODEL": "custom-model", "DB_HOST": "remotehost"},
        ):
            s = Settings(_env_file=None)
            assert s.query_analysis_model == "custom-model"
            assert s.db_host == "remotehost"


class TestModuleLevelConstants:
    def test_constants_match_settings(self):
        from tome.config import (
            ANSWER_MODEL,
            EMBEDDING_MODEL,
            QUERY_ANALYSIS_MODEL,
            RERANKER_MODEL,
            settings,
        )

        assert settings.query_analysis_model == QUERY_ANALYSIS_MODEL
        assert settings.answer_model == ANSWER_MODEL
        assert settings.embedding_model == EMBEDDING_MODEL
        assert settings.reranker_model == RERANKER_MODEL


class TestGetDbConnection:
    def test_creates_pool_and_returns_connection(self):
        from tome import config

        mock_pool = MagicMock()
        mock_pool.getconn.return_value = MagicMock()
        config._pool = None
        with patch("tome.config.ThreadedConnectionPool", return_value=mock_pool):
            conn = config.get_db_connection()
            mock_pool.getconn.assert_called_once()
            assert conn is not None
        config._pool = None

    def test_uses_settings_values(self):
        from tome import config
        from tome.config import Settings

        config._pool = None
        mock_settings = Settings(
            _env_file=None,
            db_host="myhost",
            db_port="5433",
        )
        with (
            patch("tome.config.settings", mock_settings),
            patch("tome.config.ThreadedConnectionPool") as mock_pool_cls,
        ):
            mock_pool_cls.return_value = MagicMock()
            config.get_db_connection()
            call_kwargs = mock_pool_cls.call_args[1]
            assert call_kwargs["host"] == "myhost"
            assert call_kwargs["port"] == "5433"
        config._pool = None


class TestGetMeiliClient:
    def test_returns_client(self):
        from tome.config import get_meili_client

        client = get_meili_client()
        assert client is not None


class TestGetOpenaiClient:
    def test_returns_client(self):
        from tome.config import get_openai_client

        with patch("tome.config.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            client = get_openai_client()
            assert client is not None

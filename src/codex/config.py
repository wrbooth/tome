"""
Shared configuration and client singletons for Codex.
"""

import logging
import sys
from contextlib import contextmanager

from meilisearch import Client as MeiliClient
from psycopg2.pool import ThreadedConnectionPool
from pydantic_settings import BaseSettings, SettingsConfigDict


def configure_logging(level=logging.INFO):
    """Configure logging for the Codex application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model configuration
    query_analysis_model: str = "gpt-4o-mini"
    answer_model: str = "gpt-5-mini-2025-08-07"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Database
    db_host: str = "localhost"
    db_port: str = "5432"
    db_name: str = "codex"
    db_user: str = "codex"
    db_password: str = "codex"  # noqa: S105 — dev default only

    # Meilisearch
    meili_url: str = "http://localhost:7700"

    # OpenAI
    openai_api_key: str = ""


settings = Settings()

# Module-level constants for backward compatibility
QUERY_ANALYSIS_MODEL = settings.query_analysis_model
ANSWER_MODEL = settings.answer_model
EMBEDDING_MODEL = settings.embedding_model
RERANKER_MODEL = settings.reranker_model
MEILI_URL = settings.meili_url

# Database

_pool = None


def _get_pool():
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
        )
    return _pool


@contextmanager
def db_connection():
    """Get a pooled database connection as a context manager."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def get_db_connection():
    """Get a database connection from the pool.

    Kept for backward compatibility. Prefer using ``db_connection()``
    as a context manager instead.
    """
    return _get_pool().getconn()


# Meilisearch


def get_meili_client() -> MeiliClient:
    """Get a Meilisearch client."""
    return MeiliClient(settings.meili_url)


# OpenAI


def get_openai_client():
    """Get an OpenAI client."""
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key)

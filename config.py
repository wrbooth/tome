"""
Shared configuration and client singletons for Codex.
"""

import os
import sys
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv
from meilisearch import Client as MeiliClient


def configure_logging(level=logging.INFO):
    """Configure logging for the Codex application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stderr,
    )

load_dotenv()

# Model configuration
QUERY_ANALYSIS_MODEL = os.getenv("QUERY_ANALYSIS_MODEL", "gpt-4o-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-5-mini-2025-08-07")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Database

_pool = None

def _get_pool():
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "codex"),
            user=os.getenv("DB_USER", "codex"),
            password=os.getenv("DB_PASSWORD", "codex"),
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

MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")

def get_meili_client() -> MeiliClient:
    """Get a Meilisearch client."""
    return MeiliClient(MEILI_URL)

# OpenAI

def get_openai_client():
    """Get an OpenAI client."""
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

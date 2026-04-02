"""
Shared configuration and client singletons for Codex.
"""

import os
import psycopg2
from dotenv import load_dotenv
from meilisearch import Client as MeiliClient

load_dotenv()

# Model configuration
QUERY_ANALYSIS_MODEL = os.getenv("QUERY_ANALYSIS_MODEL", "gpt-4o-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-5-mini-2025-08-07")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Database

def get_db_connection():
    """Get a new database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "codex"),
        user=os.getenv("DB_USER", "codex"),
        password=os.getenv("DB_PASSWORD", "codex")
    )

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

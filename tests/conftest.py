"""Shared fixtures for Codex unit tests."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


def make_mock_db_connection(mock_conn):
    """Create a context manager that yields the given mock connection.

    Useful for patching ``config.db_connection`` in tests.
    """

    @contextmanager
    def _db_connection():
        yield mock_conn

    return _db_connection


@pytest.fixture
def mock_db_conn():
    """Mock database connection with cursor context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def sample_pages():
    """Sample page data for chunking tests."""
    return [
        {
            "page": 1,
            "text": (
                "CHAPTER 1: The Beginning\n"
                "This is the first paragraph of content.\n"
                "This is the second paragraph."
            ),
            "headings": [
                {
                    "text": "The Beginning",
                    "line_number": 0,
                    "level": 1,
                    "full_text": "CHAPTER 1: The Beginning",
                    "detection_method": "regex",
                }
            ],
        },
        {
            "page": 2,
            "text": (
                "More content on page two.\n"
                "Another paragraph here.\n"
                "Section 1.1 Details\n"
                "Details about section one point one."
            ),
            "headings": [
                {
                    "text": "Details",
                    "line_number": 2,
                    "level": 2,
                    "full_text": "Section 1.1 Details",
                    "detection_method": "regex",
                }
            ],
        },
    ]


@pytest.fixture
def sample_pages_no_headings():
    """Sample pages without headings."""
    return [
        {
            "page": 1,
            "text": "Just some plain text.\nWith a second paragraph.",
            "headings": [],
        }
    ]


@pytest.fixture
def sample_chunks():
    """Pre-built chunk dicts."""
    return [
        {
            "page": 1,
            "text": "Doc Title | Chapter One | First chunk of text here.",
            "original_text": "First chunk of text here.",
            "headings_path": ["Chapter One"],
        },
        {
            "page": 2,
            "text": "Doc Title | Chapter Two | Second chunk of text here.",
            "original_text": "Second chunk of text here.",
            "headings_path": ["Chapter Two"],
        },
    ]


@pytest.fixture
def sample_search_results():
    """Sample search result dicts."""
    return [
        {
            "id": "passage-1",
            "text": "George Washington was the first president of the United States.",
            "page": 10,
            "title": "History of America",
            "headings_path": ["Chapter 1", "Presidents"],
            "score": 0.95,
        },
        {
            "id": "passage-2",
            "text": "The American Revolution began in 1775 and ended in 1783.",
            "page": 25,
            "title": "History of America",
            "headings_path": ["Chapter 3", "Revolution"],
            "score": 0.82,
        },
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = MagicMock()

    # Mock chat completions
    chat_response = MagicMock()
    chat_response.choices = [MagicMock()]
    chat_response.choices[0].message.content = (
        '{"query_type": "general", "person": "",'
        ' "entities": {"persons": [], "places": [],'
        ' "events": [], "dates": [], "families": [],'
        ' "companies": [], "industries": [],'
        ' "settlement_terms": []},'
        ' "expansions": ["test query"]}'
    )
    client.chat.completions.create.return_value = chat_response

    # Mock embeddings
    emb_response = MagicMock()
    emb_data = MagicMock()
    emb_data.embedding = [0.1] * 1536
    emb_response.data = [emb_data]
    client.embeddings.create.return_value = emb_response

    return client


@pytest.fixture
def mock_meili_client():
    """Mock Meilisearch client."""
    client = MagicMock()
    index = MagicMock()
    client.index.return_value = index

    # Mock search response
    index.search.return_value = {
        "hits": [
            {"id": "passage-1", "_rankingScore": 0.95},
            {"id": "passage-2", "_rankingScore": 0.82},
        ]
    }

    index.get_settings.return_value = {"embedders": {"default": {}}}
    task = MagicMock()
    task.task_uid = 1
    index.add_documents.return_value = task
    index.update_embedders.return_value = task
    index.update_filterable_attributes.return_value = task

    return client

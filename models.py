"""
Pydantic models for the Codex project.

Shared data models used across the ingestion pipeline, search, and API layers.
These models define the canonical shapes for chunks, search results, documents,
and query analysis output.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A text chunk produced by chunk_text_with_headings()."""
    page: int = Field(description="Source page number")
    text: str = Field(description="Full text including heading prefix")
    original_text: str = Field(description="Text without heading prefix")
    headings_path: List[str] = Field(default_factory=list, description="Heading hierarchy for this chunk")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class PassageDetail(BaseModel):
    """A passage returned by get_passage_details(), enriched with a reranker score."""
    id: str = Field(description="Passage UUID")
    text: str = Field(description="Full passage text")
    page: int = Field(description="Source page number")
    title: str = Field(description="Document title")
    headings_path: Optional[List[str]] = Field(default=None, description="Heading hierarchy")
    score: float = Field(default=0.0, description="Reranker score")


class QueryEntities(BaseModel):
    """Entities extracted during query analysis."""
    persons: List[str] = Field(default_factory=list)
    places: List[str] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    families: List[str] = Field(default_factory=list)
    companies: List[str] = Field(default_factory=list)
    industries: List[str] = Field(default_factory=list)
    settlement_terms: List[str] = Field(default_factory=list)


class QueryAnalysis(BaseModel):
    """Result of analyze_query()."""
    query_type: str = Field(description="Classification: who, when, where, person, company, factoid, general")
    person: Optional[str] = Field(default=None, description="Primary person referenced in query")
    entities: QueryEntities = Field(default_factory=QueryEntities)
    expansions: List[str] = Field(default_factory=list, description="Search expansion terms")


class SearchCodexResult(BaseModel):
    """Top-level result returned by search_codex()."""
    query_type: str
    answer: str
    results: List[PassageDetail] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    """Document metadata as stored in the documents table."""
    id: str = Field(description="Document UUID")
    title: str
    authors: Optional[List[str]] = Field(default=None)
    pub_year: Optional[int] = Field(default=None)
    source_path: Optional[str] = Field(default=None)


class DocumentListItem(BaseModel):
    """Document with summary stats, as returned by list_documents()."""
    id: str
    title: str
    authors: Optional[List[str]] = Field(default=None)
    pub_year: Optional[int] = Field(default=None)
    passage_count: int = 0
    embedded_count: int = 0
    min_page: Optional[int] = None
    max_page: Optional[int] = None


# ---------------------------------------------------------------------------
# API-specific models
# ---------------------------------------------------------------------------

class QueryAnalysisInfo(BaseModel):
    """Query analysis details returned alongside search results."""
    query_type: str
    person: Optional[str] = None
    entities: QueryEntities = Field(default_factory=QueryEntities)
    expansions: List[str] = Field(default_factory=list)


class DocumentPassageStats(BaseModel):
    total_passages: int = 0
    embedded_passages: int = 0


class DocumentEntityStats(BaseModel):
    entity_count: int = 0


class DocumentYearStats(BaseModel):
    year_count: int = 0


class DocumentPageRange(BaseModel):
    min_page: Optional[int] = None
    max_page: Optional[int] = None


class DocumentDetail(BaseModel):
    """Full document detail as returned by get_document_stats()."""
    document: DocumentInfo
    passages: DocumentPassageStats = Field(default_factory=DocumentPassageStats)
    entities: DocumentEntityStats = Field(default_factory=DocumentEntityStats)
    years: DocumentYearStats = Field(default_factory=DocumentYearStats)
    pages: DocumentPageRange = Field(default_factory=DocumentPageRange)


class IngestTaskInfo(BaseModel):
    """Status of a background document ingestion task."""
    task_id: str
    status: str = Field(description="pending, running, completed, or failed")
    document_id: Optional[str] = None
    filename: Optional[str] = None
    message: str = ""
    created_at: str = Field(description="ISO 8601 timestamp")

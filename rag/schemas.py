"""
MailMind RAG — Pydantic Schemas

Request/response models for the API layer and internal data transfer.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ─── Internal Data Models ────────────────────────────────────────────

class EmailRecord(BaseModel):
    """Normalized internal email representation."""
    message_id: str
    thread_id: str = ""
    sender_name: str = ""
    sender_email: str = ""
    subject: str = ""
    date: str = ""
    body: str = ""
    labels: list[str] = Field(default_factory=list)
    category: str = ""
    has_attachment: bool = False
    attachment_names: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A searchable chunk derived from an email."""
    chunk_id: int = 0
    message_id: str
    subject: str = ""
    sender_name: str = ""
    sender_email: str = ""
    date: str = ""
    text: str  # the full searchable text (metadata header + body chunk)
    body_snippet: str = ""  # just the body portion for display


# ─── API Request Models ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class SummarizeRequest(BaseModel):
    message_id: Optional[str] = None  # single email summarization
    query: Optional[str] = None       # multi-email summarization
    top_k: int = 5


class ConversationRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5


class IndexBuildRequest(BaseModel):
    force_rebuild: bool = False


# ─── API Response Models ─────────────────────────────────────────────

class SearchResult(BaseModel):
    message_id: str
    subject: str = ""
    sender_name: str = ""
    sender_email: str = ""
    date: str = ""
    score: float = 0.0
    snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult] = Field(default_factory=list)
    total: int = 0


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult] = Field(default_factory=list)


class SummarizeResponse(BaseModel):
    summary: str
    sources: list[SearchResult] = Field(default_factory=list)
    mode: str = ""  # "single" or "multi"


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ConversationResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    history: list[ConversationMessage] = Field(default_factory=list)


class IndexStatusResponse(BaseModel):
    indexed: bool = False
    total_vectors: int = 0
    total_emails: int = 0
    index_path: str = ""


class EmailDetailResponse(BaseModel):
    """Full email details."""
    message_id: str
    thread_id: str = ""
    sender_name: str = ""
    sender_email: str = ""
    subject: str = ""
    date: str = ""
    body: str = ""
    labels: list[str] = Field(default_factory=list)
    category: str = ""
    has_attachment: bool = False
    attachment_names: list[str] = Field(default_factory=list)

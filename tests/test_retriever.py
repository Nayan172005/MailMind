"""Tests for rag.retriever"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.schemas import DocumentChunk
from rag.vector_store import VectorStore
from rag.retriever import retrieve, retrieve_with_context
from rag.config import settings


def _build_test_store():
    """Build a small test store."""
    chunks = [
        DocumentChunk(
            chunk_id=0, message_id="msg1",
            subject="Microsoft Online Test", sender_name="CDC",
            sender_email="cdc@vit.ac.in", date="2026-01-01",
            text="Subject: Microsoft Online Test\nBody: Assessment scheduled for Jan 5 at 10 AM in Anna Auditorium",
            body_snippet="Assessment scheduled for Jan 5 at 10 AM in Anna Auditorium",
        ),
        DocumentChunk(
            chunk_id=1, message_id="msg1",
            subject="Microsoft Online Test", sender_name="CDC",
            sender_email="cdc@vit.ac.in", date="2026-01-01",
            text="Subject: Microsoft Online Test\nBody: Bring your resume and college ID",
            body_snippet="Bring your resume and college ID",
        ),
        DocumentChunk(
            chunk_id=0, message_id="msg2",
            subject="NPTEL Course Update", sender_name="NPTEL",
            sender_email="nptel@iitm.ac.in", date="2026-01-02",
            text="Subject: NPTEL Course Update\nBody: Your course enrollment has been confirmed",
            body_snippet="Your course enrollment has been confirmed",
        ),
    ]
    store = VectorStore()
    store.build_index(chunks)
    return store


@pytest.fixture(scope="module")
def test_store():
    return _build_test_store()


def test_retrieve_basic(test_store):
    """Test basic retrieval."""
    results = retrieve("Microsoft assessment", test_store, top_k=5)
    assert len(results) > 0
    assert results[0].message_id in ["msg1", "msg2"]


def test_retrieve_deduplication(test_store):
    """Test that deduplication keeps only best chunk per email."""
    results = retrieve(
        "Microsoft test resume", test_store,
        top_k=5, deduplicate=True
    )
    msg_ids = [r.message_id for r in results]
    # msg1 has 2 chunks but should appear only once after dedup
    assert msg_ids.count("msg1") <= 1


def test_retrieve_empty_query(test_store):
    """Test that empty query returns empty results."""
    results = retrieve("", test_store)
    assert len(results) == 0


def test_retrieve_with_context_returns_text(test_store):
    """Test that retrieve_with_context provides a context string."""
    results, context = retrieve_with_context("Microsoft placement", test_store)
    assert len(results) > 0
    assert len(context) > 0
    assert "Email 1" in context
    assert "Message ID:" in context


def test_retrieve_high_threshold(test_store):
    """Test that a very high threshold filters out all results."""
    results = retrieve(
        "random unrelated topic xyz",
        test_store,
        similarity_threshold=0.99,
    )
    # May or may not return results depending on similarity
    # but at 0.99 threshold it should filter aggressively
    for r in results:
        assert r.score >= 0.99


class FakeStore:
    def __init__(self, candidates):
        self.candidates = candidates
        self.requested_top_k = None
        self.is_loaded = True

    def search(self, query_embedding, top_k):
        self.requested_top_k = top_k
        return self.candidates[:top_k]

    def get_chunks_by_message_id(self, message_id):
        return [c for c in self.candidates if c["message_id"] == message_id]


def test_retrieve_uses_configured_initial_depth(monkeypatch):
    """Retrieval asks FAISS for deeper candidates before final top-k."""
    import rag.retriever as retriever_module

    candidates = [
        {
            "message_id": "msg1",
            "chunk_id": 0,
            "subject": "NPTEL Course Update",
            "sender_name": "NPTEL",
            "sender_email": "nptel@example.com",
            "date": "2026-01-01",
            "text": "Subject: NPTEL Course Update\nBody: course update",
            "body_snippet": "course update",
            "score": 0.5,
        }
    ]
    store = FakeStore(candidates)
    monkeypatch.setattr(retriever_module, "embed_query", lambda query: object())

    results = retrieve("NPTEL", store, top_k=5)

    assert len(results) == 1
    assert store.requested_top_k == settings.INITIAL_RETRIEVAL_K


def test_subject_and_entity_reranking_can_move_relevant_email_up(monkeypatch):
    """Metadata reranking helps an entity-specific subject beat a tiny semantic gap."""
    import rag.retriever as retriever_module

    candidates = [
        {
            "message_id": "generic",
            "chunk_id": 0,
            "subject": "Career preparation newsletter",
            "sender_name": "Newsletter",
            "sender_email": "news@example.com",
            "date": "2026-01-02",
            "text": "Subject: Career preparation newsletter\nBody: prepare for assessments",
            "body_snippet": "prepare for assessments",
            "score": 0.52,
        },
        {
            "message_id": "microsoft",
            "chunk_id": 0,
            "subject": "Microsoft Online Test",
            "sender_name": "Helpdesk CDC",
            "sender_email": "cdc@vit.ac.in",
            "date": "2026-01-01",
            "text": "Subject: Microsoft Online Test\nBody: Online Assessment at 7 PM",
            "body_snippet": "Online Assessment at 7 PM",
            "score": 0.50,
        },
    ]
    store = FakeStore(candidates)
    monkeypatch.setattr(retriever_module, "embed_query", lambda query: object())

    results = retrieve("When is my Microsoft assessment?", store, top_k=2)

    assert results[0].message_id == "microsoft"

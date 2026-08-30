"""Tests for rag.vector_store"""

import os
import sys
import pytest
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.schemas import DocumentChunk
from rag.vector_store import VectorStore
from rag.embedder import embed_texts, embed_query


def _make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=0,
            message_id="msg1",
            subject="Microsoft Placement Drive",
            sender_name="CDC",
            sender_email="cdc@vit.ac.in",
            date="2026-01-01",
            text="Subject: Microsoft Placement Drive\nBody: Online test scheduled for January 5",
            body_snippet="Online test scheduled for January 5",
        ),
        DocumentChunk(
            chunk_id=0,
            message_id="msg2",
            subject="NPTEL Registration",
            sender_name="NPTEL",
            sender_email="nptel@iitm.ac.in",
            date="2026-01-02",
            text="Subject: NPTEL Registration\nBody: Register for the online course by January 10",
            body_snippet="Register for the online course by January 10",
        ),
        DocumentChunk(
            chunk_id=0,
            message_id="msg3",
            subject="Security Alert",
            sender_name="Google",
            sender_email="no-reply@google.com",
            date="2026-01-03",
            text="Subject: Security Alert\nBody: New sign-in detected on your account",
            body_snippet="New sign-in detected on your account",
        ),
    ]


@pytest.fixture(scope="module")
def built_store():
    """Build a test store once for the module."""
    store = VectorStore()
    chunks = _make_chunks()
    store.build_index(chunks)
    return store


def test_build_index(built_store):
    """Test that index is built correctly."""
    assert built_store.is_loaded
    assert built_store.total_vectors == 3
    assert built_store.total_emails == 3


def test_search(built_store):
    """Test basic search."""
    query_emb = embed_query("Microsoft placement")
    results = built_store.search(query_emb, top_k=3)
    assert len(results) > 0
    assert results[0]["message_id"] in ["msg1", "msg2", "msg3"]
    assert "score" in results[0]


def test_search_relevance(built_store):
    """Test that search returns relevant results first."""
    query_emb = embed_query("placement drive test")
    results = built_store.search(query_emb, top_k=3)
    # The Microsoft placement chunk should rank high
    top_subjects = [r["subject"] for r in results[:2]]
    assert any("Microsoft" in s or "Placement" in s for s in top_subjects)


def test_save_and_load(built_store):
    """Test saving and loading the index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        built_store.save(tmpdir)

        new_store = VectorStore()
        new_store.load(tmpdir)

        assert new_store.is_loaded
        assert new_store.total_vectors == built_store.total_vectors
        assert new_store.total_emails == built_store.total_emails


def test_update_index():
    """Test incremental update."""
    store = VectorStore()
    initial_chunks = _make_chunks()[:2]
    store.build_index(initial_chunks)
    assert store.total_vectors == 2

    all_chunks = _make_chunks()
    added = store.update_index(all_chunks)
    assert added == 1  # only msg3 is new
    assert store.total_vectors == 3


def test_get_chunks_by_message_id(built_store):
    """Test retrieving chunks by message_id."""
    chunks = built_store.get_chunks_by_message_id("msg1")
    assert len(chunks) == 1
    assert chunks[0]["subject"] == "Microsoft Placement Drive"


def test_get_chunks_missing_id(built_store):
    """Test retrieving chunks for non-existent message_id."""
    chunks = built_store.get_chunks_by_message_id("nonexistent")
    assert len(chunks) == 0

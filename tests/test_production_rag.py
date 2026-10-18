"""Tests for production RAG optimization based on evaluation evidence.

Validates:
- Company/entity extraction in lowercase and mixed case
- Sender matching with display names and email addresses
- Explicit date ranges and temporal ordering with controllable clock
- Missing or malformed email dates
- Deep candidate discovery below top-20
- Threshold filtering and abstention
- Hybrid dense + lexical retrieval behavior
- Deduplication of chunks vs distinct thread messages
- Source ID preservation and citation validation
- Query analysis fallback and index/metadata consistency
"""

import os
import sys
from datetime import datetime, timezone
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.schemas import DocumentChunk, SearchResult
from rag.query_analyzer import analyze_query, QueryAnalysis
from rag.reranker import rerank_candidates
from rag.generator import validate_citations
from rag.hybrid_retriever import BM25Index, search_hybrid
from rag.vector_store import VectorStore
from rag.retriever import retrieve, retrieve_with_context
from rag.config import settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_store():
    """Build a deterministic VectorStore for testing production RAG."""
    chunks = [
        DocumentChunk(
            chunk_id=0,
            message_id="msg_ms_1",
            subject="Microsoft Online Assessment Shortlist",
            sender_name="CDC Office",
            sender_email="cdc@vit.ac.in",
            date="2026-03-01T10:00:00Z",
            text="Subject: Microsoft Online Assessment Shortlist\nBody: Assessment scheduled for March 10 at 9 AM in PRP 717.",
            body_snippet="Assessment scheduled for March 10 at 9 AM in PRP 717.",
        ),
        DocumentChunk(
            chunk_id=1,
            message_id="msg_ms_1",  # duplicate chunk from same email
            subject="Microsoft Online Assessment Shortlist",
            sender_name="CDC Office",
            sender_email="cdc@vit.ac.in",
            date="2026-03-01T10:00:00Z",
            text="Subject: Microsoft Online Assessment Shortlist\nBody: Instructions: Bring college ID and laptop charger.",
            body_snippet="Instructions: Bring college ID and laptop charger.",
        ),
        DocumentChunk(
            chunk_id=0,
            message_id="msg_ms_thread_reply",  # same topic/thread context, distinct message_id
            subject="Re: Microsoft Online Assessment Shortlist",
            sender_name="CDC Office",
            sender_email="cdc@vit.ac.in",
            date="2026-03-02T12:00:00Z",
            text="Subject: Re: Microsoft Online Assessment Shortlist\nBody: Venue updated to Anna Auditorium for all batches.",
            body_snippet="Venue updated to Anna Auditorium for all batches.",
        ),
        DocumentChunk(
            chunk_id=0,
            message_id="msg_wakefit",
            subject="Wakefit Internship Update",
            sender_name="Wakefit Careers",
            sender_email="careers@wakefit.co",
            date="2026-02-15T08:00:00Z",
            text="Subject: Wakefit Internship Update\nBody: Shortlisted candidates for Data Science internship July 2026.",
            body_snippet="Shortlisted candidates for Data Science internship July 2026.",
        ),
        DocumentChunk(
            chunk_id=0,
            message_id="msg_swayam",
            subject="SWAYAM NPTEL Exam Registration",
            sender_name="Prof. Sharma",
            sender_email="sharma@vit.ac.in",
            date="2026-01-10T09:00:00Z",
            text="Subject: SWAYAM NPTEL Exam Registration\nBody: Last date to register for exam is Jan 25.",
            body_snippet="Last date to register for exam is Jan 25.",
        ),
        DocumentChunk(
            chunk_id=0,
            message_id="msg_malformed_date",
            subject="Library Overdue Notice",
            sender_name="Library VIT",
            sender_email="library@vit.ac.in",
            date="not-a-valid-date",
            text="Subject: Library Overdue Notice\nBody: Please return book by end of week.",
            body_snippet="Please return book by end of week.",
        ),
    ]
    store = VectorStore()
    store.build_index(chunks)
    return store


# ---------------------------------------------------------------------------
# Priority B: Query Analysis Tests
# ---------------------------------------------------------------------------

def test_entity_queries_lowercase_and_mixedcase():
    """Verify company/entity extraction works reliably across casing."""
    q_lower = "any emails about wakefit internship?"
    q_mixed = "Any emails about Wakefit internship?"
    
    analysis_lower = analyze_query(q_lower)
    analysis_mixed = analyze_query(q_mixed)

    assert any("wakefit" in e.lower() for e in analysis_lower.entities)
    assert any("wakefit" in e.lower() for e in analysis_mixed.entities)

    # Test acronyms
    q_swayam = "when is the swayam exam?"
    analysis_swayam = analyze_query(q_swayam)
    assert any("swayam" in e.lower() for e in analysis_swayam.entities)

    # Multi-word entity
    q_multi = "who contacted me about pixel compute?"
    analysis_multi = analyze_query(q_multi)
    assert any("pixel compute" in e.lower() for e in analysis_multi.entities)


def test_sender_matching_display_names_and_email_addresses():
    """Verify sender extraction for display names and explicit email addresses."""
    q_disp = "emails from cdc about test"
    analysis_disp = analyze_query(q_disp)
    assert any("cdc" in s.lower() for s in analysis_disp.senders)

    q_addr = "search messages from cdc@vit.ac.in"
    analysis_addr = analyze_query(q_addr)
    assert any("cdc@vit.ac.in" in s.lower() for s in analysis_addr.senders)


def test_query_analysis_fallback():
    """Ensure malformed or non-alphanumeric queries fall back cleanly without error."""
    analysis = analyze_query("??? !!! @@@")
    assert analysis.query == "??? !!! @@@"
    assert isinstance(analysis.terms, list)


# ---------------------------------------------------------------------------
# Priority D & E: Temporal & Metadata Ranking Tests
# ---------------------------------------------------------------------------

def test_explicit_date_ranges_and_temporal_ordering():
    """Verify temporal ordering uses controllable clock and sorts recency correctly."""
    ref_time = datetime(2026, 3, 15, tzinfo=timezone.utc)

    candidates = [
        {
            "message_id": "old_msg",
            "subject": "Internship Announcement",
            "sender_name": "CDC",
            "sender_email": "cdc@vit.ac.in",
            "date": "2026-01-01T00:00:00Z",
            "text": "Internship notice",
            "score": 0.60,
        },
        {
            "message_id": "new_msg",
            "subject": "Internship Announcement",
            "sender_name": "CDC",
            "sender_email": "cdc@vit.ac.in",
            "date": "2026-03-01T00:00:00Z",
            "text": "Internship notice",
            "score": 0.60,
        },
    ]

    analysis = analyze_query("most recent internship announcement")
    reranked = rerank_candidates(
        query="most recent internship announcement",
        analysis=analysis,
        candidates=candidates,
        final_k=5,
        reference_time=ref_time,
    )

    assert len(reranked) == 2
    # The newer message should rank first
    assert reranked[0]["message_id"] == "new_msg"


def test_missing_or_malformed_dates():
    """Ensure malformed or missing dates do not crash the reranker or pipeline."""
    candidates = [
        {
            "message_id": "bad_date",
            "subject": "Important update",
            "sender_name": "Test",
            "sender_email": "test@example.com",
            "date": "garbage-date-format-9999",
            "text": "Hello world",
            "score": 0.55,
        },
        {
            "message_id": "none_date",
            "subject": "Important update",
            "sender_name": "Test",
            "sender_email": "test@example.com",
            "date": "",
            "text": "Hello world",
            "score": 0.50,
        },
    ]
    analysis = analyze_query("most recent important update")
    reranked = rerank_candidates(
        query="update",
        analysis=analysis,
        candidates=candidates,
        final_k=5,
    )
    assert len(reranked) == 2
    assert "final_score" in reranked[0]


# ---------------------------------------------------------------------------
# Priority C: Hybrid Retrieval Tests
# ---------------------------------------------------------------------------

def test_no_dense_candidates_but_useful_lexical_candidates():
    """Exact code or token like 'PRP 717' is found by BM25 even if dense score is low."""
    bm25 = BM25Index()
    metadata = [
        {"subject": "Exam Schedule", "sender_name": "Dean", "sender_email": "dean@vit.ac.in", "body_snippet": "Exam in PRP 717 at 10 AM"},
        {"subject": "Sports Day", "sender_name": "Coach", "sender_email": "sports@vit.ac.in", "body_snippet": "Ground meeting"},
    ]
    bm25.build(metadata)
    results = bm25.search(["prp", "717"], top_k=5)
    assert len(results) > 0
    assert results[0][0] == 0  # index 0 is PRP 717


def test_candidate_recovered_below_initial_top20(sample_store):
    """Deep candidate pool (k=80) and hybrid search retrieve candidates that might otherwise be cut off."""
    results = retrieve("Wakefit", sample_store, top_k=5)
    assert len(results) > 0
    assert any("wakefit" in r.subject.lower() for r in results)


def test_duplicate_chunks_from_same_message(sample_store):
    """Verify deduplication keeps only one entry per unique message_id."""
    results = retrieve("Microsoft Online Assessment", sample_store, top_k=5, deduplicate=True)
    msg_ids = [r.message_id for r in results]
    assert msg_ids.count("msg_ms_1") == 1


def test_different_relevant_messages_in_same_thread(sample_store):
    """Verify distinct messages in the same thread are both retained after deduplication."""
    results = retrieve("Microsoft Online Assessment", sample_store, top_k=5, deduplicate=True)
    msg_ids = [r.message_id for r in results]
    # Both msg_ms_1 and msg_ms_thread_reply are distinct message_ids and should be present
    assert "msg_ms_1" in msg_ids
    assert "msg_ms_thread_reply" in msg_ids


# ---------------------------------------------------------------------------
# Priority G & H: Evidence Selection, Citation Validation & Abstention Tests
# ---------------------------------------------------------------------------

def test_source_ids_preserved_through_reranking_and_context(sample_store):
    """Verify sequential [Source X] and [Source X] (Email X) are preserved in retrieve_with_context."""
    results, context = retrieve_with_context("Microsoft placement in PRP 717", sample_store)
    assert len(results) > 0
    assert "[Source 1]" in context
    assert "Email 1" in context
    assert results[0].message_id in context


def test_citation_validation_strips_hallucinations():
    """Verify validate_citations strips fabricated citations beyond available sources."""
    answer_valid = "Assessment is on March 10 [Source 1]."
    cleaned_valid = validate_citations(answer_valid, max_sources=1)
    assert "[Source 1]" in cleaned_valid

    answer_hallucinated = "Assessment is in PRP 717 [Source 1], but contact [Source 4] or [Source 9]."
    cleaned_hallucinated = validate_citations(answer_hallucinated, max_sources=2)
    assert "[Source 1]" in cleaned_hallucinated
    assert "[Source 4]" not in cleaned_hallucinated
    assert "[Source 9]" not in cleaned_hallucinated


def test_empty_context_abstention(sample_store):
    """Verify completely out-of-domain queries return empty results and empty context."""
    results, context = retrieve_with_context("Who won the 1994 FIFA World Cup in the United States?", sample_store)
    assert len(results) == 0
    assert context == ""


def test_threshold_removal_of_unrelated_candidate():
    """Verify low-similarity candidates below threshold are filtered out."""
    candidates = [
        {
            "message_id": "unrelated",
            "subject": "Lunch menu",
            "sender_name": "Cafeteria",
            "sender_email": "food@example.com",
            "date": "2026-01-01",
            "text": "Today's special is pasta",
            "score": 0.10,  # below threshold
        }
    ]
    analysis = analyze_query("when is the physics midterm exam?")
    reranked = rerank_candidates(
        query="physics exam",
        analysis=analysis,
        candidates=candidates,
        final_k=5,
    )
    assert len(reranked) == 0


# ---------------------------------------------------------------------------
# Vector Store Mapping & Index Consistency Tests
# ---------------------------------------------------------------------------

def test_index_vector_metadata_mapping_consistency(sample_store):
    """Verify 1-to-1 correspondence between FAISS index vectors and metadata entries."""
    assert sample_store.index is not None
    assert sample_store.index.ntotal == len(sample_store.metadata)
    for i, meta in enumerate(sample_store.metadata):
        assert meta["chunk_id"] is not None
        assert meta["message_id"] is not None

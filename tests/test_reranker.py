"""Tests for metadata-aware candidate reranking."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query_analyzer import analyze_query
from rag.reranker import parse_email_date, rerank_candidates


def test_reranker_deduplicates_message_ids():
    analysis = analyze_query("Microsoft assessment")
    candidates = [
        {
            "message_id": "msg1",
            "chunk_id": 0,
            "subject": "Microsoft Online Test",
            "sender_name": "CDC",
            "sender_email": "cdc@vit.ac.in",
            "date": "2026-01-01",
            "text": "Microsoft online test",
            "body_snippet": "Microsoft online test",
            "score": 0.45,
        },
        {
            "message_id": "msg1",
            "chunk_id": 1,
            "subject": "Microsoft Online Test",
            "sender_name": "CDC",
            "sender_email": "cdc@vit.ac.in",
            "date": "2026-01-01",
            "text": "Microsoft online assessment details",
            "body_snippet": "Microsoft online assessment details",
            "score": 0.50,
        },
    ]

    ranked = rerank_candidates("Microsoft assessment", analysis, candidates, final_k=5)

    assert len(ranked) == 1
    assert ranked[0]["message_id"] == "msg1"
    assert ranked[0]["chunk_id"] == 1


def test_temporal_reranking_prefers_recent_relevant_email():
    analysis = analyze_query("most recent placement company")
    candidates = [
        {
            "message_id": "old",
            "chunk_id": 0,
            "subject": "Placement company interview",
            "sender_name": "CDC",
            "sender_email": "cdc@vit.ac.in",
            "date": "2025-01-01",
            "text": "Placement company interview",
            "body_snippet": "Placement company interview",
            "score": 0.55,
        },
        {
            "message_id": "new",
            "chunk_id": 0,
            "subject": "Placement company interview",
            "sender_name": "CDC",
            "sender_email": "cdc@vit.ac.in",
            "date": "2026-07-22",
            "text": "Placement company interview",
            "body_snippet": "Placement company interview",
            "score": 0.54,
        },
    ]

    ranked = rerank_candidates(analysis.query, analysis, candidates, final_k=2)

    assert ranked[0]["message_id"] == "new"
    assert ranked[0]["temporal_score"] > ranked[1]["temporal_score"]


def test_parse_email_date_handles_bad_dates():
    assert parse_email_date("Wed, 22 Jul 2026 10:25:21 GMT") is not None
    assert parse_email_date("not a real date") is None


def test_low_agreement_candidate_is_filtered():
    analysis = analyze_query("recipe for chocolate cake with walnuts")
    candidates = [
        {
            "message_id": "irrelevant",
            "chunk_id": 0,
            "subject": "Campus events",
            "sender_name": "Events",
            "sender_email": "events@example.com",
            "date": "2026-07-07",
            "text": "Subject: Campus events\nBody: Club updates",
            "body_snippet": "Club updates",
            "score": 0.37,
        }
    ]

    ranked = rerank_candidates(analysis.query, analysis, candidates, final_k=5)

    assert ranked == []

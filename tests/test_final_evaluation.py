"""
MailMind Final Pass Regression Tests.

Validates:
1. Safe temporal parsing and deterministic clocks (missing/malformed date returns None).
2. Date-constraint-aware freshness prior and month/year routing.
3. Precision sender extraction and hierarchical sender matching (email > display name > body).
4. Security alert distinct message preservation under deduplication.
5. Simplified retrieval presentation metrics and honest status calculation.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query_analyzer import analyze_query, _extract_senders
from rag.reranker import (
    parse_email_date,
    _sender_score,
    _date_constraint_score,
    rerank_candidates,
    _base_score,
)
from evaluation.reporter import get_target_status


def test_temporal_parsing_safety():
    """Verify parse_email_date handles valid dates, timezones, and never defaults to now()."""
    # 1. Valid RFC2822
    dt1 = parse_email_date("Tue, 21 Jul 2026 18:41:00 +0530")
    assert dt1 is not None
    assert dt1.year == 2026 and dt1.month == 7 and dt1.day == 21
    assert dt1.tzinfo == timezone.utc

    # 2. Valid ISO
    dt2 = parse_email_date("2026-07-21T18:41:00Z")
    assert dt2 is not None
    assert dt2.year == 2026 and dt2.month == 7 and dt2.day == 21

    # 3. None input MUST return None, NEVER current timestamp
    assert parse_email_date(None) is None

    # 4. Empty/whitespace input MUST return None
    assert parse_email_date("") is None
    assert parse_email_date("   ") is None

    # 5. Malformed string MUST return None without raising
    assert parse_email_date("not-a-valid-date-string-at-all") is None


def test_sender_extraction_precision():
    """Verify sender extraction correctly isolates senders from query topic text."""
    cases = [
        ("emails from CDC about placement registration", "CDC"),
        ("Helpdesk CDC emails regarding 2027 batch queries", "Helpdesk CDC"),
        ("Emails from Director Physical Education sports achievements", "Director Physical Education"),
        ("Emails from Dean Academic Research journal publications", "Dean Academic Research"),
        ("urgent notice from cdc@vit.ac.in regarding test", "cdc@vit.ac.in"),
    ]
    for query, expected_sender in cases:
        senders = _extract_senders(query)
        assert len(senders) >= 1
        assert senders[0] == expected_sender


def test_sender_scoring_hierarchy():
    """Verify sender matching prioritizes email address, then display name, with weak body fallback."""
    # Exact email match
    cand_email = {
        "sender_name": "CDC Office",
        "sender_email": "cdc@vit.ac.in",
        "body_snippet": "Placement circular",
    }
    score_email = _sender_score(["cdc@vit.ac.in"], cand_email, [])
    assert score_email == 1.0

    # Display name match
    cand_name = {
        "sender_name": "Director Physical Education",
        "sender_email": "sports@vit.ac.in",
        "body_snippet": "Sports achievements circular",
    }
    score_name = _sender_score(["Director Physical Education"], cand_name, [])
    assert score_name >= 0.90

    # Body-text only mention (must NOT be treated as primary sender match)
    cand_body_only = {
        "sender_name": "Student Council",
        "sender_email": "studentcouncil@vit.ac.in",
        "body_snippet": "We have forwarded queries to Director Physical Education yesterday.",
    }
    score_body = _sender_score(["Director Physical Education"], cand_body_only, [])
    assert score_body <= 0.20  # Weak supporting evidence only

    # Completely unrelated sender
    cand_unrelated = {
        "sender_name": "Library Notice",
        "sender_email": "library@vit.ac.in",
        "body_snippet": "Books returned notice.",
    }
    score_none = _sender_score(["Director Physical Education"], cand_unrelated, [])
    assert score_none == 0.0


def test_date_constraint_scoring():
    """Verify date constraint scoring rewards matching dates and penalizes violations."""
    # Low semantic candidate lacking requested date token
    cand_mismatch = {
        "subject": "General Meeting",
        "body_snippet": "Meeting details for everyone.",
    }
    penalty = _date_constraint_score(["1994"], cand_mismatch, semantic=0.30)
    assert penalty < 0.0

    # Candidate with matching date token in subject
    cand_match = {
        "subject": "World Cup 1994 Highlights",
        "body_snippet": "Tournament summary.",
    }
    bonus = _date_constraint_score(["1994"], cand_match, semantic=0.30)
    assert bonus > 0.0


def test_security_distinct_alerts_preserved():
    """Verify chunk-to-message deduplication does not collapse distinct security messages."""
    cands = [
        # Chunk 1 of Message A
        {
            "message_id": "msg_sec_001",
            "subject": "Security alert for Google account",
            "sender_name": "Google",
            "sender_email": "no-reply@accounts.google.com",
            "date": "Tue, 21 Jul 2026 10:00:00 +0530",
            "score": 0.70,
            "body_snippet": "New sign-in from Windows device in Chennai.",
        },
        # Chunk 2 of Message A (same message, lower score)
        {
            "message_id": "msg_sec_001",
            "subject": "Security alert for Google account",
            "sender_name": "Google",
            "sender_email": "no-reply@accounts.google.com",
            "date": "Tue, 21 Jul 2026 10:00:00 +0530",
            "score": 0.50,
            "body_snippet": "Check activity at google.com/security.",
        },
        # Chunk 1 of Message B (distinct message with same subject)
        {
            "message_id": "msg_sec_002",
            "subject": "Security alert for Google account",
            "sender_name": "Google",
            "sender_email": "no-reply@accounts.google.com",
            "date": "Wed, 22 Jul 2026 15:30:00 +0530",
            "score": 0.68,
            "body_snippet": "Password changed from unknown location.",
        },
    ]

    analysis = analyze_query("Security alert for Google account")
    reranked = rerank_candidates("Security alert for Google account", analysis, cands, final_k=10, deduplicate=True)

    # Both distinct messages must be preserved!
    mids = [c["message_id"] for c in reranked]
    assert "msg_sec_001" in mids
    assert "msg_sec_002" in mids
    assert len(mids) == 2  # Chunk 2 was deduped, but distinct message was kept!


def test_honest_target_statuses():
    """Verify get_target_status computes honest statuses from actual values."""
    assert get_target_status(0.85, 0.80, ">=") == "PASS"
    assert get_target_status(0.75, 0.80, ">=") == "BELOW TARGET"
    assert get_target_status(0.00, 0.00, "==") == "PASS"
    assert get_target_status(0.05, 0.00, "==") == "BELOW TARGET"
    assert get_target_status(None, 0.80) == "NOT EVALUATED"
    assert get_target_status(0.50, 0.80, not_evaluated=True) == "NOT EVALUATED"

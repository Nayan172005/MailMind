"""Tests for rag.query_analyzer."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query_analyzer import (
    QueryType,
    TemporalIntent,
    analyze_query,
)


def test_analyze_entity_question():
    analysis = analyze_query("When is my Microsoft assessment?")

    assert analysis.query_type == QueryType.ENTITY_SEARCH
    assert "Microsoft" in analysis.entities
    assert "assessment" in analysis.subject_terms


def test_analyze_temporal_query():
    analysis = analyze_query("most recent placement company")

    assert analysis.query_type == QueryType.TEMPORAL_SEARCH
    assert analysis.temporal_intent == TemporalIntent.RECENT
    assert "placement" in analysis.subject_terms


def test_analyze_summary_query():
    analysis = analyze_query("summarize NPTEL registration emails")

    assert analysis.query_type == QueryType.ENTITY_SEARCH
    assert "NPTEL" in analysis.entities
    assert "registration" in analysis.subject_terms


def test_analyze_sender_query():
    analysis = analyze_query("Emails from Helpdesk CDC regarding 2027 batch queries")

    assert "Helpdesk CDC" in analysis.senders
    assert any("2027" in t for t in analysis.date_tokens)
    assert "queries" in analysis.subject_terms


def test_analyze_email_address_sender():
    analysis = analyze_query("Show notices from placement@vit.ac.in")

    assert "placement@vit.ac.in" in analysis.senders


def test_analyze_venue_extraction():
    analysis = analyze_query("Where is the exam venue in PRP 717 or Anna Auditorium?")

    assert any("PRP 717" in v or "PRP" in v for v in analysis.venues)
    assert any("Anna Auditorium" in v for v in analysis.venues)


def test_analyze_broad_search():
    analysis = analyze_query("Find all emails about NPTEL SWAYAM course registration deadlines")

    assert analysis.is_broad_search is True
    assert "NPTEL" in analysis.entities
    assert "SWAYAM" in analysis.entities


def test_analyze_conversational_verbs_not_entities():
    analysis = analyze_query("Find circular regarding School of Computer Science and Engineering (SCOPE) symposium")

    assert "Find" not in analysis.entities
    assert "circular" not in analysis.entities
    assert "SCOPE" in analysis.entities


def test_analyze_lowercase_entity():
    analysis = analyze_query("when is the test for the internship at wakefit")

    # Should detect wakefit contextually after 'at'
    assert any("wakefit" in e.lower() for e in analysis.entities)


def test_analyze_date_tokens():
    analysis = analyze_query("emails mentioning the July 2026 internship announcements")

    assert any("july 2026" in dt.lower() for dt in analysis.date_tokens)

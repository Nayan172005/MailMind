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

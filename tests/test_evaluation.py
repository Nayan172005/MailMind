"""
Unit and integration tests for the MailMind evaluation suite.
"""

import pytest
from pathlib import Path

from evaluation.benchmark import load_benchmark, check_coverage, BenchmarkQuery, RelevanceJudgment
from evaluation.retrieval_eval import (
    calculate_ndcg_at_k,
    evaluate_query,
    aggregate_query_results,
    SearchResult,
)
from evaluation.rag_eval import check_abstention, deterministic_evaluate_answer
from evaluation.classifier_eval import load_held_out_test_split


def test_benchmark_loading():
    dataset = load_benchmark()
    assert len(dataset.queries) == 50
    assert dataset.splits["test"] == 36
    assert dataset.splits["dev"] == 14

    # Verify split filtering
    test_queries = dataset.filter_split("test")
    assert len(test_queries) == 36
    dev_queries = dataset.filter_split("dev")
    assert len(dev_queries) == 14


def test_coverage_check():
    dataset = load_benchmark()
    coverage = check_coverage(dataset)
    assert coverage["status"] == "ok"
    assert coverage["missing_ids_count"] == 0
    assert coverage["indexed_emails"] > 10000


def test_ndcg_calculation():
    # Perfect ranking of grades [2, 1, 0]
    perfect_ndcg = calculate_ndcg_at_k([2, 1, 0], [2, 1], k=3)
    assert abs(perfect_ndcg - 1.0) < 1e-4

    # Inverted ranking [1, 2, 0] should have lower nDCG than 1.0
    inverted_ndcg = calculate_ndcg_at_k([1, 2, 0], [2, 1], k=3)
    assert 0.0 < inverted_ndcg < 1.0

    # Completely irrelevant results
    zero_ndcg = calculate_ndcg_at_k([0, 0, 0], [2, 1], k=3)
    assert zero_ndcg == 0.0


def test_evaluate_query_metrics():
    q = BenchmarkQuery(
        id="q_test",
        query="test query",
        category="placement",
        split="test",
        relevance_judgments={
            "m1": RelevanceJudgment(grade=2, reason="Direct"),
            "m2": RelevanceJudgment(grade=1, reason="Partial"),
        },
    )

    results = [
        SearchResult(
            message_id="m1",
            subject="Subj 1",
            sender_name="Sender",
            sender_email="s@vit.ac.in",
            date="2026-07-22",
            score=0.9,
            snippet="Snippet",
        ),
        SearchResult(
            message_id="m_irr",
            subject="Subj 2",
            sender_name="Sender",
            sender_email="s@vit.ac.in",
            date="2026-07-22",
            score=0.7,
            snippet="Snippet",
        ),
    ]

    res = evaluate_query(q, results)
    assert res.recall_at_5 == 0.5  # 1 of 2 found
    assert res.precision_at_5 == 0.2  # 1 of 5
    assert res.mrr == 1.0  # first item is relevant
    assert res.success_at_5 == 1.0


def test_rag_abstention_detection():
    assert check_abstention("I couldn't find any relevant emails in your mailbox.") is True
    assert check_abstention("Here is your Microsoft assessment schedule:") is False


def test_classifier_held_out_split():
    X_test, y_test, le = load_held_out_test_split()
    assert len(X_test) == 542
    assert len(y_test) == 542
    # Verify no overlap in indices with training
    assert set(le.classes_) == {"general", "important", "spam"}

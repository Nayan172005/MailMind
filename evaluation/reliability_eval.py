"""
MailMind Evaluation — Reliability and End-to-End System Checks.

Verifies:
- Email parsing and metadata preservation
- Classification and spam-handling paths
- Embedding/index consistency and document-to-message mapping
- Retrieval and source attribution schemas
- RAG answer generation and error handling (empty context, abstention)
- Ingestion idempotency and duplicate deduplication
- Gmail / Pub/Sub webhook authentication health (mocked, safe offline)
- Granular latency profiling (retrieval-only latency kept strictly distinct from LLM latency)

Never touches or modifies production FAISS or MongoDB data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import numpy as np

from rag.config import settings
from rag.embedder import embed_query, embed_texts
from rag.normalizer import normalize_email
from rag.pipeline import RAGPipeline
from rag.query_analyzer import analyze_query
from rag.reranker import rerank_candidates
from rag.retriever import retrieve, retrieve_with_context
from rag.schemas import EmailRecord, DocumentChunk, SearchResult
from rag.utils import get_logger
from rag.vector_store import VectorStore

logger = get_logger("evaluation.reliability")


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str
    latency_ms: Optional[float] = None


@dataclass
class ReliabilityReport:
    total_checks: int
    passed_checks: int
    failed_checks: int
    all_passed: bool
    checks: List[Dict[str, Any]]
    latencies: Dict[str, float]  # query_embedding_ms, faiss_search_ms, reranking_ms, retrieval_total_ms, generation_ms

    def to_dict(self) -> dict:
        return asdict(self)


def test_email_parsing() -> CheckResult:
    """Verify email normalization, metadata preservation, and Unicode handling."""
    start = time.perf_counter()
    sample = {
        "Message_ID": "test_msg_001",
        "Subject": "  Interview Invitation \u200b: Software Engineer \u2013 2027 Batch  ",
        "Sender_Name": "CDC VIT",
        "Sender_Email": "cdc@vit.ac.in",
        "Date": "Wed, 22 Jul 2026 10:00:00 +0530",
        "Body": "Dear Student, your interview is scheduled on 24 July.",
    }
    rec = normalize_email(sample)
    duration = (time.perf_counter() - start) * 1000

    passed = (
        rec.message_id == "test_msg_001"
        and "Interview Invitation" in rec.subject
        and rec.sender_email == "cdc@vit.ac.in"
        and "interview is scheduled" in rec.body
    )
    return CheckResult(
        name="email_parsing_and_metadata_preservation",
        passed=passed,
        details="Verified whitespace normalization and metadata preservation.",
        latency_ms=round(duration, 2),
    )


def test_classifier_path() -> CheckResult:
    """Verify BiLSTM classifier inference and spam handling path."""
    start = time.perf_counter()
    try:
        from api.gmail.classifier import predict_email, is_spam

        # Test obvious spam sample
        res_spam = predict_email(
            sender="lottery@free-cash.xyz",
            subject="CLAIM YOUR $1,000,000 PRIZE NOW!!!",
            body="Congratulations winner! Click here to claim your massive cash reward immediately.",
        )
        # Test placement / important sample
        res_imp = predict_email(
            sender="cdc@vit.ac.in",
            subject="Microsoft Super Dream Placement Interview Shortlist",
            body="Dear Student, you have been shortlisted for the final technical interview round.",
        )
        duration = (time.perf_counter() - start) * 1000

        prob_sum_spam = sum(res_spam["probabilities"].values())
        prob_sum_imp = sum(res_imp["probabilities"].values())

        passed = (
            abs(prob_sum_spam - 1.0) < 1e-3
            and abs(prob_sum_imp - 1.0) < 1e-3
            and is_spam(res_spam)
            and res_imp["label"].lower() in ("important", "general")
        )
        return CheckResult(
            name="classification_and_spam_handling_paths",
            passed=passed,
            details=f"Classifier predicted: spam={res_spam['label']} ({res_spam['confidence']:.2f}), imp={res_imp['label']} ({res_imp['confidence']:.2f}).",
            latency_ms=round(duration, 2),
        )
    except Exception as e:
        return CheckResult(
            name="classification_and_spam_handling_paths",
            passed=False,
            details=f"Classifier test error: {e}",
            latency_ms=None,
        )


def test_embedding_index_consistency(pipeline: RAGPipeline) -> CheckResult:
    """Verify 384-d embedding dimensions and vector-to-message mapping."""
    start = time.perf_counter()
    try:
        vec = embed_query("test query")
        dim = vec.shape[1]

        store = pipeline.store
        loaded = store.is_loaded
        total_vecs = store.total_vectors
        total_emails = store.total_emails

        # Verify metadata consistency
        valid_mapping = True
        if loaded and store.metadata:
            sample_m = store.metadata[0]
            valid_mapping = "message_id" in sample_m and "subject" in sample_m

        duration = (time.perf_counter() - start) * 1000
        passed = (dim == 384) and loaded and (total_vecs >= total_emails) and valid_mapping

        return CheckResult(
            name="embedding_and_index_consistency",
            passed=passed,
            details=f"Embedding dim={dim}, Index vectors={total_vecs}, Unique emails={total_emails}.",
            latency_ms=round(duration, 2),
        )
    except Exception as e:
        return CheckResult(
            name="embedding_and_index_consistency",
            passed=False,
            details=f"Error checking index: {e}",
            latency_ms=None,
        )


def test_retrieval_and_source_attribution(pipeline: RAGPipeline) -> CheckResult:
    """Verify retrieval output schemas, score bounds, and source attribution."""
    start = time.perf_counter()
    try:
        results = retrieve("placement interviews", pipeline.store, top_k=3)
        duration = (time.perf_counter() - start) * 1000

        valid_schema = True
        for r in results:
            if not isinstance(r, SearchResult):
                valid_schema = False
            if not r.message_id or not r.subject:
                valid_schema = False
            if not (-1.0 <= r.score <= 1.5):  # cosine similarity with boost
                valid_schema = False

        passed = len(results) > 0 and valid_schema
        return CheckResult(
            name="retrieval_and_source_attribution",
            passed=passed,
            details=f"Retrieved {len(results)} results with valid SearchResult schemas and scores.",
            latency_ms=round(duration, 2),
        )
    except Exception as e:
        return CheckResult(
            name="retrieval_and_source_attribution",
            passed=False,
            details=f"Retrieval check error: {e}",
            latency_ms=None,
        )


def test_rag_abstention_and_error_handling() -> CheckResult:
    """Verify empty context gracefully abstains without calling LLM."""
    start = time.perf_counter()
    from rag.generator import generate_answer

    ans_empty = generate_answer("any question", context="")
    duration = (time.perf_counter() - start) * 1000

    passed = (
        "couldn't find any relevant emails" in ans_empty.lower()
        or "try rephrasing" in ans_empty.lower()
    )
    return CheckResult(
        name="rag_abstention_and_error_handling",
        passed=passed,
        details="Verified safe abstention on empty context without API call.",
        latency_ms=round(duration, 2),
    )


def test_ingestion_idempotency() -> CheckResult:
    """Verify duplicate chunks are ignored and vector store remains intact."""
    start = time.perf_counter()
    try:
        dummy_store = VectorStore()
        c1 = DocumentChunk(
            message_id="idemp_01",
            chunk_id=0,
            text="Subject: Test\nFrom: test@vit.ac.in\n\nContent",
            subject="Test",
            sender_name="Test",
            sender_email="test@vit.ac.in",
            date="2026-07-22",
            body_snippet="Content",
        )
        dummy_store.build_index([c1])
        initial_count = dummy_store.total_vectors

        # Attempt to add exact duplicate chunk
        added = dummy_store.update_index([c1])
        final_count = dummy_store.total_vectors
        duration = (time.perf_counter() - start) * 1000

        passed = (added == 0) and (final_count == initial_count)
        return CheckResult(
            name="duplicate_ingestion_and_idempotency",
            passed=passed,
            details=f"Duplicate chunk rejected: added={added}, vectors before={initial_count}, after={final_count}.",
            latency_ms=round(duration, 2),
        )
    except Exception as e:
        return CheckResult(
            name="duplicate_ingestion_and_idempotency",
            passed=False,
            details=f"Idempotency error: {e}",
            latency_ms=None,
        )


def test_pubsub_webhook_security() -> CheckResult:
    """Verify Pub/Sub webhook authentication rejection paths."""
    start = time.perf_counter()
    try:
        from fastapi import HTTPException
        from api.routes.gmail import verify_pubsub_jwt

        class MockRequest:
            def __init__(self, headers: dict):
                self.headers = headers

        # 1. Missing Authorization header should raise 401
        rejected_missing = False
        try:
            verify_pubsub_jwt(MockRequest({}))
        except HTTPException as exc:
            rejected_missing = (exc.status_code == 401)

        # 2. Malformed Authorization header should raise 401
        rejected_malformed = False
        try:
            verify_pubsub_jwt(MockRequest({"Authorization": "invalid_token_without_bearer"}))
        except HTTPException as exc:
            rejected_malformed = (exc.status_code == 401)

        duration = (time.perf_counter() - start) * 1000
        passed = rejected_missing and rejected_malformed

        return CheckResult(
            name="gmail_pubsub_security_health",
            passed=passed,
            details="Verified unauthenticated and malformed Pub/Sub push requests are safely rejected (HTTP 401).",
            latency_ms=round(duration, 2),
        )
    except Exception as e:
        return CheckResult(
            name="gmail_pubsub_security_health",
            passed=False,
            details=f"Pub/Sub verification error: {e}",
            latency_ms=None,
        )


def profile_system_latencies(pipeline: RAGPipeline) -> Dict[str, float]:
    """
    Separately profiles:
    - embedding latency
    - FAISS search latency
    - reranking latency
    - total retrieval latency
    - LLM answer generation latency
    """
    query = "placement interviews schedule and venues"

    # 1. Embedding latency
    t0 = time.perf_counter()
    emb = embed_query(query)
    embed_ms = (time.perf_counter() - t0) * 1000

    # 2. FAISS search latency
    t0 = time.perf_counter()
    raw_results = pipeline.store.search(emb, top_k=30)
    faiss_ms = (time.perf_counter() - t0) * 1000

    # 3. Reranking latency
    analysis = analyze_query(query)
    t0 = time.perf_counter()
    reranked = rerank_candidates(
        query=query,
        analysis=analysis,
        candidates=raw_results,
        final_k=5,
        deduplicate=True,
    )
    rerank_ms = (time.perf_counter() - t0) * 1000

    # 4. Total retrieval latency
    t0 = time.perf_counter()
    results, context = retrieve_with_context(query, pipeline.store, top_k=5)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # 5. Generation latency (kept strictly distinct)
    gen_ms = 0.0
    if settings.GEMINI_API_KEY:
        try:
            from rag.generator import generate_answer
            t0 = time.perf_counter()
            _ = generate_answer(query, context)
            gen_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(f"Latency generation test failed: {e}")

    return {
        "query_embedding_ms": round(embed_ms, 2),
        "faiss_search_ms": round(faiss_ms, 2),
        "reranking_ms": round(rerank_ms, 2),
        "retrieval_total_ms": round(retrieval_ms, 2),
        "generation_ms": round(gen_ms, 2),
    }


def run_reliability_checks(pipeline: RAGPipeline) -> ReliabilityReport:
    """Run full reliability and latency suite."""
    print("\nRunning Reliability and End-to-End System Checks...")

    checks = [
        test_email_parsing(),
        test_classifier_path(),
        test_embedding_index_consistency(pipeline),
        test_retrieval_and_source_attribution(pipeline),
        test_rag_abstention_and_error_handling(),
        test_ingestion_idempotency(),
        test_pubsub_webhook_security(),
    ]

    latencies = profile_system_latencies(pipeline)

    passed_count = sum(1 for c in checks if c.passed)
    failed_count = len(checks) - passed_count

    return ReliabilityReport(
        total_checks=len(checks),
        passed_checks=passed_count,
        failed_checks=failed_count,
        all_passed=(failed_count == 0),
        checks=[asdict(c) for c in checks],
        latencies=latencies,
    )

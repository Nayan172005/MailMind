"""
MailMind Evaluation — Retrieval Metrics Engine.

Calculates macro-averaged retrieval metrics across eligible queries:
- Precision@5, Precision@10
- Recall@5, Recall@10, Recall@20 (binary convention: grades 1 and 2)
- MRR (Mean Reciprocal Rank)
- nDCG@10 (using graded relevance: 2=directly relevant, 1=partially relevant, 0=irrelevant)
- Success@5, Success@10 (Hit@K)
- False-positive analysis for out-of-domain / negative queries

Compares legacy retrieval vs improved production retrieval on the exact same benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

from rag.config import settings
from rag.embedder import embed_query
from rag.pipeline import RAGPipeline
from rag.retriever import retrieve
from rag.schemas import SearchResult
from rag.utils import get_logger, truncate_text
from evaluation.benchmark import BenchmarkQuery, BenchmarkDataset

logger = get_logger("evaluation.retrieval")


@dataclass
class QueryRetrievalResult:
    query_id: str
    query_text: str
    category: str
    split: str
    is_negative: bool
    retrieved_message_ids: List[str]
    retrieved_scores: List[float]
    retrieved_grades: List[int]
    top_subject: str
    precision_at_5: Optional[float]
    precision_at_10: Optional[float]
    recall_at_5: Optional[float]
    recall_at_10: Optional[float]
    recall_at_20: Optional[float]
    mrr: Optional[float]
    ndcg_at_10: Optional[float]
    success_at_5: Optional[float]
    success_at_10: Optional[float]
    false_positive_hit: bool  # for negative queries: True if docs returned above threshold
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregateMetrics:
    mode: str  # "legacy" or "improved"
    total_queries: int
    eligible_queries: int
    negative_queries: int
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float
    success_at_5: float
    success_at_10: float
    negative_fp_rate: float
    avg_negative_max_score: float
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_ndcg_at_k(retrieved_grades: List[int], ideal_grades: List[int], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank K.
    DCG = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    top_grades = retrieved_grades[:k]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(top_grades))

    sorted_ideal = sorted(ideal_grades, reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted_ideal))

    if idcg <= 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def _legacy_search(pipeline: RAGPipeline, query: str, top_k: int) -> List[SearchResult]:
    """
    Reproduces the legacy semantic-only retrieval path:
    Embed query -> FAISS top_k*3 -> SIMILARITY_THRESHOLD filter -> message_id deduplication.
    """
    query_embedding = embed_query(query)
    raw_results = pipeline.store.search(query_embedding, top_k=top_k * 3)
    filtered = [r for r in raw_results if r["score"] >= settings.SIMILARITY_THRESHOLD]

    seen: dict[str, dict] = {}
    for result in filtered:
        message_id = result.get("message_id", "")
        if not message_id:
            continue
        if message_id not in seen or result["score"] > seen[message_id]["score"]:
            seen[message_id] = result

    deduped = sorted(seen.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            message_id=item["message_id"],
            subject=item.get("subject", ""),
            sender_name=item.get("sender_name", ""),
            sender_email=item.get("sender_email", ""),
            date=item.get("date", ""),
            score=round(float(item.get("score", 0.0)), 4),
            snippet=truncate_text(item.get("body_snippet", item.get("text", "")), 300),
        )
        for item in deduped
    ]


def _improved_search(pipeline: RAGPipeline, query: str, top_k: int) -> List[SearchResult]:
    """
    Production improved retrieval:
    Query analysis -> deeper candidate pool -> SIMILARITY_THRESHOLD -> metadata-aware reranking -> dedup.
    """
    return retrieve(query, pipeline.store, top_k=top_k)


def evaluate_query(
    query_spec: BenchmarkQuery,
    results_20: List[SearchResult],
) -> QueryRetrievalResult:
    """Evaluate a single query's retrieved results up to top 20."""
    retrieved_ids = [r.message_id for r in results_20]
    retrieved_scores = [float(r.score) for r in results_20]

    # Map retrieved IDs to graded relevance
    retrieved_grades = [
        query_spec.relevance_judgments.get(mid).grade
        if mid in query_spec.relevance_judgments else 0
        for mid in retrieved_ids
    ]

    is_negative = query_spec.is_negative_query
    top_subject = results_20[0].subject if results_20 else ""

    if is_negative:
        # Negative / out-of-domain query evaluation:
        # Success means abstaining/returning no documents above threshold.
        # Any retrieved document above threshold is a False Positive.
        fp_hit = len(results_20) > 0
        failure = "Retrieved irrelevant documents for negative query" if fp_hit else None
        return QueryRetrievalResult(
            query_id=query_spec.id,
            query_text=query_spec.query,
            category=query_spec.category,
            split=query_spec.split,
            is_negative=True,
            retrieved_message_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            retrieved_grades=retrieved_grades,
            top_subject=top_subject,
            precision_at_5=None,
            precision_at_10=None,
            recall_at_5=None,
            recall_at_10=None,
            recall_at_20=None,
            mrr=None,
            ndcg_at_10=None,
            success_at_5=None,
            success_at_10=None,
            false_positive_hit=fp_hit,
            failure_reason=failure,
        )

    # Eligible query with ground truth relevant items
    relevant_ids = query_spec.all_relevant_ids  # grades 1 & 2
    total_relevant = len(relevant_ids)

    # Precision@K: relevant retrieved / K
    def p_at_k(k: int) -> float:
        top_ids = retrieved_ids[:k]
        if not top_ids:
            return 0.0
        rel_count = sum(1 for mid in top_ids if mid in relevant_ids)
        return rel_count / k

    # Recall@K: relevant retrieved / total relevant in corpus
    def r_at_k(k: int) -> float:
        if total_relevant == 0:
            return 0.0
        top_ids = set(retrieved_ids[:k])
        hits = len(top_ids & relevant_ids)
        return hits / total_relevant

    # MRR: reciprocal rank of first relevant doc
    mrr = 0.0
    for idx, mid in enumerate(retrieved_ids):
        if mid in relevant_ids:
            mrr = 1.0 / (idx + 1)
            break

    # Success@K: 1 if at least one relevant document in top K
    s_at_5 = 1.0 if any(mid in relevant_ids for mid in retrieved_ids[:5]) else 0.0
    s_at_10 = 1.0 if any(mid in relevant_ids for mid in retrieved_ids[:10]) else 0.0

    # nDCG@10
    ideal_grades = [j.grade for j in query_spec.relevance_judgments.values() if j.grade > 0]
    ndcg = calculate_ndcg_at_k(retrieved_grades, ideal_grades, k=10)

    # Precision and Recall
    precision_5 = p_at_k(5)
    precision_10 = p_at_k(10)
    recall_5 = r_at_k(5)
    recall_10 = r_at_k(10)
    recall_20 = r_at_k(20)

    # Failure diagnostics
    failure = None
    if s_at_10 == 0.0:
        failure = "Zero relevant documents retrieved in top 10"
    elif recall_10 < 1.0 and total_relevant > 1:
        missing_count = total_relevant - len(set(retrieved_ids[:10]) & relevant_ids)
        failure = f"Missed {missing_count} of {total_relevant} relevant documents at @10"

    return QueryRetrievalResult(
        query_id=query_spec.id,
        query_text=query_spec.query,
        category=query_spec.category,
        split=query_spec.split,
        is_negative=False,
        retrieved_message_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
        retrieved_grades=retrieved_grades,
        top_subject=top_subject,
        precision_at_5=precision_5,
        precision_at_10=precision_10,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        recall_at_20=recall_20,
        mrr=mrr,
        ndcg_at_10=ndcg,
        success_at_5=s_at_5,
        success_at_10=s_at_10,
        false_positive_hit=False,
        failure_reason=failure,
    )


def aggregate_query_results(
    mode: str,
    results: List[QueryRetrievalResult],
) -> AggregateMetrics:
    """Compute macro-averages over eligible queries."""
    eligible = [r for r in results if not r.is_negative]
    negatives = [r for r in results if r.is_negative]

    n_el = len(eligible) if eligible else 1

    def avg(vals: List[Optional[float]]) -> float:
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else 0.0

    # Negative query stats
    neg_fp_count = sum(1 for r in negatives if r.false_positive_hit)
    neg_fp_rate = (neg_fp_count / len(negatives)) if negatives else 0.0
    neg_scores = [max(r.retrieved_scores) for r in negatives if r.retrieved_scores]
    avg_neg_score = (sum(neg_scores) / len(neg_scores)) if neg_scores else 0.0

    # Category breakdown
    categories: Dict[str, List[QueryRetrievalResult]] = {}
    for r in eligible:
        categories.setdefault(r.category, []).append(r)

    per_category: Dict[str, Dict[str, float]] = {}
    for cat, items in categories.items():
        per_category[cat] = {
            "count": len(items),
            "recall@5": avg([i.recall_at_5 for i in items]),
            "recall@10": avg([i.recall_at_10 for i in items]),
            "precision@5": avg([i.precision_at_5 for i in items]),
            "precision@10": avg([i.precision_at_10 for i in items]),
            "mrr": avg([i.mrr for i in items]),
            "ndcg@10": avg([i.ndcg_at_10 for i in items]),
            "success@10": avg([i.success_at_10 for i in items]),
        }

    return AggregateMetrics(
        mode=mode,
        total_queries=len(results),
        eligible_queries=len(eligible),
        negative_queries=len(negatives),
        precision_at_5=avg([r.precision_at_5 for r in eligible]),
        precision_at_10=avg([r.precision_at_10 for r in eligible]),
        recall_at_5=avg([r.recall_at_5 for r in eligible]),
        recall_at_10=avg([r.recall_at_10 for r in eligible]),
        recall_at_20=avg([r.recall_at_20 for r in eligible]),
        mrr=avg([r.mrr for r in eligible]),
        ndcg_at_10=avg([r.ndcg_at_10 for r in eligible]),
        success_at_5=avg([r.success_at_5 for r in eligible]),
        success_at_10=avg([r.success_at_10 for r in eligible]),
        negative_fp_rate=neg_fp_rate,
        avg_negative_max_score=avg_neg_score,
        per_category=per_category,
    )


def run_retrieval_evaluation(
    pipeline: RAGPipeline,
    dataset: BenchmarkDataset,
    ks: List[int] = [5, 10, 20],
) -> dict:
    """
    Run full retrieval evaluation comparing legacy and improved retrievers.
    """
    max_k = max(ks)
    modes = ["legacy", "improved"]
    mode_results: Dict[str, List[QueryRetrievalResult]] = {"legacy": [], "improved": []}
    aggregates: Dict[str, AggregateMetrics] = {}

    print(f"\nEvaluating Retrieval on {len(dataset.queries)} queries (Max K={max_k})...")

    for mode in modes:
        search_fn = _legacy_search if mode == "legacy" else _improved_search
        results = []

        for q in dataset.queries:
            retrieved = search_fn(pipeline, q.query, top_k=max_k)
            q_res = evaluate_query(q, retrieved)
            results.append(q_res)

        mode_results[mode] = results
        aggregates[mode] = aggregate_query_results(mode, results)

    return {
        "benchmark_version": dataset.version,
        "total_queries": len(dataset.queries),
        "eligible_queries": aggregates["improved"].eligible_queries,
        "negative_queries": aggregates["improved"].negative_queries,
        "legacy_metrics": aggregates["legacy"].to_dict(),
        "improved_metrics": aggregates["improved"].to_dict(),
        "per_query": {
            "legacy": [r.to_dict() for r in mode_results["legacy"]],
            "improved": [r.to_dict() for r in mode_results["improved"]],
        },
    }

"""
MailMind RAG — Evaluation Script

Evaluates retrieval quality using a set of manually-defined queries.
Computes:
- Recall@K: whether relevant results appear in top-K
- Keyword hit rate: whether expected keywords appear in retrieved results
- Empty retrieval correctness: whether irrelevant queries return no results

Usage:
    python evaluation/evaluate.py
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline
from rag.utils import get_logger

logger = get_logger("evaluation")


def load_eval_queries(path: str = None) -> list[dict]:
    """Load evaluation queries from JSON file."""
    if path is None:
        path = Path(__file__).parent / "eval_queries.json"
    else:
        path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_keyword_hit(results: list, expected_keywords: list[str]) -> bool:
    """Check if any expected keyword appears in the retrieved results."""
    if not expected_keywords:
        return True  # No keywords to check

    # Combine all result text for keyword matching
    combined_text = ""
    for r in results:
        combined_text += f" {r.subject} {r.snippet} {r.sender_name}".lower()

    for keyword in expected_keywords:
        if keyword.lower() in combined_text:
            return True

    return False


def evaluate(pipeline: RAGPipeline, top_k: int = 5) -> dict:
    """
    Run evaluation and return metrics.

    Returns dict with:
        - queries: list of per-query results
        - recall_at_k: float
        - keyword_hit_rate: float
        - empty_retrieval_accuracy: float
    """
    queries = load_eval_queries()

    results_log = []
    recall_hits = 0
    keyword_hits = 0
    empty_correct = 0
    total_relevant = 0
    total_irrelevant = 0

    print("\n" + "=" * 70)
    print(f"  MailMind RAG Evaluation — {len(queries)} queries, top_k={top_k}")
    print("=" * 70)

    for i, q in enumerate(queries, 1):
        query = q["query"]
        expected_keywords = q.get("expected_keywords", [])
        category = q.get("category", "unknown")
        expect_not_found = q.get("expect_not_found", False)

        # Search
        response = pipeline.search(query, top_k=top_k)
        search_results = response.results
        has_results = len(search_results) > 0

        # Evaluate
        if expect_not_found:
            total_irrelevant += 1
            is_correct = not has_results or (
                has_results and search_results[0].score < 0.35
            )
            if is_correct:
                empty_correct += 1
            status = "✅ CORRECT (no results)" if is_correct else "❌ FALSE POSITIVE"
        else:
            total_relevant += 1
            keyword_hit = check_keyword_hit(search_results, expected_keywords)
            if keyword_hit:
                keyword_hits += 1
            if has_results:
                recall_hits += 1
            status = "✅ HIT" if (has_results and keyword_hit) else (
                "⚠️ PARTIAL" if has_results else "❌ MISS"
            )

        # Log
        top_result = search_results[0] if search_results else None
        print(f"\n  [{i:2d}] {query}")
        print(f"       Category: {category} | Status: {status}")
        if top_result:
            print(f"       Top result: {top_result.subject} (score: {top_result.score})")
        else:
            print(f"       No results returned")

        results_log.append({
            "query": query,
            "category": category,
            "status": status,
            "num_results": len(search_results),
            "top_score": top_result.score if top_result else 0.0,
            "top_subject": top_result.subject if top_result else "",
        })

    # Compute metrics
    recall_at_k = recall_hits / total_relevant if total_relevant > 0 else 0.0
    keyword_rate = keyword_hits / total_relevant if total_relevant > 0 else 0.0
    empty_accuracy = empty_correct / total_irrelevant if total_irrelevant > 0 else 1.0

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Recall@{top_k}:                {recall_at_k:.2%} ({recall_hits}/{total_relevant})")
    print(f"  Keyword Hit Rate:          {keyword_rate:.2%} ({keyword_hits}/{total_relevant})")
    print(f"  Empty Retrieval Accuracy:  {empty_accuracy:.2%} ({empty_correct}/{total_irrelevant})")
    print(f"\n  Total queries:    {len(queries)}")
    print(f"  Relevant queries: {total_relevant}")
    print(f"  Irrelevant queries: {total_irrelevant}")
    print("=" * 70)

    return {
        "queries": results_log,
        "recall_at_k": recall_at_k,
        "keyword_hit_rate": keyword_rate,
        "empty_retrieval_accuracy": empty_accuracy,
        "top_k": top_k,
    }


def main():
    """Run evaluation from command line."""
    pipeline = RAGPipeline()

    try:
        pipeline.load_index()
    except FileNotFoundError:
        print("\n  No FAISS index found. Building one first...")
        pipeline.build_index()

    metrics = evaluate(pipeline, top_k=5)

    # Save results
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()

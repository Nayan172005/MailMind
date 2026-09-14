"""
MailMind RAG retrieval evaluation.

Evaluates manually verified expected message IDs without fabricating labels.
The script compares:
- legacy: FAISS top-k + threshold + message_id deduplication
- improved: current retriever with deeper retrieval + metadata reranking

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --queries evaluation/evaluation_queries.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import settings
from rag.embedder import embed_query
from rag.pipeline import RAGPipeline
from rag.retriever import retrieve
from rag.schemas import SearchResult
from rag.utils import get_logger, truncate_text

logger = get_logger("evaluation")

DEFAULT_QUERY_PATH = Path(__file__).parent / "evaluation_queries.json"


def load_eval_queries(path: str | Path | None = None) -> list[dict]:
    query_path = Path(path) if path else DEFAULT_QUERY_PATH
    with open(query_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _legacy_search(pipeline: RAGPipeline, query: str, top_k: int) -> list[SearchResult]:
    """Reproduce the old semantic-only retrieval path for comparison."""
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


def _keyword_hit(results: list[SearchResult], expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return True
    combined = " ".join(
        f"{item.subject} {item.sender_name} {item.sender_email} {item.snippet}".lower()
        for item in results
    )
    return any(keyword.lower() in combined for keyword in expected_keywords)


def _recall(results: list[SearchResult], expected_message_ids: list[str]) -> float | None:
    if not expected_message_ids:
        return None
    retrieved = {item.message_id for item in results}
    expected = set(expected_message_ids)
    return len(retrieved & expected) / len(expected)


def evaluate(pipeline: RAGPipeline, queries: list[dict], ks: list[int]) -> dict:
    metrics = {
        "legacy": {f"recall@{k}": [] for k in ks},
        "improved": {f"recall@{k}": [] for k in ks},
        "queries": [],
    }

    print("\nMailMind RAG Retrieval Evaluation")
    print(f"Queries: {len(queries)} | K: {', '.join(str(k) for k in ks)}")
    print("-" * 80)

    for query_spec in queries:
        query = query_spec["query"]
        expected_ids = query_spec.get("expected_message_ids", [])
        expected_keywords = query_spec.get("expected_keywords", [])
        expect_not_found = query_spec.get("expect_not_found", False)

        row = {
            "query": query,
            "category": query_spec.get("category", "unknown"),
            "expected_message_ids": expected_ids,
            "legacy": {},
            "improved": {},
        }

        print(f"\nQuery: {query}")
        if expected_ids:
            print(f"Expected IDs: {', '.join(expected_ids)}")
        elif expect_not_found:
            print("Expected: little/no relevant information")

        for mode in ("legacy", "improved"):
            if mode == "legacy":
                search_fn = _legacy_search
            else:
                search_fn = lambda p, q, k: retrieve(q, p.store, top_k=k)

            mode_log = {}
            for k in ks:
                results = search_fn(pipeline, query, k)
                recall = _recall(results, expected_ids)
                keyword_hit = _keyword_hit(results, expected_keywords)
                ids = [item.message_id for item in results]

                if recall is not None:
                    metrics[mode][f"recall@{k}"].append(recall)

                mode_log[f"top_{k}"] = {
                    "message_ids": ids,
                    "recall": recall,
                    "keyword_hit": keyword_hit,
                    "top_subject": results[0].subject if results else "",
                    "top_score": results[0].score if results else 0.0,
                }

            row[mode] = mode_log

        for k in ks:
            legacy = row["legacy"][f"top_{k}"]
            improved = row["improved"][f"top_{k}"]
            print(
                f"  @{k:<2} legacy recall={_fmt_metric(legacy['recall'])} "
                f"improved recall={_fmt_metric(improved['recall'])}"
            )
        print(f"  top improved: {row['improved'][f'top_{ks[0]}']['top_subject']}")

        metrics["queries"].append(row)

    for mode in ("legacy", "improved"):
        for k in ks:
            values = metrics[mode][f"recall@{k}"]
            metrics[mode][f"recall@{k}"] = (
                sum(values) / len(values) if values else None
            )

    print("\nAggregate Recall")
    for mode in ("legacy", "improved"):
        values = [
            f"Recall@{k}={_fmt_metric(metrics[mode][f'recall@{k}'])}"
            for k in ks
        ]
        print(f"{mode:>8}: " + " | ".join(values))

    return metrics


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MailMind retrieval.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERY_PATH))
    parser.add_argument("--output", default=str(Path(__file__).parent / "eval_results.json"))
    parser.add_argument("--ks", default="5,10,20")
    args = parser.parse_args()

    ks = [int(item.strip()) for item in args.ks.split(",") if item.strip()]
    queries = load_eval_queries(args.queries)

    pipeline = RAGPipeline()
    pipeline.load_index()

    metrics = evaluate(pipeline, queries, ks)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

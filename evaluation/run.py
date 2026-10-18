"""
MailMind Evaluation — Unified One-Command Runner.

Usage:
    python -m evaluation.run --all
    python -m evaluation.run --retrieval
    python -m evaluation.run --rag
    python -m evaluation.run --classifier
    python -m evaluation.run --reliability
    python -m evaluation.run --split test --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import settings
from rag.pipeline import RAGPipeline
from rag.utils import get_logger

from evaluation.benchmark import load_benchmark, check_coverage
from evaluation.retrieval_eval import run_retrieval_evaluation
from evaluation.rag_eval import run_rag_evaluation
from evaluation.classifier_eval import run_classifier_evaluation
from evaluation.reliability_eval import run_reliability_checks
from evaluation.reporter import generate_reports

logger = get_logger("evaluation.run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MailMind End-to-End Rigorous Evaluation Framework"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all evaluations (retrieval, RAG, classifier, reliability).",
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Run retrieval benchmark (legacy vs improved).",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Run RAG generation evaluation.",
    )
    parser.add_argument(
        "--rag-all",
        action="store_true",
        help="Evaluate all benchmark queries for RAG instead of curated representative subset.",
    )
    parser.add_argument(
        "--classifier",
        action="store_true",
        help="Run BiLSTM email classifier evaluation on held-out test data.",
    )
    parser.add_argument(
        "--reliability",
        action="store_true",
        help="Run reliability and system latency checks.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "test", "dev"],
        help="Benchmark split to evaluate (default: all).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(Path(__file__).parent / "benchmark_v1.json"),
        help="Path to frozen benchmark JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "results"),
        help="Directory to store evaluation reports and dashboards.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run purely offline without external LLM API calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # If no specific flag is given, default to --all
    run_all = args.all or (not args.retrieval and not args.rag and not args.classifier and not args.reliability)
    do_retrieval = run_all or args.retrieval
    do_rag = run_all or args.rag
    do_classifier = run_all or args.classifier
    do_reliability = run_all or args.reliability

    print("\n" + "=" * 70)
    print("  MailMind End-to-End Rigorous Evaluation Suite")
    print(f"  Benchmark Split: {args.split.upper()} | Mode: {'OFFLINE' if args.offline else 'STANDARD'}")
    print("=" * 70)

    # 1. Load benchmark & verify coverage against corpus
    print("\nLoading benchmark dataset and verifying index coverage...")
    dataset = load_benchmark(args.benchmark, split=args.split)
    coverage = check_coverage(dataset)

    print(f"Loaded {len(dataset.queries)} queries for split '{args.split}'")
    print(f"Corpus check: {coverage.get('indexed_emails', 0)} indexed emails")
    if coverage.get("missing_ids_count", 0) > 0:
        print(f"WARNING: Missing ground-truth document IDs: {coverage['missing_ids']}")
    else:
        print("Coverage verification PASSED: 100% of benchmark ground-truth IDs present in corpus.")

    eval_results = {
        "benchmark_version": dataset.version,
        "dataset_name": dataset.dataset_name,
        "split_evaluated": args.split,
        "embedding_model": settings.EMBEDDING_MODEL,
        "gemini_model": settings.GEMINI_MODEL,
        "timestamp": time.time(),
        "coverage": coverage,
    }

    # Initialize RAG pipeline once for retrieval and RAG tasks
    pipeline = None
    if do_retrieval or do_rag or do_reliability:
        print("\nInitializing RAG pipeline and loading FAISS index...")
        pipeline = RAGPipeline()
        pipeline.load_index()

    # 2. Retrieval Evaluation
    if do_retrieval:
        retrieval_data = run_retrieval_evaluation(pipeline, dataset, ks=[5, 10, 20])
        eval_results["retrieval"] = retrieval_data

        imp = retrieval_data["improved_metrics"]
        leg = retrieval_data["legacy_metrics"]
        print("\n--- Retrieval Summary ---")
        print(f"  Eligible Queries: {imp['eligible_queries']} | Negative Queries: {imp['negative_queries']}")
        print(f"  Recall@10:    Improved = {imp['recall_at_10']:.2%}  |  Legacy = {leg['recall_at_10']:.2%}")
        print(f"  Precision@5:  Improved = {imp['precision_at_5']:.2%}  |  Legacy = {leg['precision_at_5']:.2%}")
        print(f"  nDCG@10:      Improved = {imp['ndcg_at_10']:.2%}  |  Legacy = {leg['ndcg_at_10']:.2%}")
        print(f"  MRR:          Improved = {imp['mrr']:.2%}  |  Legacy = {leg['mrr']:.2%}")
        print(f"  Success@10:   Improved = {imp['success_at_10']:.2%}  |  Legacy = {leg['success_at_10']:.2%}")
        print(f"  Negative FP:  Improved = {imp['negative_fp_rate']:.2%}  |  Legacy = {leg['negative_fp_rate']:.2%}")

    # 3. RAG Answer Evaluation
    if do_rag:
        use_llm = (not args.offline) and bool(settings.GEMINI_API_KEY)
        curated_ids = [
            "q01", "q03", "q06", "q10", "q11", "q13", "q16", "q17",
            "q21", "q27", "q31", "q32", "q36", "q40", "q46", "q48", "q49", "q50"
        ]
        sample_subset = None if args.rag_all else curated_ids
        rag_data = run_rag_evaluation(
            pipeline=pipeline,
            dataset=dataset,
            use_llm_judge=use_llm,
            sample_queries=sample_subset,
        )
        eval_results["rag"] = rag_data
        rag_sum = rag_data["summary"]
        print("\n--- RAG Answer Generation Summary ---")
        print(f"  Evaluated: {rag_sum['queries_evaluated']} queries (Method: {rag_sum['eval_method']})")
        print(f"  Groundedness:       {rag_sum['mean_groundedness']:.2%}")
        print(f"  Correctness:        {rag_sum['mean_correctness']:.2%}")
        print(f"  Relevance:          {rag_sum['mean_relevance']:.2%}")
        print(f"  Citation Match:     {rag_sum['mean_citation_correctness']:.2%}")
        print(f"  Abstention Acc:     {rag_sum['abstention_accuracy']:.2%}")

    # 4. Classifier Evaluation
    if do_classifier:
        classifier_data = run_classifier_evaluation()
        eval_results["classifier"] = classifier_data.to_dict()
        print("\n--- BiLSTM Classifier Summary ---")
        print(f"  Test Samples: {classifier_data.total_test_samples} (held-out)")
        print(f"  Accuracy:     {classifier_data.accuracy:.2%}")
        print(f"  Macro-F1:     {classifier_data.macro_f1:.2%}")
        print(f"  Weighted-F1:  {classifier_data.weighted_f1:.2%}")
        for cname, cmetrics in classifier_data.per_class_metrics.items():
            print(f"    [{cname:<10}] Precision={cmetrics['precision']:.2%} | Recall={cmetrics['recall']:.2%} | F1={cmetrics['f1']:.2%} (N={cmetrics['support']})")

    # 5. Reliability & Latency
    if do_reliability:
        reliability_data = run_reliability_checks(pipeline)
        eval_results["reliability"] = reliability_data.to_dict()
        print("\n--- Reliability & Latency Summary ---")
        print(f"  Checks: {reliability_data.passed_checks}/{reliability_data.total_checks} passed")
        lats = reliability_data.latencies
        print(f"  Embedding Latency:  {lats.get('query_embedding_ms', 0):.2f} ms")
        print(f"  FAISS Search:       {lats.get('faiss_search_ms', 0):.2f} ms")
        print(f"  Metadata Rerank:    {lats.get('reranking_ms', 0):.2f} ms")
        print(f"  Total Retrieval:    {lats.get('retrieval_total_ms', 0):.2f} ms")
        print(f"  Gemini Gen Latency: {lats.get('generation_ms', 0):.2f} ms")

    # 6. Generate All Reports
    print("\nGenerating evaluation artifacts...")
    artifacts = generate_reports(eval_results, output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("  Evaluation Completed Successfully!")
    print(f"  JSON Report:   {artifacts['json_report']}")
    print(f"  CSV Breakdown: {artifacts['csv_report']}")
    print(f"  Markdown Doc:  {artifacts['markdown_report']}")
    print(f"  HTML Dashboard:{artifacts['html_dashboard']}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

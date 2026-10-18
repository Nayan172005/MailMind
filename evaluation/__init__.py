"""
MailMind Evaluation Package.

Rigorous, reproducible end-to-end evaluation suite for MailMind:
- Frozen, relevance-judged retrieval benchmarks (grades 2, 1, 0)
- Precision@K, Recall@K, MRR, nDCG@K, Success@K
- RAG generation groundedness, correctness, and abstention evaluation
- BiLSTM classifier evaluation on held-out test data
- End-to-end reliability, idempotency, and latency profiling
"""

from evaluation.benchmark import (
    load_benchmark,
    check_coverage,
    BenchmarkQuery,
    BenchmarkDataset,
    RelevanceJudgment,
)
from evaluation.retrieval_eval import run_retrieval_evaluation
from evaluation.rag_eval import run_rag_evaluation
from evaluation.classifier_eval import run_classifier_evaluation
from evaluation.reliability_eval import run_reliability_checks
from evaluation.reporter import generate_reports

__all__ = [
    "load_benchmark",
    "check_coverage",
    "BenchmarkQuery",
    "BenchmarkDataset",
    "RelevanceJudgment",
    "run_retrieval_evaluation",
    "run_rag_evaluation",
    "run_classifier_evaluation",
    "run_reliability_checks",
    "generate_reports",
]

"""
MailMind Evaluation — Benchmark Management and Coverage Verification.

Manages frozen benchmarks, relevance schemas, coverage verification against
the indexed corpus, and an interactive/LLM proposal and review workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from rag.config import settings
from rag.utils import get_logger

logger = get_logger("evaluation.benchmark")

DEFAULT_BENCHMARK_PATH = Path(__file__).parent / "benchmark_v1.json"
DEFAULT_METADATA_PATH = Path("data/rag/index/metadata.json")


@dataclass
class RelevanceJudgment:
    grade: int  # 2 = directly relevant, 1 = partially relevant, 0 = irrelevant
    reason: str
    source: str = "human_reviewed"  # "initial", "llm_proposed", "human_reviewed"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkQuery:
    id: str
    query: str
    category: str
    split: str  # "test" or "dev"
    expected_message_ids: List[str] = field(default_factory=list)
    relevance_judgments: Dict[str, RelevanceJudgment] = field(default_factory=dict)
    reference_answer: Optional[str] = None
    key_evidence: List[str] = field(default_factory=list)
    expect_abstention: bool = False

    @property
    def directly_relevant_ids(self) -> set[str]:
        return {
            mid for mid, j in self.relevance_judgments.items()
            if j.grade == 2
        }

    @property
    def all_relevant_ids(self) -> set[str]:
        """Binary recall convention: grades 1 and 2 are counted as relevant."""
        return {
            mid for mid, j in self.relevance_judgments.items()
            if j.grade in (1, 2)
        }

    @property
    def is_negative_query(self) -> bool:
        """True if query has no relevant documents in corpus (out of domain)."""
        return len(self.all_relevant_ids) == 0 or self.expect_abstention


@dataclass
class BenchmarkDataset:
    version: str
    dataset_name: str
    description: str
    splits: Dict[str, int]
    categories: List[str]
    queries: List[BenchmarkQuery]

    def filter_split(self, split: Optional[str] = None) -> List[BenchmarkQuery]:
        if not split or split.lower() == "all":
            return self.queries
        split_lower = split.lower()
        return [q for q in self.queries if q.split.lower() == split_lower]


def load_benchmark(
    path: str | Path | None = None,
    split: Optional[str] = None,
) -> BenchmarkDataset:
    """Load benchmark dataset from JSON."""
    file_path = Path(path) if path else DEFAULT_BENCHMARK_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries: List[BenchmarkQuery] = []
    for q in data.get("queries", []):
        judgments: Dict[str, RelevanceJudgment] = {}
        for mid, j in q.get("relevance_judgments", {}).items():
            if isinstance(j, dict):
                judgments[mid] = RelevanceJudgment(
                    grade=int(j.get("grade", 0)),
                    reason=j.get("reason", ""),
                    source=j.get("source", "human_reviewed"),
                )
            elif isinstance(j, int):
                judgments[mid] = RelevanceJudgment(grade=j, reason="legacy grade")

        # Fallback to expected_message_ids if relevance_judgments not set
        if not judgments and q.get("expected_message_ids"):
            for mid in q["expected_message_ids"]:
                judgments[mid] = RelevanceJudgment(grade=2, reason="historical expected ID")

        queries.append(
            BenchmarkQuery(
                id=q.get("id", f"q{len(queries)+1:02d}"),
                query=q["query"],
                category=q.get("category", "general"),
                split=q.get("split", "test"),
                expected_message_ids=q.get("expected_message_ids", []),
                relevance_judgments=judgments,
                reference_answer=q.get("reference_answer"),
                key_evidence=q.get("key_evidence", []),
                expect_abstention=q.get("expect_abstention", False),
            )
        )

    dataset = BenchmarkDataset(
        version=data.get("benchmark_version", "1.0.0"),
        dataset_name=data.get("dataset_name", "MailMind Benchmark"),
        description=data.get("description", ""),
        splits=data.get("splits", {"total": len(queries)}),
        categories=data.get("categories", []),
        queries=queries,
    )

    if split:
        dataset.queries = dataset.filter_split(split)

    return dataset


def check_coverage(
    dataset: BenchmarkDataset,
    corpus_metadata_path: str | Path | None = None,
) -> dict:
    """
    Verify coverage of benchmark IDs against the indexed corpus.
    Explicitly reports missing ground-truth documents.
    """
    meta_path = Path(corpus_metadata_path) if corpus_metadata_path else DEFAULT_METADATA_PATH
    if not meta_path.exists():
        return {
            "status": "error",
            "message": f"Corpus metadata not found at {meta_path}",
            "indexed_emails": 0,
            "missing_ids": [],
        }

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    indexed_ids = {item["message_id"] for item in meta}

    missing_ids: set[str] = set()
    total_judgments = 0
    queries_with_missing_docs: list[str] = []

    for q in dataset.queries:
        query_missing = []
        for mid in q.relevance_judgments.keys():
            total_judgments += 1
            if mid not in indexed_ids:
                missing_ids.add(mid)
                query_missing.append(mid)
        if query_missing:
            queries_with_missing_docs.append(q.id)

    return {
        "status": "ok" if not missing_ids else "coverage_warning",
        "indexed_emails": len(indexed_ids),
        "total_queries": len(dataset.queries),
        "total_judgments": total_judgments,
        "missing_ids_count": len(missing_ids),
        "missing_ids": sorted(list(missing_ids)),
        "queries_with_missing_docs": queries_with_missing_docs,
    }


def propose_llm_labels(
    query_text: str,
    candidate_emails: list[dict],
    model: str | None = None,
) -> dict[str, dict]:
    """
    Use an LLM (Gemini) to propose graded relevance judgments with rationales.
    Outputs proposals for human review; does NOT modify frozen ground truth.
    """
    from google import genai
    from rag.config import settings

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured for LLM proposal.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    model_name = model or settings.GEMINI_MODEL

    email_blocks = []
    for idx, cand in enumerate(candidate_emails, 1):
        mid = cand.get("message_id", f"cand_{idx}")
        subj = cand.get("subject", "No subject")
        sender = cand.get("sender_name") or cand.get("sender_email", "Unknown")
        snippet = cand.get("snippet") or cand.get("body_snippet") or cand.get("text", "")
        email_blocks.append(
            f"Candidate [{idx}] ID: {mid}\nFrom: {sender}\nSubject: {subj}\nContent: {snippet[:400]}\n"
        )

    prompt = f"""You are an expert retrieval benchmark judge for an email retrieval system.
Evaluate the relevance of each candidate email to the user search query.

USER QUERY: "{query_text}"

GRADING RUBRIC:
2 = Directly relevant: Directly answers or announces the exact subject matter of the query.
1 = Partially relevant: Useful background, related event/company, or follow-up notice, but not the direct answer.
0 = Irrelevant: Unrelated, generic notice, or false positive match.

CANDIDATES:
{chr(10).join(email_blocks)}

Return a strict JSON object mapping each candidate message_id to an object with "grade" (integer 0, 1, or 2) and "reason" (short string explanation).
Format:
{{
  "message_id": {{"grade": 2, "reason": "..."}}
}}
Return ONLY valid JSON.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        proposals = json.loads(text)
        return proposals
    except Exception as e:
        logger.error(f"Failed to parse LLM proposal response: {e}")
        return {}

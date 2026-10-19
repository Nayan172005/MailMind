"""
MailMind Evaluation — RAG Answer Generation Evaluator.

Evaluates generated answers from the production pipeline (`pipeline.ask`) against:
- Answer Correctness (versus reference answer)
- Faithfulness / Groundedness (versus retrieved context, checking for hallucinations)
- Answer Relevance (to the question)
- Citation / Source Correctness (checking source attribution)
- Abstention / Uncertainty Handling (for out-of-domain / unanswerable queries)

Supports both LLM-as-Judge (with explicit rubrics) and deterministic rule-based evaluation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from rag.config import settings
from rag.pipeline import RAGPipeline
from rag.schemas import AskResponse
from rag.utils import get_logger, truncate_text
from evaluation.benchmark import BenchmarkQuery, BenchmarkDataset

logger = get_logger("evaluation.rag")

ABSTENTION_PHRASES = [
    "couldn't find any relevant emails",
    "couldn't find enough information",
    "could not find any relevant emails",
    "could not find enough information",
    "no relevant emails",
    "please try rephrasing",
]


@dataclass
class RAGQueryEvaluation:
    query_id: str
    query_text: str
    category: str
    generated_answer: str
    reference_answer: Optional[str]
    retrieved_sources_count: int
    retrieved_message_ids: List[str]
    is_abstention_expected: bool
    did_abstain: bool
    abstention_correct: bool
    correctness_score: float  # 0.0 - 1.0 (or 1-5 normalized)
    groundedness_score: float
    relevance_score: float
    citation_correctness: float
    key_evidence_coverage: float
    unsupported_claims: List[str]
    eval_method: str  # "llm_judge" or "deterministic_rule"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def check_abstention(text: str) -> bool:
    """Check if generated response is an abstention/not-found response."""
    lower = text.lower()
    return any(phrase in lower for phrase in ABSTENTION_PHRASES)


def deterministic_evaluate_answer(
    query: BenchmarkQuery,
    answer: str,
    context_text: str,
    retrieved_sources: list,
) -> RAGQueryEvaluation:
    """
    Deterministic rule-based evaluator (works fully offline without API calls).
    Evaluates key evidence presence, abstention behavior, context token containment, and citation match.
    """
    did_abstain = check_abstention(answer)
    abstention_correct = (did_abstain == query.expect_abstention)

    # If abstention was expected and occurred, give full marks for groundedness & correctness
    if query.expect_abstention:
        return RAGQueryEvaluation(
            query_id=query.id,
            query_text=query.query,
            category=query.category,
            generated_answer=answer,
            reference_answer=query.reference_answer,
            retrieved_sources_count=len(retrieved_sources),
            retrieved_message_ids=[s.message_id for s in retrieved_sources],
            is_abstention_expected=True,
            did_abstain=did_abstain,
            abstention_correct=abstention_correct,
            correctness_score=1.0 if abstention_correct else 0.0,
            groundedness_score=1.0 if abstention_correct else 0.2,
            relevance_score=1.0 if abstention_correct else 0.5,
            citation_correctness=1.0,
            key_evidence_coverage=1.0 if abstention_correct else 0.0,
            unsupported_claims=["Hallucinated answer for irrelevant query"] if not abstention_correct else [],
            eval_method="deterministic_rule",
        )

    # Check key evidence coverage
    evidence_hits = 0
    answer_lower = answer.lower()
    for ev in query.key_evidence:
        if ev.lower() in answer_lower:
            evidence_hits += 1
    evidence_cov = (evidence_hits / len(query.key_evidence)) if query.key_evidence else 1.0

    # Check citations: verify if mentioned sources match retrieved search results
    citation_score = 1.0
    if retrieved_sources:
        # Check if subject words or sender words appear when citing
        mentioned = 0
        for src in retrieved_sources[:3]:
            subj_words = [w for w in re.findall(r"\w+", src.subject.lower()) if len(w) > 4]
            if any(w in answer_lower for w in subj_words):
                mentioned += 1
        citation_score = min(1.0, mentioned / max(1, len(retrieved_sources[:2])))

    # Basic groundedness heuristic: check token overlap with retrieved context
    context_words = set(re.findall(r"\w+", context_text.lower()))
    answer_words = [w for w in re.findall(r"\w+", answer_lower) if len(w) > 3]
    in_context = sum(1 for w in answer_words if w in context_words)
    groundedness = (in_context / len(answer_words)) if answer_words else 1.0

    # Relevance heuristic: question terms present in answer
    q_words = [w for w in re.findall(r"\w+", query.query.lower()) if len(w) > 3]
    q_in_ans = sum(1 for w in q_words if w in answer_lower)
    relevance = (q_in_ans / len(q_words)) if q_words else 1.0

    return RAGQueryEvaluation(
        query_id=query.id,
        query_text=query.query,
        category=query.category,
        generated_answer=answer,
        reference_answer=query.reference_answer,
        retrieved_sources_count=len(retrieved_sources),
        retrieved_message_ids=[s.message_id for s in retrieved_sources],
        is_abstention_expected=False,
        did_abstain=did_abstain,
        abstention_correct=abstention_correct,
        correctness_score=round(evidence_cov, 3),
        groundedness_score=round(groundedness, 3),
        relevance_score=round(relevance, 3),
        citation_correctness=round(citation_score, 3),
        key_evidence_coverage=round(evidence_cov, 3),
        unsupported_claims=[],
        eval_method="deterministic_rule",
    )


def llm_judge_evaluate_answer(
    query: BenchmarkQuery,
    answer: str,
    context_text: str,
    retrieved_sources: list,
    model: str | None = None,
) -> RAGQueryEvaluation:
    """
    LLM-as-Judge evaluator using Gemini with strict scoring rubrics (1-5).
    Falls back gracefully to deterministic evaluation on API error.
    """
    from google import genai
    from rag.config import settings

    if not settings.GEMINI_API_KEY:
        return deterministic_evaluate_answer(query, answer, context_text, retrieved_sources)

    did_abstain = check_abstention(answer)
    abstention_correct = (did_abstain == query.expect_abstention)

    prompt = f"""You are an expert impartial judge evaluating an automated Email RAG Question-Answering system.
Evaluate the candidate answer against the query, retrieved context, and reference answer.

USER QUERY: {query.query}
EXPECT ABSTENTION: {query.expect_abstention}
REFERENCE ANSWER: {query.reference_answer}
KEY EVIDENCE POINTS: {json.dumps(query.key_evidence)}

RETRIEVED EMAIL CONTEXT:
{context_text[:2000]}

GENERATED ANSWER:
{answer}

SCORING RUBRIC (Scale 1 to 5):
1. Correctness (1-5): Does the answer convey the accurate facts as stated in the reference answer?
   - 5 = Completely correct
   - 3 = Partially correct or missing minor facts
   - 1 = Incorrect or fabricated facts
   (If expect_abstention is true, award 5 if system abstained, 1 if it fabricated an answer)

2. Groundedness/Faithfulness (1-5): Is every statement in the answer strictly supported by the retrieved email context?
   - 5 = 100% grounded, zero hallucination
   - 3 = Mostly grounded, but has slight embellishment
   - 1 = Severe hallucination or contradicts context

3. Answer Relevance (1-5): Does the answer directly answer the user query?
   - 5 = Perfectly relevant and concise
   - 1 = Off-topic or fails to answer

4. Citation Correctness (1-5): If emails are cited or referenced, are they accurate and from the retrieved context?
   - 5 = Perfectly cited or appropriately factual

Return a strict JSON object:
{{
  "correctness": <int 1-5>,
  "groundedness": <int 1-5>,
  "relevance": <int 1-5>,
  "citation_correctness": <int 1-5>,
  "unsupported_claims": [<list of strings of any fabricated claims, or empty list>],
  "reasoning": "<short rationale>"
}}
Return ONLY valid JSON.
"""

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        model_name = model or settings.GEMINI_MODEL
        resp = client.models.generate_content(model=model_name, contents=prompt)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        # Normalize 1-5 scale to 0.0-1.0
        c_score = (float(data.get("correctness", 3)) - 1.0) / 4.0
        g_score = (float(data.get("groundedness", 3)) - 1.0) / 4.0
        r_score = (float(data.get("relevance", 3)) - 1.0) / 4.0
        cit_score = (float(data.get("citation_correctness", 3)) - 1.0) / 4.0

        # Check evidence coverage
        ev_hits = sum(1 for e in query.key_evidence if e.lower() in answer.lower())
        ev_cov = (ev_hits / len(query.key_evidence)) if query.key_evidence else 1.0

        return RAGQueryEvaluation(
            query_id=query.id,
            query_text=query.query,
            category=query.category,
            generated_answer=answer,
            reference_answer=query.reference_answer,
            retrieved_sources_count=len(retrieved_sources),
            retrieved_message_ids=[s.message_id for s in retrieved_sources],
            is_abstention_expected=query.expect_abstention,
            did_abstain=did_abstain,
            abstention_correct=abstention_correct,
            correctness_score=round(c_score, 3),
            groundedness_score=round(g_score, 3),
            relevance_score=round(r_score, 3),
            citation_correctness=round(cit_score, 3),
            key_evidence_coverage=round(ev_cov, 3),
            unsupported_claims=data.get("unsupported_claims", []),
            eval_method="llm_judge",
        )
    except Exception as e:
        logger.warning(f"LLM judge failed for {query.id}: {e}. Falling back to deterministic.")
        res = deterministic_evaluate_answer(query, answer, context_text, retrieved_sources)
        res.error = str(e)
        return res


def run_rag_evaluation(
    pipeline: RAGPipeline,
    dataset: BenchmarkDataset,
    use_llm_judge: bool = True,
    sample_queries: Optional[List[str]] = None,
) -> dict:
    """
    Run full end-to-end RAG answer evaluation.
    Evaluates queries that have reference answers.
    """
    queries = [q for q in dataset.queries if q.reference_answer is not None]
    if sample_queries:
        queries = [q for q in queries if q.id in sample_queries]

    print(f"\nEvaluating RAG answer generation on {len(queries)} queries (LLM judge: {use_llm_judge})...")

    results: List[RAGQueryEvaluation] = []
    for q in queries:
        try:
            # Run production retrieve_with_context + ask
            from rag.retriever import retrieve_with_context
            from rag.generator import generate_answer

            search_results, context = retrieve_with_context(q.query, pipeline.store, top_k=5)

            if not search_results:
                answer = (
                    "I couldn't find any relevant emails in your mailbox for this query. "
                    "Please try rephrasing your question."
                )
            else:
                answer = generate_answer(q.query, context)

            if use_llm_judge and settings.GEMINI_API_KEY:
                eval_res = llm_judge_evaluate_answer(q, answer, context, search_results)
            else:
                eval_res = deterministic_evaluate_answer(q, answer, context, search_results)

            results.append(eval_res)
            if use_llm_judge and settings.GEMINI_API_KEY:
                time.sleep(2.0)
        except Exception as e:
            logger.error(f"Error during RAG evaluation for query {q.id}: {e}")
            results.append(
                RAGQueryEvaluation(
                    query_id=q.id,
                    query_text=q.query,
                    category=q.category,
                    generated_answer="",
                    reference_answer=q.reference_answer,
                    retrieved_sources_count=0,
                    retrieved_message_ids=[],
                    is_abstention_expected=q.expect_abstention,
                    did_abstain=False,
                    abstention_correct=False,
                    correctness_score=0.0,
                    groundedness_score=0.0,
                    relevance_score=0.0,
                    citation_correctness=0.0,
                    key_evidence_coverage=0.0,
                    unsupported_claims=[],
                    eval_method="failed",
                    error=str(e),
                )
            )

    valid_results = [r for r in results if r.error is None or r.generated_answer]

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    abstention_eligible = [r for r in valid_results if r.is_abstention_expected]
    abstention_accuracy = (
        sum(1 for r in abstention_eligible if r.abstention_correct) / len(abstention_eligible)
        if abstention_eligible else 1.0
    )

    summary = {
        "queries_evaluated": len(results),
        "successful_evaluations": len(valid_results),
        "mean_correctness": round(avg([r.correctness_score for r in valid_results]), 3),
        "mean_groundedness": round(avg([r.groundedness_score for r in valid_results]), 3),
        "mean_relevance": round(avg([r.relevance_score for r in valid_results]), 3),
        "mean_citation_correctness": round(avg([r.citation_correctness for r in valid_results]), 3),
        "mean_evidence_coverage": round(avg([r.key_evidence_coverage for r in valid_results]), 3),
        "abstention_accuracy": round(abstention_accuracy, 3),
        "total_unsupported_claims": sum(len(r.unsupported_claims) for r in valid_results),
        "eval_method": "llm_judge" if use_llm_judge and settings.GEMINI_API_KEY else "deterministic_rule",
    }

    return {
        "summary": summary,
        "per_query": [r.to_dict() for r in results],
    }

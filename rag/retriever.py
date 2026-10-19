"""
MailMind RAG — Retriever

High-level retrieval workflow:
1. Deterministic query analysis (entities, senders, venues, date constraints, temporal intent)
2. Hybrid candidate discovery (Dense FAISS + In-Memory Lexical BM25 via RRF)
3. Semantic similarity filtering and evidence-aware out-of-domain rejection
4. Multi-signal metadata-aware reranking and message deduplication
5. High-quality evidence selection and clean prompt context formatting
"""

from __future__ import annotations

from typing import Optional

from rag.config import settings
from rag.schemas import SearchResult
from rag.embedder import embed_query
from rag.query_analyzer import QueryAnalysis, analyze_query
from rag.hybrid_retriever import search_hybrid
from rag.reranker import rerank_candidates
from rag.vector_store import VectorStore
from rag.utils import get_logger, truncate_text

logger = get_logger("retriever")


def _final_k(top_k: int | None) -> int:
    return top_k or settings.FINAL_RETRIEVAL_K or settings.TOP_K


def _initial_k(final_k: int, deduplicate: bool) -> int:
    if deduplicate:
        return max(settings.INITIAL_RETRIEVAL_K, final_k * 4)
    return max(settings.INITIAL_RETRIEVAL_K, final_k * 2)


def _to_search_result(candidate: dict) -> SearchResult:
    return SearchResult(
        message_id=candidate["message_id"],
        subject=candidate.get("subject", ""),
        sender_name=candidate.get("sender_name", ""),
        sender_email=candidate.get("sender_email", ""),
        date=candidate.get("date", ""),
        score=round(float(candidate.get("final_score", candidate.get("score", 0.0))), 4),
        snippet=truncate_text(
            candidate.get("body_snippet", candidate.get("text", "")),
            300,
        ),
    )


def _retrieve_ranked_candidates(
    query: str,
    store: VectorStore,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    deduplicate: bool = True,
    use_hybrid: bool | None = None,
) -> list[dict]:
    if not query.strip():
        logger.warning("Empty query received")
    if hasattr(store, "reload_if_changed"):
        store.reload_if_changed()

    if not store.is_loaded:
        raise RuntimeError("Vector store is not loaded.")

    final_k = _final_k(top_k)
    initial_k = _initial_k(final_k, deduplicate)
    threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
    enable_hybrid = use_hybrid if use_hybrid is not None else settings.ENABLE_HYBRID_RETRIEVAL

    analysis = analyze_query(query)

    # 1. Candidate discovery
    if enable_hybrid:
        candidates = search_hybrid(
            query=query,
            analysis=analysis,
            store=store,
            initial_k=initial_k,
            dense_weight=settings.HYBRID_DENSE_WEIGHT,
            lex_weight=settings.HYBRID_LEXICAL_WEIGHT,
        )
    else:
        query_embedding = embed_query(query)
        candidates = store.search(query_embedding, top_k=initial_k)

    # 2. Similarity threshold filtering
    filtered = [c for c in candidates if float(c.get("score", 0.0)) >= threshold]

    if not filtered:
        logger.info("No results above threshold %.2f for: %s", threshold, truncate_text(query, 80))
        return []

    # 3. Evidence-aware out-of-domain check
    # If the top candidate semantic similarity is very low and key query entities/date tokens are absent,
    # reject early to prevent false positives on negative queries
    top_semantic = max(float(c.get("score", 0.0)) for c in filtered)
    if top_semantic < 0.40 and (analysis.entities or analysis.date_tokens):
        top_cand = max(filtered, key=lambda c: float(c.get("score", 0.0)))
        cand_text = f"{top_cand.get('subject', '')} {top_cand.get('body_snippet', '')}".lower()
        key_constraints = [dt.lower() for dt in analysis.date_tokens] + [
            e.lower() for e in analysis.entities if len(e) > 3
        ]
        if key_constraints and not any(kc in cand_text for kc in key_constraints):
            logger.info("Out-of-domain rejection for query: %s", truncate_text(query, 60))
            return []

    # 4. Multi-signal metadata-aware reranking
    ranked = rerank_candidates(
        query=query,
        analysis=analysis,
        candidates=filtered,
        final_k=final_k,
        deduplicate=deduplicate,
    )

    logger.info(
        "Retrieved %d results for: %s (initial=%d, filtered=%d)",
        len(ranked),
        truncate_text(query, 60),
        len(candidates),
        len(filtered),
    )
    return ranked


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    deduplicate: bool = True,
    use_hybrid: bool | None = None,
) -> list[SearchResult]:
    """
    Perform semantic retrieval for a query.

    Args:
        query: User search query.
        store: Loaded VectorStore instance.
        top_k: Number of results to return.
        similarity_threshold: Minimum similarity score to include.
        deduplicate: If True, keep only the best chunk per email.
        use_hybrid: Override hybrid retrieval configuration.

    Returns:
        List of SearchResult objects, sorted by descending score.
    """
    ranked = _retrieve_ranked_candidates(
        query=query,
        store=store,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        deduplicate=deduplicate,
        use_hybrid=use_hybrid,
    )
    return [_to_search_result(candidate) for candidate in ranked]


def retrieve_with_context(
    query: str,
    store: VectorStore,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    use_hybrid: bool | None = None,
) -> tuple[list[SearchResult], str]:
    """
    Retrieve results AND build clean evidence context for LLM generation.
    Excludes debug scores to prevent hallucination, and establishes explicit
    [Source X] source IDs for citation alignment.

    Returns:
        (results, context_text)
    """
    ranked = _retrieve_ranked_candidates(
        query=query,
        store=store,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        deduplicate=True,
        use_hybrid=use_hybrid,
    )
    results = [_to_search_result(candidate) for candidate in ranked]

    if not results:
        return [], ""

    # Evidence selection: select high-confidence relevant candidates (avoiding weakly related padding)
    top_score = ranked[0]["final_score"]
    confident_ranked = [
        c for c in ranked
        if c["final_score"] >= max(top_score - 0.12, settings.MIN_RERANK_SCORE)
    ][:3]
    if not confident_ranked:
        confident_ranked = ranked[:1]

    confident_results = [_to_search_result(c) for c in confident_ranked]

    evidence_items = []
    for i, (result, candidate) in enumerate(zip(confident_results, confident_ranked), 1):
        chunk_text = candidate.get("text", result.snippet).strip()
        evidence_items.append(
            f"[Source {i}] (Email {i})\n"
            f"Message ID: {result.message_id}\n"
            f"Subject: {result.subject}\n"
            f"From: {result.sender_name} <{result.sender_email}>\n"
            f"Date: {result.date}\n\n"
            f"{chunk_text}\n"
        )

    context_text = "\n---\n\n".join(evidence_items)
    return results, context_text

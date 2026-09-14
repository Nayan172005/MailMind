"""
MailMind RAG — Retriever

High-level retrieval logic:
1. Analyze query intent/entities/temporal signals
2. Embed query
3. Search FAISS with deeper candidate depth
4. Filter by semantic similarity threshold
5. Deduplicate by message_id
6. Metadata-aware reranking
7. Return ranked results
"""

from rag.config import settings
from rag.schemas import SearchResult
from rag.embedder import embed_query
from rag.query_analyzer import analyze_query
from rag.reranker import rerank_candidates
from rag.vector_store import VectorStore
from rag.utils import get_logger, truncate_text

logger = get_logger("retriever")


def _final_k(top_k: int | None) -> int:
    return top_k or settings.FINAL_RETRIEVAL_K or settings.TOP_K


def _initial_k(final_k: int, deduplicate: bool) -> int:
    if deduplicate:
        return max(settings.INITIAL_RETRIEVAL_K, final_k * 3)
    return max(settings.INITIAL_RETRIEVAL_K, final_k)


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
) -> list[dict]:
    if not query.strip():
        logger.warning("Empty query received")
        return []

    if not store.is_loaded:
        raise RuntimeError("Vector store is not loaded.")

    final_k = _final_k(top_k)
    initial_k = _initial_k(final_k, deduplicate)
    threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

    analysis = analyze_query(query)
    query_embedding = embed_query(query)
    raw_results = store.search(query_embedding, top_k=initial_k)
    filtered = [r for r in raw_results if r["score"] >= threshold]

    if not filtered:
        logger.info(f"No results above threshold {threshold} for: {truncate_text(query, 80)}")
        return []

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
        len(raw_results),
        len(filtered),
    )
    return ranked


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    deduplicate: bool = True,
) -> list[SearchResult]:
    """
    Perform semantic retrieval for a query.

    Args:
        query: User search query.
        store: Loaded VectorStore instance.
        top_k: Number of results to return (before dedup).
        similarity_threshold: Minimum similarity score to include.
        deduplicate: If True, keep only the best chunk per email.

    Returns:
        List of SearchResult objects, sorted by descending score.
    """
    ranked = _retrieve_ranked_candidates(
        query=query,
        store=store,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        deduplicate=deduplicate,
    )
    return [_to_search_result(candidate) for candidate in ranked]


def retrieve_with_context(
    query: str,
    store: VectorStore,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> tuple[list[SearchResult], str]:
    """
    Retrieve results AND build a context string for LLM generation.

    Returns:
        (results, context_text) — the context text is formatted for the prompt.
    """
    ranked = _retrieve_ranked_candidates(query, store, top_k, similarity_threshold)
    results = [_to_search_result(candidate) for candidate in ranked]

    if not results:
        return results, ""

    # Build context from the selected reranked chunks, preserving useful
    # metadata for grounded generation.
    context_parts = []
    for i, (result, candidate) in enumerate(zip(results, ranked), 1):
        context_parts.append(
            f"--- Email {i} ---\n"
            f"Message ID: {result.message_id}\n"
            f"Subject: {result.subject}\n"
            f"From: {result.sender_name} <{result.sender_email}>\n"
            f"Date: {result.date}\n"
            f"Relevance Score: {result.score}\n\n"
            f"{candidate.get('text', result.snippet)}\n"
        )

    context_text = "\n".join(context_parts)
    return results, context_text

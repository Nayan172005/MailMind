"""
MailMind RAG — Retriever

High-level retrieval logic:
1. Embed query
2. Search FAISS
3. Filter by similarity threshold
4. Deduplicate by message_id (keep best chunk per email)
5. Return ranked results
"""

from rag.config import settings
from rag.schemas import SearchResult
from rag.embedder import embed_query
from rag.vector_store import VectorStore
from rag.utils import get_logger, truncate_text

logger = get_logger("retriever")


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
    if not query.strip():
        logger.warning("Empty query received")
        return []

    if not store.is_loaded:
        raise RuntimeError("Vector store is not loaded.")

    top_k = top_k or settings.TOP_K
    threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

    # Retrieve more candidates before dedup to ensure we get enough unique emails
    search_k = top_k * 3 if deduplicate else top_k

    # 1. Embed query
    query_embedding = embed_query(query)

    # 2. Search FAISS
    raw_results = store.search(query_embedding, top_k=search_k)

    # 3. Filter by similarity threshold
    filtered = [r for r in raw_results if r["score"] >= threshold]

    if not filtered:
        logger.info(f"No results above threshold {threshold} for: {truncate_text(query, 80)}")
        return []

    # 4. Deduplicate by message_id (keep highest-scoring chunk per email)
    if deduplicate:
        seen: dict[str, dict] = {}
        for r in filtered:
            msg_id = r["message_id"]
            if msg_id not in seen or r["score"] > seen[msg_id]["score"]:
                seen[msg_id] = r
        deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    else:
        deduped = filtered

    # 5. Trim to top_k
    final = deduped[:top_k]

    # 6. Convert to SearchResult
    results = []
    for r in final:
        results.append(
            SearchResult(
                message_id=r["message_id"],
                subject=r.get("subject", ""),
                sender_name=r.get("sender_name", ""),
                sender_email=r.get("sender_email", ""),
                date=r.get("date", ""),
                score=round(r["score"], 4),
                snippet=truncate_text(r.get("body_snippet", r.get("text", "")), 300),
            )
        )

    logger.info(
        f"Retrieved {len(results)} results for: {truncate_text(query, 60)}"
    )
    return results


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
    results = retrieve(query, store, top_k, similarity_threshold)

    if not results:
        return results, ""

    # Build context from the full chunk texts (not just snippets)
    # We need the full text from metadata for generation quality
    context_parts = []
    for i, result in enumerate(results, 1):
        # Get the full chunk text from the store metadata
        chunks = store.get_chunks_by_message_id(result.message_id)
        if chunks:
            # Use the best chunk's full text
            best_chunk = max(chunks, key=lambda c: c.get("score", 0)) if len(chunks) > 1 else chunks[0]
            context_parts.append(
                f"--- Email {i} ---\n"
                f"Subject: {result.subject}\n"
                f"From: {result.sender_name} <{result.sender_email}>\n"
                f"Date: {result.date}\n"
                f"Relevance Score: {result.score}\n\n"
                f"{best_chunk.get('text', result.snippet)}\n"
            )
        else:
            context_parts.append(
                f"--- Email {i} ---\n"
                f"Subject: {result.subject}\n"
                f"From: {result.sender_name} <{result.sender_email}>\n"
                f"Date: {result.date}\n\n"
                f"{result.snippet}\n"
            )

    context_text = "\n".join(context_parts)
    return results, context_text

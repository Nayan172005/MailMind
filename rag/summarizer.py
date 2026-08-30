"""
MailMind RAG — Email Summarizer

Two modes:
  Mode A: Single email summarization (by message_id)
  Mode B: Multi-email summarization (from retrieval query)
"""

from rag.config import settings
from rag.schemas import SearchResult, SummarizeResponse
from rag.vector_store import VectorStore
from rag.retriever import retrieve_with_context
from rag.generator import generate_summary
from rag.utils import get_logger

logger = get_logger("summarizer")


def summarize_single_email(
    message_id: str,
    store: VectorStore,
    emails_lookup: dict[str, dict] | None = None,
) -> SummarizeResponse:
    """
    Summarize a single email by its message_id.

    Args:
        message_id: The email's Message_ID.
        store: Loaded VectorStore instance.
        emails_lookup: Optional dict mapping message_id -> raw email for full body.

    Returns:
        SummarizeResponse with summary and source info.
    """
    # Get chunks for this email from the store metadata
    chunks = store.get_chunks_by_message_id(message_id)

    if not chunks:
        return SummarizeResponse(
            summary=f"Email with ID '{message_id}' was not found in the index.",
            sources=[],
            mode="single",
        )

    # Combine all chunks for this email to get the full content
    # Sort by chunk_id to maintain order
    chunks_sorted = sorted(chunks, key=lambda c: c.get("chunk_id", 0))

    # Build context from the first chunk's metadata + all body chunks
    first = chunks_sorted[0]
    context_parts = [
        f"Subject: {first.get('subject', '')}",
        f"From: {first.get('sender_name', '')} <{first.get('sender_email', '')}>",
        f"Date: {first.get('date', '')}",
        "",
    ]

    # If we have the full email lookup, use the original body
    if emails_lookup and message_id in emails_lookup:
        email_data = emails_lookup[message_id]
        context_parts.append(email_data.get("body", ""))
    else:
        # Combine body snippets from chunks
        for chunk in chunks_sorted:
            context_parts.append(chunk.get("body_snippet", ""))

    context = "\n".join(context_parts)

    # Generate summary
    summary = generate_summary(context, mode="single")

    source = SearchResult(
        message_id=message_id,
        subject=first.get("subject", ""),
        sender_name=first.get("sender_name", ""),
        sender_email=first.get("sender_email", ""),
        date=first.get("date", ""),
        score=1.0,
        snippet=first.get("body_snippet", "")[:300],
    )

    return SummarizeResponse(
        summary=summary,
        sources=[source],
        mode="single",
    )


def summarize_by_query(
    query: str,
    store: VectorStore,
    top_k: int | None = None,
) -> SummarizeResponse:
    """
    Summarize emails matching a semantic query.

    Args:
        query: Search query (e.g., "placement-related emails").
        store: Loaded VectorStore instance.
        top_k: Number of emails to include in the summary.

    Returns:
        SummarizeResponse with combined summary and sources.
    """
    top_k = top_k or settings.TOP_K

    results, context = retrieve_with_context(query, store, top_k=top_k)

    if not results:
        return SummarizeResponse(
            summary=(
                "I couldn't find any relevant emails to summarize for this query. "
                "Please try a different search term."
            ),
            sources=[],
            mode="multi",
        )

    # Generate multi-email summary
    summary = generate_summary(context, mode="multi")

    return SummarizeResponse(
        summary=summary,
        sources=results,
        mode="multi",
    )

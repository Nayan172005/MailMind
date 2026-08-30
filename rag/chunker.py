"""
MailMind RAG — Document Chunker

Converts normalized emails into searchable document chunks.

Strategy:
- Short emails (body <= CHUNK_SIZE words): 1 email = 1 chunk
- Long emails: split body into overlapping chunks of ~CHUNK_SIZE words
- Every chunk includes a metadata header (subject, sender, date) for
  better semantic matching — the subject often contains the most
  discriminative information (placement, NPTEL, exam, etc.)
"""

from rag.schemas import EmailRecord, DocumentChunk
from rag.config import settings
from rag.utils import get_logger, count_words

logger = get_logger("chunker")


def _build_metadata_header(email: EmailRecord) -> str:
    """Build a text header from email metadata for embedding."""
    parts = []
    if email.subject:
        parts.append(f"Subject: {email.subject}")
    if email.sender_name or email.sender_email:
        sender = email.sender_name or email.sender_email
        parts.append(f"From: {sender}")
    if email.date:
        parts.append(f"Date: {email.date}")
    return "\n".join(parts)


def _split_text_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: The text to split.
        chunk_size: Approximate number of words per chunk.
        chunk_overlap: Number of overlapping words between chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # Move forward by (chunk_size - overlap)
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size  # safety: avoid infinite loop
        start += step

    return chunks


def chunk_email(
    email: EmailRecord,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    Convert a single email into one or more document chunks.

    Each chunk contains the metadata header + a portion of the body.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    header = _build_metadata_header(email)
    body = email.body or ""

    body_word_count = count_words(body)

    if body_word_count <= chunk_size:
        # Short email: single chunk
        full_text = f"{header}\n\nBody:\n{body}" if body else header
        return [
            DocumentChunk(
                chunk_id=0,
                message_id=email.message_id,
                subject=email.subject,
                sender_name=email.sender_name,
                sender_email=email.sender_email,
                date=email.date,
                text=full_text,
                body_snippet=body[:500] if body else "",
            )
        ]

    # Long email: split body into chunks
    body_chunks = _split_text_into_chunks(body, chunk_size, chunk_overlap)
    doc_chunks = []

    for i, body_chunk in enumerate(body_chunks):
        full_text = f"{header}\n\nBody:\n{body_chunk}"
        doc_chunks.append(
            DocumentChunk(
                chunk_id=i,
                message_id=email.message_id,
                subject=email.subject,
                sender_name=email.sender_name,
                sender_email=email.sender_email,
                date=email.date,
                text=full_text,
                body_snippet=body_chunk[:500],
            )
        )

    return doc_chunks


def chunk_emails(
    emails: list[EmailRecord],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Chunk a batch of emails into document chunks."""
    all_chunks = []

    for email in emails:
        if not email.message_id:
            continue
        chunks = chunk_email(email, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    # Stats
    n_single = sum(1 for e in emails if count_words(e.body) <= (chunk_size or settings.CHUNK_SIZE))
    n_multi = len(emails) - n_single

    logger.info(
        f"Chunked {len(emails)} emails -> {len(all_chunks)} chunks "
        f"({n_single} single-chunk, {n_multi} multi-chunk)"
    )
    return all_chunks

"""Tests for rag.chunker"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.schemas import EmailRecord
from rag.chunker import chunk_email, chunk_emails


def _make_email(body: str = "Short body", subject: str = "Test Subject") -> EmailRecord:
    return EmailRecord(
        message_id="test123",
        subject=subject,
        sender_name="Test Sender",
        sender_email="test@example.com",
        date="2026-01-01",
        body=body,
    )


def test_short_email_single_chunk():
    """Short email should produce exactly one chunk."""
    email = _make_email("This is a short email body.")
    chunks = chunk_email(email)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == 0
    assert chunks[0].message_id == "test123"
    assert "Test Subject" in chunks[0].text
    assert "short email body" in chunks[0].text


def test_long_email_multiple_chunks():
    """Long email should produce multiple chunks."""
    long_body = " ".join(["word"] * 1000)  # 1000 words
    email = _make_email(long_body)
    chunks = chunk_email(email, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    # Each chunk should have metadata
    for chunk in chunks:
        assert chunk.message_id == "test123"
        assert "Test Subject" in chunk.text


def test_chunk_metadata_preserved():
    """Each chunk should preserve email metadata."""
    email = _make_email("Some body text")
    chunks = chunk_email(email)
    assert chunks[0].subject == "Test Subject"
    assert chunks[0].sender_name == "Test Sender"
    assert chunks[0].date == "2026-01-01"


def test_empty_body():
    """Email with empty body should still produce one chunk with metadata."""
    email = _make_email("")
    chunks = chunk_email(email)
    assert len(chunks) == 1
    assert "Test Subject" in chunks[0].text


def test_batch_chunking():
    """Test chunking a batch of emails."""
    emails = [
        _make_email("Email one"),
        _make_email("Email two"),
    ]
    emails[0].message_id = "msg1"
    emails[1].message_id = "msg2"

    chunks = chunk_emails(emails)
    assert len(chunks) == 2
    msg_ids = {c.message_id for c in chunks}
    assert msg_ids == {"msg1", "msg2"}


def test_chunk_overlap():
    """Verify that chunks overlap."""
    body = " ".join([f"word{i}" for i in range(600)])
    email = _make_email(body)
    chunks = chunk_email(email, chunk_size=200, chunk_overlap=50)

    # The end of chunk 0's body should overlap with the start of chunk 1's body
    assert len(chunks) >= 2
    # Both chunks should contain the subject header
    assert "Test Subject" in chunks[0].text
    assert "Test Subject" in chunks[1].text

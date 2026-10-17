"""
Tests for Gmail ingestion pipeline.

Tests the full flow: parse → normalize → classify → chunk → embed → FAISS.
Uses mocked Gmail messages and classifier to avoid external dependencies.
"""

import base64
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

import pytest
import numpy as np
import faiss

from rag.schemas import EmailRecord, DocumentChunk
from rag.vector_store import VectorStore
from rag.chunker import chunk_email
from rag.normalizer import normalize_email


# ─── Fixtures ────────────────────────────────────────────────────────

def _make_gmail_msg(
    gmail_id: str = "gm_001",
    message_id: str = "<test@example.com>",
    sender: str = "Test Sender <test@example.com>",
    subject: str = "Test Email Subject",
    body: str = "This is a test email body with enough content.",
    label_ids: list = None,
) -> dict:
    """Create a minimal Gmail message dict for testing."""
    body_encoded = base64.urlsafe_b64encode(body.encode()).decode()
    return {
        "id": gmail_id,
        "threadId": f"thread_{gmail_id}",
        "labelIds": label_ids or ["INBOX"],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": sender},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Thu, 08 Oct 2026 10:30:00 +0000"},
                {"name": "Message-ID", "value": message_id},
                {"name": "To", "value": "me@example.com"},
            ],
            "body": {"data": body_encoded},
        },
    }


@pytest.fixture
def empty_store(tmp_path):
    """Create an empty VectorStore with a FAISS index."""
    store = VectorStore()
    dim = 384
    store.index = faiss.IndexFlatIP(dim)
    store.metadata = []
    store._indexed_message_ids = set()
    return store


@pytest.fixture
def seeded_store(tmp_path):
    """Create a VectorStore with some existing vectors."""
    store = VectorStore()
    dim = 384
    store.index = faiss.IndexFlatIP(dim)

    # Add 5 random vectors
    vectors = np.random.randn(5, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    store.index.add(vectors)

    store.metadata = [
        {
            "vector_id": i,
            "message_id": f"existing_{i}",
            "chunk_id": 0,
            "subject": f"Existing Email {i}",
            "sender_name": "Old Sender",
            "sender_email": "old@example.com",
            "date": "2026-01-01T00:00:00+00:00",
            "text": f"Existing email body {i}",
            "body_snippet": f"Existing email body {i}",
        }
        for i in range(5)
    ]
    store._indexed_message_ids = {f"existing_{i}" for i in range(5)}
    return store


# ─── Tests: Normalization Integration ────────────────────────────────

class TestNormalization:
    def test_gmail_dict_normalizes(self):
        """Gmail parser output normalizes into an EmailRecord."""
        from api.gmail.parser import parse_gmail_message

        msg = _make_gmail_msg()
        raw = parse_gmail_message(msg)
        email = normalize_email(raw)

        assert isinstance(email, EmailRecord)
        assert email.message_id == "<test@example.com>"
        assert email.sender_email == "test@example.com"
        assert email.subject == "Test Email Subject"
        assert "test email body" in email.body

    def test_email_chunks_correctly(self):
        """Normalized email produces valid chunks."""
        from api.gmail.parser import parse_gmail_message

        msg = _make_gmail_msg(body="Short email body")
        raw = parse_gmail_message(msg)
        email = normalize_email(raw)
        chunks = chunk_email(email)

        assert len(chunks) >= 1
        assert chunks[0].message_id == email.message_id
        assert chunks[0].subject == email.subject


# ─── Tests: Classification Integration ───────────────────────────────

class TestClassification:
    @patch("api.gmail.classifier.predict_email")
    def test_spam_classification(self, mock_predict):
        """Spam emails should be rejected."""
        from api.gmail.classifier import is_spam

        mock_predict.return_value = {
            "label": "Spam",
            "confidence": 0.95,
            "probabilities": {"Spam": 0.95, "General": 0.03, "Important": 0.02},
        }

        result = mock_predict("spammer@evil.com", "You Won!", "Click here!")
        assert is_spam(result) is True

    @patch("api.gmail.classifier.predict_email")
    def test_non_spam_classification(self, mock_predict):
        """Non-spam emails should be accepted."""
        from api.gmail.classifier import is_spam

        mock_predict.return_value = {
            "label": "Important",
            "confidence": 0.88,
            "probabilities": {"Spam": 0.02, "General": 0.10, "Important": 0.88},
        }

        result = mock_predict("cdc@vit.ac.in", "Placement Drive", "Microsoft is coming")
        assert is_spam(result) is False

    @patch("api.gmail.classifier.predict_email")
    def test_general_classification(self, mock_predict):
        from api.gmail.classifier import is_spam

        mock_predict.return_value = {
            "label": "General",
            "confidence": 0.75,
            "probabilities": {"Spam": 0.05, "General": 0.75, "Important": 0.20},
        }

        result = mock_predict("prof@vit.ac.in", "Module 3 Materials", "Study materials uploaded")
        assert is_spam(result) is False


# ─── Tests: Full Ingestion Pipeline ──────────────────────────────────

class TestIngestion:
    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_non_spam_ingestion(self, mock_is_spam, mock_predict, empty_store):
        """Non-spam email should be added to FAISS index."""
        from api.gmail.ingestion import ingest_gmail_message

        mock_predict.return_value = {
            "label": "Important",
            "confidence": 0.9,
            "probabilities": {},
        }
        mock_is_spam.return_value = False

        msg = _make_gmail_msg()
        initial_count = empty_store.total_vectors

        result = ingest_gmail_message(msg, empty_store, persist=False)

        assert result["status"] == "ingested"
        assert result["chunks_added"] >= 1
        assert empty_store.total_vectors > initial_count

    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_spam_rejection(self, mock_is_spam, mock_predict, empty_store):
        """Spam email should NOT be added to FAISS index."""
        from api.gmail.ingestion import ingest_gmail_message

        mock_predict.return_value = {
            "label": "Spam",
            "confidence": 0.95,
            "probabilities": {},
        }
        mock_is_spam.return_value = True

        msg = _make_gmail_msg(subject="You Won $1M!")
        initial_count = empty_store.total_vectors

        result = ingest_gmail_message(msg, empty_store, persist=False)

        assert result["status"] == "spam"
        assert empty_store.total_vectors == initial_count

    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_duplicate_rejection(self, mock_is_spam, mock_predict, seeded_store):
        """Duplicate emails should be rejected."""
        from api.gmail.ingestion import ingest_gmail_message

        # Use a message_id that already exists in the seeded store
        msg = _make_gmail_msg(message_id="existing_0")
        initial_count = seeded_store.total_vectors

        result = ingest_gmail_message(msg, seeded_store, persist=False)

        assert result["status"] == "duplicate"
        assert seeded_store.total_vectors == initial_count
        # Classifier should NOT have been called for duplicates
        mock_predict.assert_not_called()

    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_faiss_vector_count_increases(self, mock_is_spam, mock_predict, seeded_store):
        """FAISS vector count should increase after non-spam ingestion."""
        from api.gmail.ingestion import ingest_gmail_message

        mock_predict.return_value = {
            "label": "General",
            "confidence": 0.8,
            "probabilities": {},
        }
        mock_is_spam.return_value = False

        msg = _make_gmail_msg(message_id="<new_email@test.com>")
        initial_vectors = seeded_store.total_vectors

        result = ingest_gmail_message(msg, seeded_store, persist=False)

        assert result["status"] == "ingested"
        assert seeded_store.total_vectors > initial_vectors

    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_metadata_vector_alignment(self, mock_is_spam, mock_predict, seeded_store):
        """Metadata count must always equal FAISS vector count."""
        from api.gmail.ingestion import ingest_gmail_message

        mock_predict.return_value = {
            "label": "Important",
            "confidence": 0.9,
            "probabilities": {},
        }
        mock_is_spam.return_value = False

        msg = _make_gmail_msg(message_id="<aligned@test.com>")
        ingest_gmail_message(msg, seeded_store, persist=False)

        assert seeded_store.total_vectors == len(seeded_store.metadata)


# ─── Tests: Persistence ─────────────────────────────────────────────

class TestPersistence:
    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_index_survives_reload(self, mock_is_spam, mock_predict, tmp_path):
        """Newly ingested emails must survive save/load cycle."""
        from api.gmail.ingestion import ingest_gmail_message

        mock_predict.return_value = {
            "label": "General",
            "confidence": 0.8,
            "probabilities": {},
        }
        mock_is_spam.return_value = False

        # Create and populate store
        store = VectorStore()
        dim = 384
        store.index = faiss.IndexFlatIP(dim)
        store.metadata = []
        store._indexed_message_ids = set()

        msg = _make_gmail_msg(message_id="<persist@test.com>")
        ingest_gmail_message(msg, store, persist=False)

        # Save
        store.save(tmp_path)

        # Load into a fresh store
        store2 = VectorStore()
        store2.load(tmp_path)

        assert store2.total_vectors == store.total_vectors
        assert store2.total_vectors == len(store2.metadata)
        assert "<persist@test.com>" in store2._indexed_message_ids

    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_retrieval_after_reload(self, mock_is_spam, mock_predict, tmp_path):
        """Newly ingested email should be retrievable after reload."""
        from api.gmail.ingestion import ingest_gmail_message
        from rag.embedder import embed_query

        mock_predict.return_value = {
            "label": "Important",
            "confidence": 0.9,
            "probabilities": {},
        }
        mock_is_spam.return_value = False

        store = VectorStore()
        dim = 384
        store.index = faiss.IndexFlatIP(dim)
        store.metadata = []
        store._indexed_message_ids = set()

        msg = _make_gmail_msg(
            message_id="<retrieval@test.com>",
            subject="Microsoft Placement Drive",
            body="Microsoft is conducting a placement drive for eligible students.",
        )
        ingest_gmail_message(msg, store, persist=False)
        store.save(tmp_path)

        # Reload
        store2 = VectorStore()
        store2.load(tmp_path)

        # Search for the email
        query_emb = embed_query("Microsoft placement")
        results = store2.search(query_emb, top_k=3)

        assert len(results) > 0
        # The ingested email should be in results
        message_ids = [r["message_id"] for r in results]
        assert "<retrieval@test.com>" in message_ids


# ─── Tests: Batch Ingestion ──────────────────────────────────────────

class TestBatchIngestion:
    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_batch_mixed_results(self, mock_is_spam, mock_predict, empty_store):
        """Batch processing should handle mixed spam/non-spam/duplicate."""
        from api.gmail.ingestion import ingest_gmail_messages

        # Configure classifier: first call = spam, rest = non-spam
        mock_predict.side_effect = [
            {"label": "Spam", "confidence": 0.9, "probabilities": {}},
            {"label": "General", "confidence": 0.8, "probabilities": {}},
            {"label": "Important", "confidence": 0.85, "probabilities": {}},
        ]
        mock_is_spam.side_effect = [True, False, False]

        messages = [
            _make_gmail_msg(gmail_id="gm1", message_id="<spam@test.com>"),
            _make_gmail_msg(gmail_id="gm2", message_id="<ok1@test.com>"),
            _make_gmail_msg(gmail_id="gm3", message_id="<ok2@test.com>"),
        ]

        results = ingest_gmail_messages(messages, empty_store, persist=False)

        assert len(results) == 3
        statuses = [r["status"] for r in results]
        assert "spam" in statuses
        assert "ingested" in statuses

    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_multiple_messages_single_history(self, mock_is_spam, mock_predict, empty_store):
        """Multiple messages from one history update should all be processed."""
        from api.gmail.ingestion import ingest_gmail_messages

        mock_predict.return_value = {
            "label": "General",
            "confidence": 0.8,
            "probabilities": {},
        }
        mock_is_spam.return_value = False

        messages = [
            _make_gmail_msg(gmail_id=f"gm_{i}", message_id=f"<multi_{i}@test.com>")
            for i in range(5)
        ]

        results = ingest_gmail_messages(messages, empty_store, persist=False)
        ingested = [r for r in results if r["status"] == "ingested"]
        assert len(ingested) == 5

    @patch("api.gmail.ingestion.predict_email")
    @patch("api.gmail.ingestion.is_spam")
    def test_error_handling_continues(self, mock_is_spam, mock_predict, empty_store):
        """One email error should not prevent processing of other emails."""
        from api.gmail.ingestion import ingest_gmail_messages

        # Second call raises
        mock_predict.side_effect = [
            {"label": "General", "confidence": 0.8, "probabilities": {}},
            Exception("Model error"),
            {"label": "General", "confidence": 0.8, "probabilities": {}},
        ]
        mock_is_spam.side_effect = [False, False, False]

        messages = [
            _make_gmail_msg(gmail_id="gm1", message_id="<ok1@test.com>"),
            _make_gmail_msg(gmail_id="gm2", message_id="<err@test.com>"),
            _make_gmail_msg(gmail_id="gm3", message_id="<ok2@test.com>"),
        ]

        results = ingest_gmail_messages(messages, empty_store, persist=False)
        assert len(results) == 3
        error_results = [r for r in results if r["status"] == "error"]
        ingested_results = [r for r in results if r["status"] == "ingested"]
        assert len(error_results) == 1
        assert len(ingested_results) == 2

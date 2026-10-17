"""
MailMind Gmail — Email Ingestion Pipeline

Orchestrates the full flow:
    Gmail message → parse → normalize → classify → chunk → embed → FAISS

Reuses all existing RAG components. Does NOT rebuild the index.
"""

from datetime import datetime, timezone
from typing import Optional

from api.gmail.parser import parse_gmail_message
from api.gmail.classifier import predict_email, is_spam, load_classifier
from rag.normalizer import normalize_email
from rag.chunker import chunk_email
from rag.vector_store import VectorStore
from rag.utils import get_logger

logger = get_logger("gmail.ingestion")


# ─── Ingestion Stats ─────────────────────────────────────────────────

class IngestionStats:
    """Track ingestion statistics for the current session."""

    def __init__(self):
        self.total_processed: int = 0
        self.total_ingested: int = 0
        self.total_spam: int = 0
        self.total_duplicates: int = 0
        self.total_errors: int = 0
        self.recent_emails: list[dict] = []  # last N processed emails
        self.last_sync_time: Optional[str] = None

    def record_ingested(self, info: dict) -> None:
        self.total_processed += 1
        self.total_ingested += 1
        self._add_recent(info)

    def record_spam(self, info: dict) -> None:
        self.total_processed += 1
        self.total_spam += 1
        self._add_recent(info)

    def record_duplicate(self, info: dict) -> None:
        self.total_processed += 1
        self.total_duplicates += 1

    def record_error(self, info: dict) -> None:
        self.total_processed += 1
        self.total_errors += 1
        self._add_recent(info)

    def _add_recent(self, info: dict) -> None:
        self.recent_emails.insert(0, info)
        if len(self.recent_emails) > 50:
            self.recent_emails = self.recent_emails[:50]

    def update_sync_time(self) -> None:
        self.last_sync_time = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "total_processed": self.total_processed,
            "total_ingested": self.total_ingested,
            "total_spam": self.total_spam,
            "total_duplicates": self.total_duplicates,
            "total_errors": self.total_errors,
            "last_sync_time": self.last_sync_time,
        }


# Global stats instance
stats = IngestionStats()


# ─── Core Ingestion Functions ────────────────────────────────────────

def ingest_gmail_message(
    gmail_msg: dict,
    store: VectorStore,
    persist: bool = True,
) -> dict:
    """
    Process a single Gmail API message through the full ingestion pipeline.

    Flow:
        1. Parse Gmail message → MailMind raw dict
        2. Check for duplicate (message_id already in store)
        3. Normalize → EmailRecord
        4. Classify using BiLSTM
        5. If spam → discard
        6. If non-spam → chunk → embed → add to FAISS → persist

    Args:
        gmail_msg: Gmail API message resource (format='full').
        store: Loaded VectorStore instance.
        persist: Whether to save FAISS index after ingestion.

    Returns:
        Dict with ingestion result:
            status: 'ingested' | 'spam' | 'duplicate' | 'error'
            message_id: str
            subject: str
            sender: str
            classification: dict (if classified)
            chunks_added: int (if ingested)
            error: str (if error)
    """
    result = {
        "status": "error",
        "message_id": "",
        "subject": "",
        "sender": "",
        "classification": None,
        "chunks_added": 0,
        "error": None,
    }

    try:
        # 1. Parse Gmail message
        raw_email = parse_gmail_message(gmail_msg)
        message_id = raw_email["Message_ID"]
        result["message_id"] = message_id
        result["subject"] = raw_email.get("Subject", "")
        result["sender"] = raw_email.get("Sender_Email", "")

        if not message_id:
            result["error"] = "No message_id found"
            result["status"] = "error"
            stats.record_error(result)
            logger.warning("Skipping email with no message_id")
            return result

        # 2. Duplicate check
        if message_id in store._indexed_message_ids:
            result["status"] = "duplicate"
            stats.record_duplicate(result)
            logger.info(f"Duplicate email skipped: {message_id[:30]}")
            return result

        # 3. Normalize
        email_record = normalize_email(raw_email)

        # 4. Classify
        classification = predict_email(
            sender=email_record.sender_email or email_record.sender_name,
            subject=email_record.subject,
            body=email_record.body,
        )
        result["classification"] = classification

        logger.info(
            f"Classified email: subject='{email_record.subject[:50]}' "
            f"label={classification['label']} "
            f"confidence={classification['confidence']:.2%}"
        )

        # 5. Check if spam
        if is_spam(classification):
            result["status"] = "spam"
            stats.record_spam(result)
            logger.info(
                f"Spam email discarded: {email_record.subject[:50]} "
                f"from {email_record.sender_email}"
            )
            return result

        # 6. Chunk and embed
        # Store the classification label in the email record's category
        email_record.category = classification["label"]

        chunks = chunk_email(email_record)

        if not chunks:
            result["error"] = "No chunks generated"
            result["status"] = "error"
            stats.record_error(result)
            return result

        # 7. Add to FAISS index (VectorStore handles embedding + dedup internally)
        added = store.update_index(chunks)

        if added == 0:
            # update_index filtered it as duplicate internally
            result["status"] = "duplicate"
            stats.record_duplicate(result)
            return result

        # 8. Persist
        if persist:
            store.save()
            logger.info(f"Index persisted after ingesting: {message_id[:30]}")

        result["status"] = "ingested"
        result["chunks_added"] = added
        stats.record_ingested(result)

        logger.info(
            f"Email ingested: subject='{email_record.subject[:50]}' "
            f"chunks={added} label={classification['label']}"
        )
        return result

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
        stats.record_error(result)
        logger.error(
            f"Ingestion error for {result.get('message_id', 'unknown')}: {e}",
            exc_info=True,
        )
        return result


def ingest_gmail_messages(
    gmail_messages: list[dict],
    store: VectorStore,
    persist: bool = True,
) -> list[dict]:
    """
    Process multiple Gmail messages through the ingestion pipeline.

    Persists the index once after all messages are processed (not per-message).

    Args:
        gmail_messages: List of Gmail API message resources.
        store: Loaded VectorStore instance.
        persist: Whether to save FAISS index after batch ingestion.

    Returns:
        List of ingestion result dicts.
    """
    results = []
    any_ingested = False

    for msg in gmail_messages:
        # Don't persist per-message; do it once at the end
        result = ingest_gmail_message(msg, store, persist=False)
        results.append(result)

        if result["status"] == "ingested":
            any_ingested = True

    # Persist once after batch
    if persist and any_ingested:
        try:
            store.save()
            logger.info(f"Index persisted after batch ingestion of {len(gmail_messages)} messages")
        except Exception as e:
            logger.error(f"Failed to persist index after batch ingestion: {e}")

    stats.update_sync_time()

    # Summary log
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    logger.info(f"Batch ingestion complete: {statuses}")

    return results


def ensure_classifier_loaded() -> None:
    """Pre-load the classifier at startup to avoid first-request latency."""
    try:
        load_classifier()
        logger.info("BiLSTM classifier pre-loaded for Gmail ingestion")
    except Exception as e:
        logger.warning(f"Failed to pre-load classifier: {e}")

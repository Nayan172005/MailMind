"""
MailMind RAG — Email Normalizer

Converts raw email records (from the Gmail dataset) into a common internal
representation (EmailRecord). Handles missing fields gracefully.
"""

from typing import Union

from rag.schemas import EmailRecord
from rag.utils import get_logger, safe_str, clean_whitespace

logger = get_logger("normalizer")


def _parse_list_field(value) -> list[str]:
    """Parse a field that should be a list (Labels, Attachment_Names)."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        # Handle string representations like "['a', 'b']"
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        # Comma-separated fallback
        if "," in stripped:
            return [s.strip() for s in stripped.split(",") if s.strip()]
        if stripped:
            return [stripped]
    return []


def normalize_email(raw: dict) -> EmailRecord:
    """
    Convert a raw email dict (from the Gmail dataset) into an EmailRecord.

    Mapping from Gmail CSV/parquet schema:
        Message_ID    -> message_id
        Thread_ID     -> thread_id
        Sender_Name   -> sender_name
        Sender_Email  -> sender_email
        Subject       -> subject
        Date          -> date
        Body          -> body
        Labels        -> labels
        Gmail_Category -> category
        Has_Attachment -> has_attachment
        Attachment_Names -> attachment_names
    """
    body = safe_str(raw.get("Body", ""))
    body = clean_whitespace(body)

    return EmailRecord(
        message_id=safe_str(raw.get("Message_ID", "")),
        thread_id=safe_str(raw.get("Thread_ID", "")),
        sender_name=safe_str(raw.get("Sender_Name", "")),
        sender_email=safe_str(raw.get("Sender_Email", "")),
        subject=safe_str(raw.get("Subject", "")),
        date=safe_str(raw.get("Date", "")),
        body=body,
        labels=_parse_list_field(raw.get("Labels", [])),
        category=safe_str(raw.get("Gmail_Category", "")),
        has_attachment=bool(raw.get("Has_Attachment", False)),
        attachment_names=_parse_list_field(raw.get("Attachment_Names", [])),
    )


def normalize_emails(raw_records: list[dict]) -> list[EmailRecord]:
    """Normalize a batch of raw email records."""
    emails = []
    skipped = 0

    for raw in raw_records:
        msg_id = safe_str(raw.get("Message_ID", ""))
        if not msg_id:
            skipped += 1
            continue

        try:
            email = normalize_email(raw)
            emails.append(email)
        except Exception as e:
            logger.warning(f"Skipping email {msg_id[:20]}: {e}")
            skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} emails during normalization")

    logger.info(f"Normalized {len(emails)} emails")
    return emails

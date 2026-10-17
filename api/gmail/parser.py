"""
MailMind Gmail — Message Parser

Converts Gmail API message resources into MailMind's normalized dict format
(compatible with rag.normalizer.normalize_email).

Handles:
- Multipart MIME emails
- HTML-only bodies
- Plain text bodies
- From / To / CC / BCC headers
- Replies and forwards detection
- Attachment metadata
"""

import base64
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

from rag.utils import get_logger, clean_whitespace

logger = get_logger("gmail.parser")


def parse_gmail_message(msg: dict) -> dict:
    """
    Convert a Gmail API message resource into MailMind's raw email dict format.

    The output dict uses the capitalized keys expected by
    rag.normalizer.normalize_email:
        Message_ID, Thread_ID, Sender_Name, Sender_Email, Subject,
        Date, Body, Labels, Gmail_Category, Has_Attachment, Attachment_Names

    Args:
        msg: Gmail API message resource (format='full').

    Returns:
        Dict compatible with normalize_email().
    """
    headers = _extract_headers(msg)
    payload = msg.get("payload", {})

    # Parse sender
    sender_name, sender_email = _parse_address(headers.get("from", ""))

    # Parse recipients
    to_str = headers.get("to", "")
    cc_str = headers.get("cc", "")
    bcc_str = headers.get("bcc", "")

    # Parse subject
    subject = headers.get("subject", "")

    # Parse date — use the email Date header, fall back to Gmail internalDate
    date_str = _parse_date(headers.get("date", ""), msg.get("internalDate"))

    # Extract body (prefer plain text, fall back to HTML)
    body = _extract_body(payload)
    body = clean_whitespace(body) if body else ""

    # Detect attachments
    attachment_names = _extract_attachment_names(payload)
    has_attachment = len(attachment_names) > 0

    # Gmail labels and category
    label_ids = msg.get("labelIds", [])
    gmail_category = _detect_category(label_ids)

    # Gmail message ID (internal) vs RFC Message-ID header
    gmail_id = msg.get("id", "")
    rfc_message_id = headers.get("message-id", gmail_id)
    thread_id = msg.get("threadId", "")

    return {
        "Message_ID": rfc_message_id,
        "Thread_ID": thread_id,
        "Sender_Name": sender_name,
        "Sender_Email": sender_email,
        "Subject": subject,
        "Date": date_str,
        "Body": body,
        "Labels": label_ids,
        "Gmail_Category": gmail_category,
        "Has_Attachment": has_attachment,
        "Attachment_Names": attachment_names,
        # Extra fields for Gmail-specific tracking
        "_gmail_id": gmail_id,
        "_to": to_str,
        "_cc": cc_str,
        "_bcc": bcc_str,
        "_source": "gmail",
    }


def _extract_headers(msg: dict) -> dict:
    """Extract headers into a lowercase-key dict for easy lookup."""
    headers = {}
    payload = msg.get("payload", {})
    for header in payload.get("headers", []):
        name = header.get("name", "").lower()
        value = header.get("value", "")
        headers[name] = value
    return headers


def _parse_address(address_str: str) -> tuple[str, str]:
    """
    Parse a From/To header into (name, email).

    Handles formats like:
        'John Doe <john@example.com>'
        '<john@example.com>'
        'john@example.com'
        '"John Doe" <john@example.com>'
    """
    if not address_str:
        return "", ""

    # Match: "Name" <email> or Name <email>
    match = re.match(r'^["\']?(.+?)["\']?\s*<([^>]+)>', address_str)
    if match:
        name = match.group(1).strip().strip('"').strip("'")
        email = match.group(2).strip()
        return name, email

    # Match: <email>
    match = re.match(r'^<([^>]+)>', address_str)
    if match:
        return "", match.group(1).strip()

    # Plain email
    if "@" in address_str:
        return "", address_str.strip()

    return address_str.strip(), ""


def _parse_date(date_header: str, internal_date_ms: Optional[str]) -> str:
    """
    Parse the email date. Prefers the RFC 2822 Date header.
    Falls back to Gmail's internalDate (milliseconds since epoch).
    Returns ISO 8601 format string.
    """
    if date_header:
        try:
            dt = parsedate_to_datetime(date_header)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

    if internal_date_ms:
        try:
            ts = int(internal_date_ms) / 1000.0
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

    return ""


def _extract_body(payload: dict) -> str:
    """
    Extract the plain-text body from a Gmail message payload.

    Strategy:
    1. If payload is text/plain, decode and return.
    2. If multipart, recursively search for text/plain.
    3. If only text/html available, strip HTML tags.
    """
    mime_type = payload.get("mimeType", "")

    # Direct text/plain body
    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return _decode_base64(data)

    # Multipart: recurse into parts
    parts = payload.get("parts", [])
    if parts:
        # First pass: look for text/plain
        for part in parts:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return _decode_base64(data)

        # Second pass: recurse into multipart sub-parts
        for part in parts:
            if part.get("mimeType", "").startswith("multipart/"):
                result = _extract_body(part)
                if result:
                    return result

        # Third pass: fall back to text/html
        for part in parts:
            if part.get("mimeType") == "text/html":
                data = part.get("body", {}).get("data", "")
                if data:
                    html = _decode_base64(data)
                    return _html_to_text(html)

    # Direct text/html body (non-multipart)
    if mime_type == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            html = _decode_base64(data)
            return _html_to_text(html)

    return ""


def _extract_attachment_names(payload: dict) -> list[str]:
    """Extract attachment filenames from a Gmail message payload."""
    names = []
    parts = payload.get("parts", [])

    for part in parts:
        filename = part.get("filename", "")
        if filename:
            names.append(filename)

        # Recurse into nested parts
        if part.get("parts"):
            names.extend(_extract_attachment_names(part))

    return names


def _detect_category(label_ids: list[str]) -> str:
    """Map Gmail label IDs to a human-readable category."""
    category_map = {
        "CATEGORY_PRIMARY": "Primary",
        "CATEGORY_SOCIAL": "Social",
        "CATEGORY_PROMOTIONS": "Promotions",
        "CATEGORY_UPDATES": "Updates",
        "CATEGORY_FORUMS": "Forums",
    }
    for label in label_ids:
        if label in category_map:
            return category_map[label]
    return ""


def _decode_base64(data: str) -> str:
    """Decode Gmail's URL-safe base64 encoded content."""
    try:
        # Gmail uses URL-safe base64
        padded = data + "=" * (4 - len(data) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        return decoded.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Failed to decode base64 content: {e}")
        return ""


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "head"]):
            element.decompose()

        text = soup.get_text(separator="\n")
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        return text

    except ImportError:
        # Fallback: basic regex HTML stripping
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

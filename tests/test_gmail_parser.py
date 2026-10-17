"""
Tests for Gmail message parser.

Tests the conversion of Gmail API message resources into
MailMind's normalized email dict format.
"""

import pytest
from api.gmail.parser import (
    parse_gmail_message,
    _parse_address,
    _parse_date,
    _extract_body,
    _extract_headers,
    _extract_attachment_names,
    _decode_base64,
    _html_to_text,
    _detect_category,
)
import base64


# ─── Helper to build Gmail-like message dicts ────────────────────────

def _make_header(name: str, value: str) -> dict:
    return {"name": name, "value": value}


def _make_plain_text_msg(
    from_header: str = "John Doe <john@example.com>",
    subject: str = "Test Subject",
    date: str = "Thu, 08 Oct 2026 10:30:00 +0000",
    body: str = "Hello, this is a test email body.",
    msg_id: str = "msg_001",
    thread_id: str = "thread_001",
    label_ids: list = None,
    internal_date: str = None,
    message_id_header: str = "<unique@example.com>",
) -> dict:
    """Build a minimal plain-text Gmail message dict."""
    body_encoded = base64.urlsafe_b64encode(body.encode()).decode()

    headers = [
        _make_header("From", from_header),
        _make_header("Subject", subject),
        _make_header("Date", date),
    ]
    if message_id_header:
        headers.append(_make_header("Message-ID", message_id_header))

    msg = {
        "id": msg_id,
        "threadId": thread_id,
        "labelIds": label_ids or ["INBOX"],
        "payload": {
            "mimeType": "text/plain",
            "headers": headers,
            "body": {"data": body_encoded},
        },
    }
    if internal_date:
        msg["internalDate"] = internal_date

    return msg


def _make_multipart_msg(
    plain_body: str = None,
    html_body: str = None,
    attachments: list = None,
) -> dict:
    """Build a multipart Gmail message dict."""
    parts = []

    if plain_body:
        encoded = base64.urlsafe_b64encode(plain_body.encode()).decode()
        parts.append({
            "mimeType": "text/plain",
            "body": {"data": encoded},
        })

    if html_body:
        encoded = base64.urlsafe_b64encode(html_body.encode()).decode()
        parts.append({
            "mimeType": "text/html",
            "body": {"data": encoded},
        })

    if attachments:
        for filename in attachments:
            parts.append({
                "mimeType": "application/octet-stream",
                "filename": filename,
                "body": {"attachmentId": "att_123", "size": 1024},
            })

    return {
        "id": "msg_multi",
        "threadId": "thread_multi",
        "labelIds": ["INBOX"],
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                _make_header("From", "sender@test.com"),
                _make_header("Subject", "Multipart Test"),
                _make_header("Date", "Thu, 08 Oct 2026 10:30:00 +0000"),
                _make_header("Message-ID", "<multi@test.com>"),
            ],
            "parts": parts,
        },
    }


# ─── Tests: Address Parsing ──────────────────────────────────────────

class TestAddressParsing:
    def test_name_and_email(self):
        name, email = _parse_address("John Doe <john@example.com>")
        assert name == "John Doe"
        assert email == "john@example.com"

    def test_quoted_name(self):
        name, email = _parse_address('"Jane Smith" <jane@test.com>')
        assert name == "Jane Smith"
        assert email == "jane@test.com"

    def test_email_only_brackets(self):
        name, email = _parse_address("<nobody@test.com>")
        assert name == ""
        assert email == "nobody@test.com"

    def test_plain_email(self):
        name, email = _parse_address("user@domain.com")
        assert name == ""
        assert email == "user@domain.com"

    def test_empty(self):
        name, email = _parse_address("")
        assert name == ""
        assert email == ""

    def test_via_format(self):
        """Test the 'Name via Group <email>' format from classifier demo."""
        name, email = _parse_address(
            "'Helpdesk CDC' via VIT Placement Group <cdc@vit.ac.in>"
        )
        assert email == "cdc@vit.ac.in"
        assert "CDC" in name or "Placement" in name


# ─── Tests: Date Parsing ─────────────────────────────────────────────

class TestDateParsing:
    def test_rfc_2822_date(self):
        result = _parse_date("Thu, 08 Oct 2026 10:30:00 +0000", None)
        assert result != ""
        assert "2026" in result

    def test_internal_date_fallback(self):
        """When Date header is empty, use Gmail's internalDate."""
        # 1696745400000 ms = some timestamp
        result = _parse_date("", "1696745400000")
        assert result != ""

    def test_empty_date(self):
        result = _parse_date("", None)
        assert result == ""

    def test_iso_format(self):
        """Verify output is ISO 8601."""
        result = _parse_date("Thu, 08 Oct 2026 10:30:00 +0000", None)
        assert "T" in result  # ISO format has T separator


# ─── Tests: Body Extraction ──────────────────────────────────────────

class TestBodyExtraction:
    def test_plain_text_body(self):
        body = "Hello world"
        encoded = base64.urlsafe_b64encode(body.encode()).decode()
        payload = {
            "mimeType": "text/plain",
            "body": {"data": encoded},
        }
        assert _extract_body(payload) == "Hello world"

    def test_multipart_plain(self):
        body = "Plain text body"
        encoded = base64.urlsafe_b64encode(body.encode()).decode()
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": encoded}},
                {"mimeType": "text/html", "body": {"data": "not_used"}},
            ],
        }
        result = _extract_body(payload)
        assert result == "Plain text body"

    def test_html_only_body(self):
        html = "<html><body><p>HTML only content</p></body></html>"
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        payload = {
            "mimeType": "text/html",
            "body": {"data": encoded},
        }
        result = _extract_body(payload)
        assert "HTML only content" in result

    def test_multipart_html_fallback(self):
        """When no plain text part exists, fall back to HTML."""
        html = "<p>Fallback HTML</p>"
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/html", "body": {"data": encoded}},
            ],
        }
        result = _extract_body(payload)
        assert "Fallback HTML" in result

    def test_empty_payload(self):
        payload = {"mimeType": "text/plain", "body": {}}
        assert _extract_body(payload) == ""

    def test_nested_multipart(self):
        """Test nested multipart structure."""
        body = "Nested plain text"
        encoded = base64.urlsafe_b64encode(body.encode()).decode()
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": encoded}},
                    ],
                },
            ],
        }
        result = _extract_body(payload)
        assert "Nested plain text" in result


# ─── Tests: Attachment Detection ─────────────────────────────────────

class TestAttachments:
    def test_no_attachments(self):
        payload = {"mimeType": "text/plain", "body": {"data": "test"}}
        assert _extract_attachment_names(payload) == []

    def test_single_attachment(self):
        payload = {
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "test"}},
                {
                    "mimeType": "application/pdf",
                    "filename": "report.pdf",
                    "body": {"attachmentId": "att1", "size": 5000},
                },
            ]
        }
        names = _extract_attachment_names(payload)
        assert names == ["report.pdf"]

    def test_multiple_attachments(self):
        payload = {
            "parts": [
                {"mimeType": "application/pdf", "filename": "a.pdf", "body": {}},
                {"mimeType": "image/png", "filename": "b.png", "body": {}},
            ]
        }
        names = _extract_attachment_names(payload)
        assert len(names) == 2
        assert "a.pdf" in names
        assert "b.png" in names


# ─── Tests: Full Message Parsing ─────────────────────────────────────

class TestFullParsing:
    def test_plain_text_message(self):
        msg = _make_plain_text_msg()
        result = parse_gmail_message(msg)

        assert result["Message_ID"] == "<unique@example.com>"
        assert result["Thread_ID"] == "thread_001"
        assert result["Sender_Name"] == "John Doe"
        assert result["Sender_Email"] == "john@example.com"
        assert result["Subject"] == "Test Subject"
        assert result["Date"] != ""
        assert "test email body" in result["Body"]
        assert result["_gmail_id"] == "msg_001"
        assert result["_source"] == "gmail"

    def test_multipart_message(self):
        msg = _make_multipart_msg(
            plain_body="This is the plain text version.",
            html_body="<p>This is HTML</p>",
            attachments=["doc.pdf"],
        )
        result = parse_gmail_message(msg)

        assert "plain text version" in result["Body"]
        assert result["Has_Attachment"] is True
        assert "doc.pdf" in result["Attachment_Names"]

    def test_html_only_message(self):
        msg = _make_multipart_msg(
            html_body="<html><body><h1>Important</h1><p>HTML email</p></body></html>",
        )
        result = parse_gmail_message(msg)
        assert "HTML email" in result["Body"]

    def test_gmail_category_detection(self):
        msg = _make_plain_text_msg(
            label_ids=["INBOX", "CATEGORY_PROMOTIONS"]
        )
        result = parse_gmail_message(msg)
        assert result["Gmail_Category"] == "Promotions"

    def test_internal_date_fallback(self):
        """Use internalDate when Date header is missing."""
        msg = _make_plain_text_msg(date="")
        msg["internalDate"] = "1696745400000"
        result = parse_gmail_message(msg)
        assert result["Date"] != ""

    def test_no_message_id_header(self):
        """Use Gmail ID when Message-ID header is missing."""
        msg = _make_plain_text_msg(message_id_header=None)
        result = parse_gmail_message(msg)
        assert result["Message_ID"] == "msg_001"  # falls back to Gmail ID


# ─── Tests: HTML to Text ─────────────────────────────────────────────

class TestHtmlToText:
    def test_basic_html(self):
        result = _html_to_text("<p>Hello <b>World</b></p>")
        assert "Hello" in result
        assert "World" in result

    def test_script_removal(self):
        html = "<p>Text</p><script>alert('xss')</script>"
        result = _html_to_text(html)
        assert "alert" not in result
        assert "Text" in result

    def test_style_removal(self):
        html = "<style>.red{color:red}</style><p>Content</p>"
        result = _html_to_text(html)
        assert "Content" in result
        assert "color" not in result


# ─── Tests: Category Detection ───────────────────────────────────────

class TestCategoryDetection:
    def test_primary(self):
        assert _detect_category(["INBOX", "CATEGORY_PRIMARY"]) == "Primary"

    def test_social(self):
        assert _detect_category(["CATEGORY_SOCIAL"]) == "Social"

    def test_no_category(self):
        assert _detect_category(["INBOX", "UNREAD"]) == ""


# ─── Tests: Base64 Decoding ──────────────────────────────────────────

class TestBase64Decode:
    def test_valid_decode(self):
        original = "Hello, World!"
        encoded = base64.urlsafe_b64encode(original.encode()).decode()
        assert _decode_base64(encoded) == original

    def test_padding(self):
        """Test handling of missing padding."""
        original = "Test"
        encoded = base64.urlsafe_b64encode(original.encode()).decode().rstrip("=")
        assert _decode_base64(encoded) == original

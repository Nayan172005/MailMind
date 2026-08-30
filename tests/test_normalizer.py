"""Tests for rag.normalizer"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.normalizer import normalize_email, normalize_emails


def test_normalize_basic_email():
    """Test normalizing a basic email record."""
    raw = {
        "Message_ID": "abc123",
        "Thread_ID": "thread1",
        "Sender_Name": "John Doe",
        "Sender_Email": "john@example.com",
        "Subject": "Test Subject",
        "Date": "Mon, 1 Jan 2026 10:00:00 GMT",
        "Body": "Hello, this is a test email.",
        "Labels": ["INBOX", "UNREAD"],
        "Gmail_Category": "Primary",
        "Has_Attachment": False,
        "Attachment_Names": [],
    }

    email = normalize_email(raw)
    assert email.message_id == "abc123"
    assert email.sender_name == "John Doe"
    assert email.subject == "Test Subject"
    assert email.body == "Hello, this is a test email."
    assert email.labels == ["INBOX", "UNREAD"]
    assert email.has_attachment is False


def test_normalize_missing_fields():
    """Test that missing fields get safe defaults."""
    raw = {"Message_ID": "xyz789"}

    email = normalize_email(raw)
    assert email.message_id == "xyz789"
    assert email.subject == ""
    assert email.body == ""
    assert email.labels == []


def test_normalize_string_labels():
    """Test parsing labels from a string representation."""
    raw = {
        "Message_ID": "test1",
        "Labels": "['INBOX', 'UNREAD']",
    }

    email = normalize_email(raw)
    assert email.labels == ["INBOX", "UNREAD"]


def test_normalize_batch():
    """Test batch normalization."""
    records = [
        {"Message_ID": "a1", "Subject": "First"},
        {"Message_ID": "a2", "Subject": "Second"},
        {"Message_ID": "", "Subject": "No ID"},  # should be skipped
    ]

    emails = normalize_emails(records)
    assert len(emails) == 2
    assert emails[0].message_id == "a1"
    assert emails[1].message_id == "a2"


def test_normalize_whitespace_body():
    """Test that body whitespace is cleaned."""
    raw = {
        "Message_ID": "ws1",
        "Body": "Hello\r\n\r\n\r\n\r\nWorld   test",
    }

    email = normalize_email(raw)
    assert "\r" not in email.body
    assert "\n\n\n" not in email.body

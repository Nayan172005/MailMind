"""Tests for rag.loader"""

import os
import sys
import pytest
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loader import load_emails


def test_load_parquet():
    """Test loading from the real parquet dataset."""
    from rag.config import settings
    if not settings.dataset_abs_path.exists():
        pytest.skip("Dataset not found")

    records = load_emails()
    assert len(records) > 0
    assert "Message_ID" in records[0]
    assert "Subject" in records[0]
    assert "Body" in records[0]


def test_load_json():
    """Test loading from a temporary JSON file."""
    data = [
        {
            "Message_ID": "test123",
            "Subject": "Test Email",
            "Body": "Hello world",
            "Sender_Name": "Test",
            "Sender_Email": "test@example.com",
            "Date": "2026-01-01",
        }
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f)
        tmp_path = f.name

    try:
        records = load_emails(tmp_path)
        assert len(records) == 1
        assert records[0]["Message_ID"] == "test123"
    finally:
        os.unlink(tmp_path)


def test_load_missing_file():
    """Test that loading a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_emails("/nonexistent/path/data.parquet")


def test_load_unsupported_format():
    """Test that unsupported formats raise ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        f.write(b"<data/>")
        tmp_path = f.name

    try:
        with pytest.raises(ValueError):
            load_emails(tmp_path)
    finally:
        os.unlink(tmp_path)

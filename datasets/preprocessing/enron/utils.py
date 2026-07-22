"""Shared utilities for email preprocessing."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging.

    Args:
        level: Logging level name, such as ``INFO`` or ``DEBUG``.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOGGER_FORMAT,
    )


def require_pyarrow() -> None:
    """Raise a helpful error if pyarrow is not available."""

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for Parquet output. Install it with: pip install pyarrow"
        ) from exc


def read_email_csv(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    """Read the raw Enron CSV in chunks.

    Args:
        path: Location of the raw ``emails.csv`` file.
        chunk_size: Number of rows per chunk.

    Yields:
        DataFrame chunks containing ``file`` and ``message`` columns.
    """

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    return pd.read_csv(
        path,
        usecols=["file", "message"],
        dtype={"file": "string", "message": "string"},
        chunksize=chunk_size,
        keep_default_na=False,
        encoding_errors="replace",
        on_bad_lines="warn",
    )


def safe_text(value: object) -> str:
    """Convert a value to a stripped string without leaking pandas nulls."""

    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_whitespace(text: object) -> str:
    """Normalize whitespace to single spaces.

    Args:
        text: Text-like value.

    Returns:
        Text with leading/trailing whitespace removed and internal whitespace collapsed.
    """

    return " ".join(safe_text(text).split())


def normalize_body_text(text: object) -> str:
    """Normalize body whitespace while preserving paragraph breaks.

    Tabs and repeated spaces are collapsed inside lines. Runs of blank lines are
    collapsed to a single paragraph break.

    Args:
        text: Email body text.

    Returns:
        Normalized body text suitable for search and summarization.
    """

    value = safe_text(text).replace("\r\n", "\n").replace("\r", "\n")
    if not value:
        return ""

    normalized_lines: list[str] = []
    previous_blank = False

    for line in value.split("\n"):
        cleaned_line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        is_blank = cleaned_line == ""

        if is_blank:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue

        normalized_lines.append(cleaned_line)
        previous_blank = False

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines)


def sha256_text(text: object) -> str:
    """Return a stable SHA-256 hash for a text-like value."""

    return hashlib.sha256(safe_text(text).encode("utf-8", errors="ignore")).hexdigest()


def strip_leading_header_block(text: object) -> str:
    """Remove a leading RFC822-like header block accidentally left in a body.

    Some malformed Enron messages contain a second header block at the start of
    the extracted body. This function only strips a leading block when multiple
    common email header lines are found before the first blank line.

    Args:
        text: Extracted email body.

    Returns:
        Body text without an accidental leading header block.
    """

    body = safe_text(text)
    if not body:
        return ""

    header_names = {
        "bcc",
        "cc",
        "content-transfer-encoding",
        "content-type",
        "date",
        "from",
        "message-id",
        "mime-version",
        "subject",
        "to",
        "x-bcc",
        "x-cc",
        "x-fileName".lower(),
        "x-folder",
        "x-from",
        "x-origin",
        "x-to",
    }

    lines = body.splitlines()
    header_count = 0
    first_blank_index: int | None = None

    for index, line in enumerate(lines[:50]):
        stripped = line.strip()
        if not stripped:
            first_blank_index = index
            break
        if line[:1].isspace() and header_count:
            continue
        name, separator, _ = stripped.partition(":")
        if separator and name.lower() in header_names:
            header_count += 1
            continue
        break

    if first_blank_index is not None and header_count >= 2:
        return "\n".join(lines[first_blank_index + 1 :]).strip()
    return body

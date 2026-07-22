"""Shared utilities for SpamAssassin preprocessing."""

from __future__ import annotations

import base64
import binascii
import codecs
import logging
import quopri
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
ENCODED_WORD_PATTERN = re.compile(
    r"=\?(?P<charset>[^?\s]+)\?(?P<encoding>[bqBQ])\?(?P<payload>.*?)\?="
)


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
    """Raise a helpful error when the Parquet dependency is unavailable."""

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for Parquet output. Install it with: pip install pyarrow"
        ) from exc


def read_spamassassin_csv(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    """Read the raw SpamAssassin CSV in bounded-memory chunks.

    Args:
        path: Location of ``spam_assassin.csv``.
        chunk_size: Number of rows returned in each chunk.

    Returns:
        An iterable of DataFrame chunks containing ``text`` and ``target``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    return pd.read_csv(
        path,
        usecols=["text", "target"],
        dtype={"text": "string", "target": "string"},
        chunksize=chunk_size,
        keep_default_na=False,
        encoding_errors="replace",
        on_bad_lines="warn",
    )


def safe_text(value: object) -> str:
    """Return a string value without converting missing values to text."""

    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_model_text(text: object) -> str:
    """Normalize whitespace while preserving wording, punctuation, and URLs.

    Args:
        text: Text-like content.

    Returns:
        A single-space-normalized string suitable for a text classifier.
    """

    return " ".join(safe_text(text).split())


def count_urls(text: object) -> int:
    """Count HTTP(S) and ``www.`` URLs without altering the supplied text."""

    return len(URL_PATTERN.findall(safe_text(text)))


def parse_charset(content_type: str) -> str | None:
    """Extract a declared charset from a Content-Type header value."""

    match = re.search(
        r"(?i)\bcharset\s*=\s*(?:\"(?P<quoted>[^\"]+)\"|'(?P<single>[^']+)'|(?P<bare>[^\s;]+))",
        content_type,
    )
    if match is None:
        return None
    return next(value for value in match.groups() if value is not None).strip()


def decode_bytes(payload: bytes, declared_charset: str | None) -> str:
    """Decode bytes using the declared charset and safe common fallbacks.

    Args:
        payload: Bytes to decode.
        declared_charset: Charset extracted from message metadata, when present.

    Returns:
        Decoded text. Replacement decoding is always available as a final fallback.
    """

    candidates = [declared_charset, "utf-8", "latin-1", "cp1252"]
    attempted: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = candidate.strip().strip("\"'").replace("_", "-")
        if not normalized or normalized.lower() in attempted:
            continue
        attempted.add(normalized.lower())
        try:
            codecs.lookup(normalized)
            return payload.decode(normalized)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")


def decode_transfer_encoding(
    body: str,
    transfer_encoding: str,
    declared_charset: str | None,
) -> str:
    """Decode quoted-printable or base64 body data without raising exceptions.

    Args:
        body: Raw message body.
        transfer_encoding: Content-Transfer-Encoding header value.
        declared_charset: Charset inferred from Content-Type.

    Returns:
        Decoded body text, or the original text when decoding is not applicable.
    """

    encoding = safe_text(transfer_encoding).lower().split(";", maxsplit=1)[0].strip()
    if encoding not in {"quoted-printable", "base64"}:
        return body

    try:
        if encoding == "quoted-printable":
            decoded = quopri.decodestring(body.encode("latin-1", errors="replace"))
        else:
            compact = re.sub(r"\s+", "", body)
            compact += "=" * (-len(compact) % 4)
            decoded = base64.b64decode(compact, validate=False)
        return decode_bytes(decoded, declared_charset)
    except (ValueError, UnicodeError, binascii.Error):
        return body


def decode_encoded_words(value: str) -> str:
    """Decode RFC 2047-style encoded words without strict email-header parsing."""

    def replace(match: re.Match[str]) -> str:
        charset = match.group("charset")
        encoding = match.group("encoding").lower()
        payload = match.group("payload")
        try:
            if encoding == "b":
                raw = base64.b64decode(payload + "=" * (-len(payload) % 4), validate=False)
            else:
                raw = quopri.decodestring(payload.replace("_", " ").encode("latin-1"))
            return decode_bytes(raw, charset)
        except (ValueError, UnicodeError, binascii.Error):
            return match.group(0)

    return ENCODED_WORD_PATTERN.sub(replace, safe_text(value))


class _HTMLTextExtractor(HTMLParser):
    """Tolerant HTML-to-text extractor that retains URLs from attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() not in {"a", "area", "img"}:
            return
        for name, value in attrs:
            if name.lower() in {"href", "src"} and value:
                self.parts.append(f" {value} ")


def html_to_text(text: str) -> str:
    """Extract readable text from HTML while retaining linked URLs.

    Plain text is returned unchanged. Parser errors are intentionally ignored so
    malformed spam messages cannot stop preprocessing.
    """

    if "<" not in text or ">" not in text:
        return text

    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(text)
        extractor.close()
    except Exception:  # pragma: no cover - HTMLParser is intentionally tolerant.
        return text
    extracted = " ".join(extractor.parts)
    return extracted if extracted.strip() else text


def remove_mime_artifacts(text: str) -> str:
    """Remove MIME part delimiters and embedded content metadata from body text."""

    artifact_pattern = re.compile(
        r"(?i)^(?:mime-version|content-type|content-transfer-encoding|content-disposition):"
    )
    retained: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"--[^\s]{1,200}(?:--)?", stripped):
            continue
        if artifact_pattern.match(stripped):
            continue
        retained.append(line)
    return "\n".join(retained)

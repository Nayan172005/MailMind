"""Fault-tolerant, SpamAssassin-oriented raw email parser."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from tqdm import tqdm

from .utils import (
    decode_encoded_words,
    decode_transfer_encoding,
    html_to_text,
    normalize_model_text,
    parse_charset,
    remove_mime_artifacts,
    safe_text,
)


logger = logging.getLogger(__name__)

# These are deliberately bounded to mail metadata. The flattened-CSV fallback
# must not treat ordinary body phrases such as "price:" as headers.
_HEADER_NAME_PATTERN = (
    r"return-path|delivered-to|delivery-date|received|date|from|to|cc|bcc|"
    r"subject|message-id|mime-version|content-type|content-transfer-encoding|"
    r"content-disposition|reply-to|sender|errors-to|precedence|references|"
    r"in-reply-to|mail-followup-to|user-agent|x-[a-z0-9-]+|list-[a-z0-9-]+"
)
_FLAT_HEADER_PATTERN = re.compile(
    rf"(?i)(?<!\S)(?P<name>{_HEADER_NAME_PATTERN})\s*:\s*"
)
_CONTENT_TRANSFER_PREFIX = re.compile(
    r"(?is)^(?P<value>quoted-printable|base64|7bit|8bit|binary)\b\s*(?P<body>.*)$"
)
_CONTENT_TYPE_PREFIX = re.compile(
    r"(?is)^(?P<value>(?:text|multipart|application|image|audio|video|message)/[^\s;]+"
    r"(?:\s*;\s*(?:charset|boundary|format|name)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s;]+))*)\s*(?P<body>.*)$"
)
_ADDRESS_OR_ANGLE_PREFIX = re.compile(
    r"(?is)^(?P<value>(?:<[^>]+>|[^\s<>]+@[^\s<>]+))\s+(?P<body>.+)$"
)

# Minimum body length (in characters) to consider a recovered body usable.
_MIN_BODY_LENGTH = 1


@dataclass(slots=True)
class ParsedSpamEmail:
    """Intermediate representation of one SpamAssassin email."""

    sender: str
    subject: str
    content_type: str
    content_transfer_encoding: str
    body: str
    label: int | None
    _parse_success: bool
    _parse_error: str
    _malformed_header_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a dict suitable for a pandas DataFrame."""

        return asdict(self)


@dataclass(slots=True)
class FailedEmailRecord:
    """Details about a single failed email for debugging output."""

    original_row: object
    reason: str
    preview: str


@dataclass(slots=True)
class ParseReport:
    """Per-chunk parsing counters."""

    total: int = 0
    successfully_parsed: int = 0
    failed_parses: int = 0
    recovered_emails: int = 0
    malformed_emails: int = 0
    malformed_header_lines: int = 0

    def add(self, parsed: ParsedSpamEmail, was_recovered: bool = False) -> None:
        """Add a parsed record to this report."""

        self.total += 1
        self.malformed_header_lines += parsed._malformed_header_count
        if parsed._malformed_header_count:
            self.malformed_emails += 1
        if parsed._parse_success:
            self.successfully_parsed += 1
            if was_recovered:
                self.recovered_emails += 1
        else:
            self.failed_parses += 1


class SpamAssassinEmailParser:
    """Parse unreliable raw emails without using RFC address/header parsing."""

    def __init__(self, warning_limit: int = 100) -> None:
        self.warning_limit = max(0, warning_limit)
        self._warnings_emitted = 0
        self._warnings_suppressed = False

    def parse_dataframe(
        self,
        frame: pd.DataFrame,
        chunk_number: int,
    ) -> tuple[pd.DataFrame, ParseReport, list[FailedEmailRecord]]:
        """Parse a source DataFrame and return records, parse counters, and failed details."""

        records: list[dict[str, Any]] = []
        failed_records: list[FailedEmailRecord] = []
        report = ParseReport()
        for row_index, row in tqdm(
            frame.iterrows(),
            total=len(frame),
            desc=f"Parsing SpamAssassin chunk {chunk_number}",
            leave=False,
        ):
            raw_text = row.get("text")
            parsed, was_recovered = self.parse_record(raw_text, row.get("target"), row_index)
            records.append(parsed.to_dict())
            report.add(parsed, was_recovered)
            if not parsed._parse_success:
                preview = safe_text(raw_text)[:300]
                failed_records.append(
                    FailedEmailRecord(
                        original_row=row_index,
                        reason=parsed._parse_error,
                        preview=preview,
                    )
                )
        return pd.DataFrame.from_records(records), report, failed_records

    def parse_record(
        self,
        raw_message: object,
        raw_label: object,
        row_index: object,
    ) -> tuple[ParsedSpamEmail, bool]:
        """Parse one record and convert all malformed input into a failed record.

        Returns:
            A tuple of (parsed email, was_recovered) where was_recovered indicates
            the email required fallback recovery to succeed.
        """

        raw = safe_text(raw_message)
        label = self._parse_label(raw_label)
        if not raw:
            return self._failed("empty message", label), False
        if label is None:
            return self._failed("invalid target label", label), False

        was_recovered = False
        try:
            headers_text, body, malformed_count, used_flat_fallback = self._split_message(raw)
            headers, parsed_malformed_count = self._parse_headers(headers_text)
            malformed_count += parsed_malformed_count

            sender = headers.get("from", "")
            subject = decode_encoded_words(headers.get("subject", ""))
            content_type = headers.get("content-type", "")
            transfer_encoding = headers.get("content-transfer-encoding", "")
            decoded_body = decode_transfer_encoding(
                body,
                transfer_encoding,
                parse_charset(content_type),
            )
            decoded_body = remove_mime_artifacts(decoded_body)
            if "html" in content_type.lower() or re.search(r"(?is)<(?:html|body|div|p|table)\b", decoded_body):
                decoded_body = html_to_text(decoded_body)
            cleaned_body = normalize_model_text(decoded_body)

            # --- Last-resort body recovery ---
            # If normal parsing left the body empty, try recovering from raw text.
            if not cleaned_body:
                cleaned_body, was_recovered = self._recover_body_from_raw(raw)

            if not cleaned_body:
                detail = "empty body after header extraction"
                if used_flat_fallback:
                    detail = "unable to infer body boundary in flattened message"
                self._warn(row_index, detail)
                return self._failed(detail, label, malformed_count), False

            if malformed_count:
                self._warn(row_index, f"ignored {malformed_count} malformed header line(s)")
            return ParsedSpamEmail(
                sender=safe_text(sender),
                subject=safe_text(subject),
                content_type=safe_text(content_type),
                content_transfer_encoding=safe_text(transfer_encoding),
                body=cleaned_body,
                label=label,
                _parse_success=True,
                _parse_error="",
                _malformed_header_count=malformed_count,
            ), was_recovered
        except Exception as exc:  # A malformed email must never stop the run.
            self._warn(row_index, f"unexpected parser error: {type(exc).__name__}")
            return self._failed("unexpected parser error", label), False

    def _split_message(self, raw: str) -> tuple[str, str, int, bool]:
        """Split standard RFC822 text, with support for flattened CSV exports.

        Enhanced with recovery of literal ``\\n`` sequences that can appear when
        CSV serialization escapes real newlines.
        """

        normalized = raw.replace("\r\n", "\n").replace("\r", "\n")

        # Attempt 1: standard RFC822 blank-line split.
        lines = normalized.split("\n")
        for index, line in enumerate(lines):
            if not line.strip():
                return "\n".join(lines[:index]), "\n".join(lines[index + 1 :]), 0, False

        # Attempt 2: recover literal escaped newlines from CSV serialization.
        # Some CSV exports store \\n (two characters) instead of real newlines.
        if "\\n" in normalized:
            unescaped = normalized.replace("\\n", "\n")
            lines2 = unescaped.split("\n")
            for index, line in enumerate(lines2):
                if not line.strip():
                    return "\n".join(lines2[:index]), "\n".join(lines2[index + 1 :]), 0, False

        return self._split_flattened_message(normalized)

    def _split_flattened_message(self, raw: str) -> tuple[str, str, int, bool]:
        """Recover headers from CSV rows where original newlines were flattened.

        This is intentionally a best-effort fallback. It recognises only common
        mail-header names, never tries to validate addresses, and treats an
        unresolvable terminal header as malformed rather than leaking transport
        metadata into the classifier body.

        When no recognisable headers are found, the entire raw text is treated
        as body rather than being discarded.
        """

        matches = list(_FLAT_HEADER_PATTERN.finditer(raw))
        if len(matches) < 2:
            # No recognisable header structure: treat everything as body.
            # This is the correct fallback for NLP—the body text is what matters.
            return "", raw, 0, True

        header_lines: list[str] = []
        for index, match in enumerate(matches):
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            value = raw[match.end() : value_end].strip()
            header_lines.append(f"{match.group('name')}: {value}")

        last_match = matches[-1]
        final_name = last_match.group("name")
        final_value = raw[last_match.end() :].strip()
        header_value, body = self._split_terminal_flat_header(final_name, final_value)
        header_lines[-1] = f"{final_name}: {header_value}"
        return "\n".join(header_lines), body, 0 if body else 1, True

    @staticmethod
    def _split_terminal_flat_header(header_name: str, value: str) -> tuple[str, str]:
        """Separate the final flattened header value from its following body."""

        normalized_name = header_name.lower()
        pattern = None
        if normalized_name == "content-transfer-encoding":
            pattern = _CONTENT_TRANSFER_PREFIX
        elif normalized_name == "content-type":
            pattern = _CONTENT_TYPE_PREFIX

        if pattern is not None:
            match = pattern.match(value)
            if match:
                return match.group("value").strip(), match.group("body").strip()

        address_match = _ADDRESS_OR_ANGLE_PREFIX.match(value)
        if address_match and normalized_name in {
            "sender",
            "errors-to",
            "list-id",
            "list-archive",
            "list-unsubscribe",
            "x-beenthere",
        }:
            return address_match.group("value").strip(), address_match.group("body").strip()

        return value, ""

    @staticmethod
    def _parse_headers(headers_text: str) -> tuple[dict[str, str], int]:
        """Parse manual ``Header: Value`` lines, keeping first occurrences.

        Malformed lines are silently ignored—they never reject the email.
        """

        headers: dict[str, str] = {}
        malformed_count = 0
        previous_name: str | None = None
        for line in headers_text.splitlines():
            if not line.strip():
                continue
            if line[:1].isspace() and previous_name is not None:
                headers[previous_name] = f"{headers[previous_name]} {line.strip()}".strip()
                continue
            name, separator, value = line.partition(":")
            normalized_name = name.strip().lower()
            if not separator or not normalized_name:
                malformed_count += 1
                previous_name = None
                continue
            if normalized_name not in headers:
                headers[normalized_name] = value.strip()
            previous_name = normalized_name
        return headers, malformed_count

    @staticmethod
    def _parse_label(value: object) -> int | None:
        """Return a valid binary label, or ``None`` for malformed values.

        Accepts a wide variety of label formats found in SpamAssassin datasets:
        numeric (0/1), float-like (0.0/1.0), and textual (spam/ham).
        """

        text = safe_text(value).strip().lower()
        if text in {"0", "0.0", "ham"}:
            return 0
        if text in {"1", "1.0", "spam"}:
            return 1
        # Try numeric parsing as a last resort for edge cases like "1.00".
        try:
            numeric = float(text)
            if numeric == 0.0:
                return 0
            if numeric == 1.0:
                return 1
        except (ValueError, OverflowError):
            pass
        return None

    @staticmethod
    def _recover_body_from_raw(raw: str) -> tuple[str, bool]:
        """Last-resort recovery: extract usable text from the raw message.

        When normal header/body splitting leaves the body empty, this method
        strips anything that looks like email headers from the beginning and
        uses whatever text remains. For an NLP pipeline, any usable text is
        better than discarding the entire email.

        Returns:
            A tuple of (cleaned_body, was_recovered).
        """

        # Strip leading header-like lines.
        normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        body_start = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            # Skip blank lines, header-like lines, and continuation lines.
            if not stripped:
                body_start = index + 1
                break
            if ":" in stripped and not stripped.startswith(" "):
                name_part = stripped.split(":", 1)[0].strip().lower()
                # If it looks like a known header name, skip it.
                if re.match(r"^[a-z][a-z0-9-]*$", name_part) and len(name_part) <= 40:
                    continue
            # If we get a line that doesn't look like a header, this is body.
            body_start = index
            break

        body_text = "\n".join(lines[body_start:])
        body_text = remove_mime_artifacts(body_text)
        if re.search(r"(?is)<(?:html|body|div|p|table)\b", body_text):
            body_text = html_to_text(body_text)
        cleaned = normalize_model_text(body_text)

        if len(cleaned) >= _MIN_BODY_LENGTH:
            return cleaned, True
        return "", False

    @staticmethod
    def _failed(
        error: str,
        label: int | None,
        malformed_count: int = 0,
    ) -> ParsedSpamEmail:
        """Build a failed record without raising an exception."""

        return ParsedSpamEmail(
            sender="",
            subject="",
            content_type="",
            content_transfer_encoding="",
            body="",
            label=label,
            _parse_success=False,
            _parse_error=error,
            _malformed_header_count=malformed_count,
        )

    def _warn(self, row_index: object, message: str) -> None:
        """Emit bounded malformed-record warnings."""

        if self._warnings_emitted < self.warning_limit:
            logger.warning("Malformed SpamAssassin email at row %s: %s", row_index, message)
            self._warnings_emitted += 1
            return
        if not self._warnings_suppressed:
            logger.warning(
                "Malformed-email warning limit reached; further warnings are suppressed."
            )
            self._warnings_suppressed = True

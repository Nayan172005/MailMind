"""Fault-tolerant text parser for the Enron email dataset."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd
from tqdm import tqdm

try:
    from .utils import normalize_body_text, safe_text, sha256_text
except ImportError:  # pragma: no cover - supports direct script execution.
    from utils import normalize_body_text, safe_text, sha256_text


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParsedEmail:
    """Structured representation of one Enron email.

    The parser stores header values as raw text. It does not validate or parse
    email addresses because Enron headers frequently violate RFC rules.
    """

    file: str
    message_id: str
    date: str
    sender: str
    receiver: str
    cc: str
    bcc: str
    subject: str
    body: str
    body_length: int
    word_count: int
    has_subject: bool
    is_reply: bool
    is_forward: bool
    _parse_success: bool
    _parse_error: str
    _malformed_header_count: int
    _body_hash: str
    _message_id_key: str
    _file_key: str


@dataclass(slots=True)
class ParseReport:
    """Aggregate parse-health counters for one chunk."""

    total: int = 0
    successfully_parsed: int = 0
    failed_parses: int = 0
    malformed_emails: int = 0
    malformed_header_lines: int = 0

    def add(self, other: "ParseReport") -> None:
        """Add counters from another report."""

        self.total += other.total
        self.successfully_parsed += other.successfully_parsed
        self.failed_parses += other.failed_parses
        self.malformed_emails += other.malformed_emails
        self.malformed_header_lines += other.malformed_header_lines


class EnronEmailParser:
    """Parse malformed Enron messages without RFC header validation."""

    def __init__(self, warning_limit: int = 100) -> None:
        """Initialize the parser.

        Args:
            warning_limit: Maximum malformed-email examples to log at WARNING.
        """

        self.warning_limit = warning_limit
        self._warnings_emitted = 0
        self._warning_suppressed_notice_emitted = False

    def parse_dataframe(
        self,
        frame: pd.DataFrame,
        chunk_number: int | None = None,
    ) -> tuple[pd.DataFrame, ParseReport]:
        """Parse a raw email DataFrame chunk.

        Args:
            frame: DataFrame with ``file`` and ``message`` columns.
            chunk_number: Optional chunk number for progress labeling.

        Returns:
            Tuple of parsed records and parse report.
        """

        label = f"Parsing chunk {chunk_number}" if chunk_number is not None else "Parsing emails"
        records: list[dict[str, object]] = []
        report = ParseReport(total=len(frame))

        for row in tqdm(
            frame.itertuples(index=False),
            total=len(frame),
            desc=label,
            leave=False,
        ):
            parsed = self.parse_record(row.file, row.message)
            records.append(asdict(parsed))

            if parsed._parse_success:
                report.successfully_parsed += 1
            else:
                report.failed_parses += 1
            if parsed._malformed_header_count or not parsed._parse_success:
                report.malformed_emails += 1
                report.malformed_header_lines += parsed._malformed_header_count
                self._warn_malformed(parsed)

        return pd.DataFrame.from_records(records), report

    def parse_record(self, file_path: object, raw_message: object) -> ParsedEmail:
        """Parse a single Enron email without strict RFC semantics.

        Args:
            file_path: Original file path from the Enron CSV.
            raw_message: Raw message text from the CSV.

        Returns:
            ParsedEmail. Missing or malformed fields are represented as empty
            strings and never raise exceptions.
        """

        file_text = safe_text(file_path)
        raw_text = safe_text(raw_message)

        try:
            header_text, body_text, split_error = self._split_headers_body(raw_text)
            headers, malformed_count = self._parse_headers(header_text)
            body = normalize_body_text(body_text)
            subject = self._header(headers, "subject")
            subject_casefold = subject.casefold().lstrip()
            parse_success = split_error == ""

            return ParsedEmail(
                file=file_text,
                message_id=self._header(headers, "message-id"),
                date=self._header(headers, "date"),
                sender=self._header(headers, "from"),
                receiver=self._header(headers, "to"),
                cc=self._header(headers, "cc"),
                bcc=self._header(headers, "bcc"),
                subject=subject,
                body=body,
                body_length=len(body),
                word_count=len(body.split()),
                has_subject=bool(subject),
                is_reply=subject_casefold.startswith("re:"),
                is_forward=subject_casefold.startswith(("fw:", "fwd:")),
                _parse_success=parse_success,
                _parse_error=split_error,
                _malformed_header_count=malformed_count,
                _body_hash=sha256_text(body),
                _message_id_key=self._dedupe_key(self._header(headers, "message-id")),
                _file_key=self._dedupe_key(file_text),
            )
        except Exception as exc:
            logger.exception("Unexpected parser failure for file=%s", file_text)
            return self._empty_failed_record(file_text, f"unexpected parser error: {exc}")

    def _split_headers_body(self, raw_text: str) -> tuple[str, str, str]:
        """Split a raw message at the first blank line.

        Args:
            raw_text: Raw message text.

        Returns:
            Header text, body text, and an empty string on success or an error
            label when the message cannot be split as expected.
        """

        if not raw_text:
            return "", "", "empty message"

        normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        for index, line in enumerate(lines):
            if line.strip() == "":
                return "\n".join(lines[:index]), "\n".join(lines[index + 1 :]), ""

        return normalized, "", "missing header/body separator"

    def _parse_headers(self, header_text: str) -> tuple[dict[str, str], int]:
        """Parse simple ``Header: Value`` lines and ignore malformed lines."""

        headers: dict[str, str] = {}
        malformed_count = 0

        for line in header_text.splitlines():
            if not line.strip():
                continue
            name, separator, value = line.partition(":")
            normalized_name = name.strip().lower()
            if not separator or not normalized_name:
                malformed_count += 1
                continue
            if normalized_name not in headers:
                headers[normalized_name] = value.strip()

        return headers, malformed_count

    def _header(self, headers: dict[str, str], name: str) -> str:
        """Return a header value or an empty string."""

        return headers.get(name, "")

    def _dedupe_key(self, value: str) -> str:
        """Return a stable case-insensitive deduplication key."""

        return " ".join(value.split()).casefold()

    def _empty_failed_record(self, file_text: str, error: str) -> ParsedEmail:
        """Build a failed record without raising."""

        return ParsedEmail(
            file=file_text,
            message_id="",
            date="",
            sender="",
            receiver="",
            cc="",
            bcc="",
            subject="",
            body="",
            body_length=0,
            word_count=0,
            has_subject=False,
            is_reply=False,
            is_forward=False,
            _parse_success=False,
            _parse_error=error,
            _malformed_header_count=0,
            _body_hash=sha256_text(""),
            _message_id_key="",
            _file_key=self._dedupe_key(file_text),
        )

    def _warn_malformed(self, parsed: ParsedEmail) -> None:
        """Emit bounded warnings for malformed emails."""

        if self._warnings_emitted < self.warning_limit:
            logger.warning(
                "Malformed email tolerated: file=%s parse_success=%s error=%s "
                "malformed_header_lines=%s",
                parsed.file,
                parsed._parse_success,
                parsed._parse_error or "none",
                parsed._malformed_header_count,
            )
            self._warnings_emitted += 1
            return

        if not self._warning_suppressed_notice_emitted:
            logger.warning(
                "Additional malformed-email warnings suppressed after %s examples",
                self.warning_limit,
            )
            self._warning_suppressed_notice_emitted = True

"""Cleaning and streaming deduplication for parsed Enron emails."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

try:
    from .config import PreprocessingConfig
    from .utils import normalize_body_text, safe_text, sha256_text, strip_leading_header_block
except ImportError:  # pragma: no cover - supports direct script execution.
    from config import PreprocessingConfig
    from utils import normalize_body_text, safe_text, sha256_text, strip_leading_header_block


logger = logging.getLogger(__name__)


PUBLIC_COLUMNS = [
    "file",
    "message_id",
    "date",
    "sender",
    "receiver",
    "cc",
    "bcc",
    "subject",
    "body",
    "body_length",
    "word_count",
    "has_subject",
    "is_reply",
    "is_forward",
]


@dataclass(slots=True)
class CleaningReport:
    """Summary of cleaning operations."""

    input_count: int = 0
    duplicate_bodies_removed: int = 0
    duplicate_ids_removed: int = 0
    duplicate_files_removed: int = 0
    empty_bodies_removed: int = 0
    short_bodies_removed: int = 0
    output_count: int = 0

    def add(self, other: "CleaningReport") -> None:
        """Add counters from another cleaning report."""

        self.input_count += other.input_count
        self.duplicate_bodies_removed += other.duplicate_bodies_removed
        self.duplicate_ids_removed += other.duplicate_ids_removed
        self.duplicate_files_removed += other.duplicate_files_removed
        self.empty_bodies_removed += other.empty_bodies_removed
        self.short_bodies_removed += other.short_bodies_removed
        self.output_count += other.output_count


class EmailCleaner:
    """Clean parsed emails and deduplicate records across chunks."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._seen_body_hashes: set[str] = set()
        self._seen_message_ids: set[str] = set()
        self._seen_file_paths: set[str] = set()

    def clean_chunk(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """Clean a parsed chunk and apply global deduplication.

        Args:
            frame: Parsed email records for one chunk.

        Returns:
            Cleaned public records and a report for the chunk.
        """

        report = CleaningReport(input_count=len(frame))
        if frame.empty:
            return self.public_columns(frame), report

        cleaned = frame.copy()
        logger.info("Cleaning chunk with %s parsed emails", len(cleaned))

        self._normalize_columns(cleaned)

        before = len(cleaned)
        cleaned = cleaned[cleaned["body"].str.len() > 0]
        report.empty_bodies_removed = before - len(cleaned)

        before = len(cleaned)
        cleaned = cleaned[cleaned["body_length"] >= self.config.min_body_chars]
        report.short_bodies_removed = before - len(cleaned)

        cleaned, removed = self._drop_seen(
            cleaned,
            key_column="_body_hash",
            seen=self._seen_body_hashes,
            ignore_empty=False,
        )
        report.duplicate_bodies_removed = removed

        cleaned, removed = self._drop_seen(
            cleaned,
            key_column="_message_id_key",
            seen=self._seen_message_ids,
            ignore_empty=True,
        )
        report.duplicate_ids_removed = removed

        cleaned, removed = self._drop_seen(
            cleaned,
            key_column="_file_key",
            seen=self._seen_file_paths,
            ignore_empty=True,
        )
        report.duplicate_files_removed = removed

        cleaned = cleaned.reset_index(drop=True)
        report.output_count = len(cleaned)
        logger.info("Cleaned chunk retained %s emails", report.output_count)
        return self.public_columns(cleaned), report

    def _normalize_columns(self, frame: pd.DataFrame) -> None:
        """Normalize body text and recompute body metadata in-place."""

        frame["body"] = (
            frame["body"]
            .fillna("")
            .map(strip_leading_header_block)
            .map(normalize_body_text)
        )
        frame["body_length"] = frame["body"].str.len()
        frame["word_count"] = frame["body"].map(lambda value: len(value.split()))
        frame["_body_hash"] = frame["body"].map(sha256_text)

        for column in ["file", "message_id", "date", "sender", "receiver", "cc", "bcc", "subject"]:
            frame[column] = frame[column].fillna("").map(safe_text)

        subject_casefold = frame["subject"].str.casefold().str.lstrip()
        frame["has_subject"] = frame["subject"].astype(bool)
        frame["is_reply"] = subject_casefold.str.startswith("re:", na=False)
        frame["is_forward"] = subject_casefold.str.startswith(("fw:", "fwd:"), na=False)
        frame["_message_id_key"] = frame["message_id"].map(self._dedupe_key)
        frame["_file_key"] = frame["file"].map(self._dedupe_key)

    def _drop_seen(
        self,
        frame: pd.DataFrame,
        key_column: str,
        seen: set[str],
        ignore_empty: bool,
    ) -> tuple[pd.DataFrame, int]:
        """Remove rows whose dedupe key has already been seen."""

        if frame.empty:
            return frame, 0

        keep_mask: list[bool] = []
        removed = 0

        for key in frame[key_column].tolist():
            text_key = safe_text(key)
            if ignore_empty and not text_key:
                keep_mask.append(True)
                continue
            if text_key in seen:
                keep_mask.append(False)
                removed += 1
                continue
            seen.add(text_key)
            keep_mask.append(True)

        return frame.loc[keep_mask], removed

    @staticmethod
    def _dedupe_key(value: object) -> str:
        """Return a stable case-insensitive deduplication key."""

        return " ".join(safe_text(value).split()).casefold()

    @staticmethod
    def public_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Return output columns without internal parser/cleaner fields."""

        for column in PUBLIC_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame.loc[:, PUBLIC_COLUMNS].copy()

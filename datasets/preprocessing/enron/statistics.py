"""Streaming statistics and final report rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

try:
    from .cleaner import CleaningReport
    from .parser import ParseReport
except ImportError:  # pragma: no cover - supports direct script execution.
    from cleaner import CleaningReport
    from parser import ParseReport


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessingReport:
    """Final preprocessing report."""

    raw_emails: int = 0
    successfully_parsed: int = 0
    failed_parses: int = 0
    malformed_emails: int = 0
    malformed_header_lines: int = 0
    cleaned_emails: int = 0
    duplicate_bodies_removed: int = 0
    duplicate_ids_removed: int = 0
    duplicate_files_removed: int = 0
    empty_bodies_removed: int = 0
    short_bodies_removed: int = 0
    body_length_sum: int = 0
    word_count_sum: int = 0
    missing_subjects: int = 0
    missing_senders: int = 0
    missing_receivers: int = 0
    replies: int = 0
    forwards: int = 0
    saved_paths: list[Path] = field(default_factory=list)

    @property
    def average_body_length(self) -> float:
        """Average body length among saved cleaned emails."""

        if self.cleaned_emails == 0:
            return 0.0
        return self.body_length_sum / self.cleaned_emails

    @property
    def average_word_count(self) -> float:
        """Average word count among saved cleaned emails."""

        if self.cleaned_emails == 0:
            return 0.0
        return self.word_count_sum / self.cleaned_emails


class StatisticsGenerator:
    """Accumulate preprocessing statistics and render the final report."""

    def __init__(self) -> None:
        self.report = PreprocessingReport()

    def update(
        self,
        cleaned_frame: pd.DataFrame,
        parse_report: ParseReport,
        cleaning_report: CleaningReport,
    ) -> None:
        """Update aggregate statistics from one processed chunk."""

        self.report.raw_emails += parse_report.total
        self.report.successfully_parsed += parse_report.successfully_parsed
        self.report.failed_parses += parse_report.failed_parses
        self.report.malformed_emails += parse_report.malformed_emails
        self.report.malformed_header_lines += parse_report.malformed_header_lines

        self.report.cleaned_emails += cleaning_report.output_count
        self.report.duplicate_bodies_removed += cleaning_report.duplicate_bodies_removed
        self.report.duplicate_ids_removed += cleaning_report.duplicate_ids_removed
        self.report.duplicate_files_removed += cleaning_report.duplicate_files_removed
        self.report.empty_bodies_removed += cleaning_report.empty_bodies_removed
        self.report.short_bodies_removed += cleaning_report.short_bodies_removed

        if cleaned_frame.empty:
            return

        self.report.body_length_sum += int(cleaned_frame["body_length"].sum())
        self.report.word_count_sum += int(cleaned_frame["word_count"].sum())
        self.report.missing_subjects += int(
            (cleaned_frame["subject"].fillna("").str.strip() == "").sum()
        )
        self.report.missing_senders += int(
            (cleaned_frame["sender"].fillna("").str.strip() == "").sum()
        )
        self.report.missing_receivers += int(
            (cleaned_frame["receiver"].fillna("").str.strip() == "").sum()
        )
        self.report.replies += int(cleaned_frame["is_reply"].sum())
        self.report.forwards += int(cleaned_frame["is_forward"].sum())

    def set_saved_paths(self, paths: list[Path]) -> None:
        """Set saved output paths for final report rendering."""

        self.report.saved_paths = paths

    def render(self) -> str:
        """Render a production-style preprocessing report."""

        lines = [
            "Preprocessing Complete",
            "===============================",
            "",
            f"Raw Emails                : {self._fmt(self.report.raw_emails)}",
            f"Successfully Parsed       : {self._fmt(self.report.successfully_parsed)}",
            f"Failed                    : {self._fmt(self.report.failed_parses)}",
            "",
            f"Malformed Emails          : {self._fmt(self.report.malformed_emails)}",
            f"Malformed Header Lines    : {self._fmt(self.report.malformed_header_lines)}",
            "",
            f"Duplicate Bodies Removed  : {self._fmt(self.report.duplicate_bodies_removed)}",
            f"Duplicate IDs Removed     : {self._fmt(self.report.duplicate_ids_removed)}",
            f"Duplicate Files Removed   : {self._fmt(self.report.duplicate_files_removed)}",
            f"Empty Bodies Removed      : {self._fmt(self.report.empty_bodies_removed)}",
            f"Short Bodies Removed      : {self._fmt(self.report.short_bodies_removed)}",
            "",
            f"Missing Subject           : {self._fmt(self.report.missing_subjects)}",
            f"Missing Sender            : {self._fmt(self.report.missing_senders)}",
            f"Missing Receiver          : {self._fmt(self.report.missing_receivers)}",
            f"Replies                   : {self._fmt(self.report.replies)}",
            f"Forwards                  : {self._fmt(self.report.forwards)}",
            "",
            f"Average Body Length       : {self._fmt(round(self.report.average_body_length))} chars",
            f"Average Words             : {self._fmt(round(self.report.average_word_count))}",
            "",
            "Saved",
            "",
        ]
        lines.extend(str(path) for path in self.report.saved_paths)
        return "\n".join(lines)

    def print(self) -> None:
        """Print the final preprocessing report."""

        print(self.render())

    def write(self, path: Path) -> None:
        """Write the final preprocessing report to disk."""

        path.write_text(self.render() + "\n", encoding="utf-8")
        logger.info("Wrote preprocessing report: %s", path)

    @staticmethod
    def _fmt(value: int) -> str:
        """Format integer counters with thousands separators."""

        return f"{value:,}"

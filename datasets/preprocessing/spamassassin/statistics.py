"""Streaming statistics and reporting for SpamAssassin preprocessing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .cleaner import CleaningReport
from .parser import ParseReport


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SpamAssassinReport:
    """Aggregated SpamAssassin preprocessing metrics."""

    total_emails: int = 0
    successfully_parsed: int = 0
    failed_parses: int = 0
    recovered_emails: int = 0
    malformed_emails: int = 0
    malformed_header_lines: int = 0
    saved_emails: int = 0
    spam: int = 0
    ham: int = 0
    body_length_sum: int = 0
    word_count_sum: int = 0
    missing_subjects: int = 0
    missing_senders: int = 0
    missing_receivers: int = 0
    headers_extracted: int = 0
    saved_paths: list[Path] = field(default_factory=list)

    @property
    def average_body_length(self) -> float:
        """Return average saved body length, or zero for no saved records."""

        return self.body_length_sum / self.saved_emails if self.saved_emails else 0.0

    @property
    def average_word_count(self) -> float:
        """Return average saved word count, or zero for no saved records."""

        return self.word_count_sum / self.saved_emails if self.saved_emails else 0.0


class SpamAssassinStatistics:
    """Accumulate chunk-level metrics and render the final report."""

    def __init__(self) -> None:
        self.report = SpamAssassinReport()

    def update(
        self,
        cleaned_frame: pd.DataFrame,
        parse_report: ParseReport,
        cleaning_report: CleaningReport,
    ) -> None:
        """Add parser, cleaner, and output metrics from one processed chunk."""

        report = self.report
        report.total_emails += parse_report.total
        report.successfully_parsed += parse_report.successfully_parsed
        report.failed_parses += parse_report.failed_parses
        report.recovered_emails += parse_report.recovered_emails
        report.malformed_emails += parse_report.malformed_emails
        report.malformed_header_lines += parse_report.malformed_header_lines
        report.saved_emails += cleaning_report.output_count

        if cleaned_frame.empty:
            return

        report.spam += int(cleaned_frame["label"].eq(1).sum())
        report.ham += int(cleaned_frame["label"].eq(0).sum())
        report.body_length_sum += int(cleaned_frame["body_length"].sum())
        report.word_count_sum += int(cleaned_frame["word_count"].sum())
        report.missing_subjects += int(cleaned_frame["subject"].eq("").sum())
        report.missing_senders += int(cleaned_frame["sender"].eq("").sum())
        report.missing_receivers += int(cleaned_frame["receiver"].eq("").sum())
        hdr_mask = (
            cleaned_frame["message_id"].ne("")
            | cleaned_frame["date"].ne("")
            | cleaned_frame["receiver"].ne("")
            | cleaned_frame["cc"].ne("")
            | cleaned_frame["bcc"].ne("")
        )
        report.headers_extracted += int(hdr_mask.sum())

    def set_saved_paths(self, paths: list[Path]) -> None:
        """Set the paths displayed in the final report."""

        self.report.saved_paths = paths

    def render(self) -> str:
        """Render the human-readable preprocessing report."""

        report = self.report
        lines = [
            "SpamAssassin Preprocessing Complete",
            "===============================",
            "",
            f"Emails Processed          : {self._fmt(report.total_emails)}",
            f"Successfully Parsed       : {self._fmt(report.successfully_parsed)}",
            f"Failed / Skipped          : {self._fmt(report.failed_parses)}",
            f"Headers Extracted         : {self._fmt(report.headers_extracted)}",
            "",
            f"Spam                      : {self._fmt(report.spam)}",
            f"Ham                       : {self._fmt(report.ham)}",
            f"Emails Missing Subject    : {self._fmt(report.missing_subjects)}",
            f"Emails Missing Sender     : {self._fmt(report.missing_senders)}",
            f"Emails Missing Receiver   : {self._fmt(report.missing_receivers)}",
            "",
            f"Average Body Length       : {self._fmt(round(report.average_body_length))} chars",
            f"Average Words             : {self._fmt(round(report.average_word_count))}",
            "",
            "Saved",
            "",
        ]
        lines.extend(str(path) for path in report.saved_paths)
        return "\n".join(lines)

    def print(self) -> None:
        """Print the final report to stdout."""

        print(self.render())

    def write(self, path: Path) -> None:
        """Write the final report to a UTF-8 text file."""

        path.write_text(self.render() + "\n", encoding="utf-8")
        logger.info("Wrote SpamAssassin preprocessing report: %s", path)

    @staticmethod
    def _fmt(value: int) -> str:
        """Format integer metrics with thousands separators."""

        return f"{value:,}"

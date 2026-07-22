"""Configuration for the SpamAssassin preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SpamAssassinConfig:
    """Runtime configuration for SpamAssassin preprocessing.

    Attributes:
        project_root: Root directory of the MailMind repository.
        raw_dir: Directory containing raw datasets.
        processed_dir: Directory where processed outputs are written.
        raw_filename: Name of the SpamAssassin source CSV.
        parquet_filename: Name of the primary Parquet output.
        csv_filename: Name of the secondary CSV output.
        report_filename: Name of the text report output.
        failed_emails_filename: Name of the failed-emails debugging CSV.
        chunk_size: Number of source rows processed at once.
        malformed_warning_limit: Maximum malformed-record warnings to emit.
        log_level: Logging level name.
    """

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    raw_filename: str = "spam_assassin.csv"
    parquet_filename: str = "spam_processed.parquet"
    csv_filename: str = "spam_processed.csv"
    report_filename: str = "spam_preprocessing_report.txt"
    failed_emails_filename: str = "failed_emails.csv"
    chunk_size: int = 10_000
    malformed_warning_limit: int = 100
    log_level: str = "INFO"

    @classmethod
    def default(cls) -> "SpamAssassinConfig":
        """Create a configuration using paths relative to this repository."""

        project_root = Path(__file__).resolve().parents[3]
        datasets_dir = project_root / "datasets"
        return cls(
            project_root=project_root,
            raw_dir=datasets_dir / "raw",
            processed_dir=datasets_dir / "processed",
        )

    @property
    def raw_path(self) -> Path:
        """Return the raw SpamAssassin CSV path."""

        return self.raw_dir / self.raw_filename

    @property
    def parquet_path(self) -> Path:
        """Return the processed Parquet output path."""

        return self.processed_dir / self.parquet_filename

    @property
    def csv_path(self) -> Path:
        """Return the processed CSV output path."""

        return self.processed_dir / self.csv_filename

    @property
    def report_path(self) -> Path:
        """Return the preprocessing report path."""

        return self.processed_dir / self.report_filename

    @property
    def failed_emails_path(self) -> Path:
        """Return the failed-emails debugging CSV path."""

        return self.processed_dir / self.failed_emails_filename

    def ensure_directories(self) -> None:
        """Create required directories when they do not already exist."""

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

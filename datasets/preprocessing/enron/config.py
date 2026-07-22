"""Configuration for the Enron email preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    """Runtime configuration for preprocessing raw Enron emails.

    Attributes:
        project_root: Root directory of the MailMind repository.
        raw_dir: Directory containing raw datasets.
        processed_dir: Directory where processed outputs are written.
        raw_emails_filename: Name of the raw Enron CSV file.
        parsed_parquet_filename: Name of the primary parsed Parquet output.
        parsed_csv_filename: Name of the secondary parsed CSV output.
        report_filename: Name of the human-readable preprocessing report.
        sample_sizes: Sample sizes to generate from the cleaned dataset.
        random_state: Random seed used for reproducible sampling.
        chunk_size: Number of raw CSV rows to parse per chunk.
        min_body_chars: Minimum cleaned body length to retain.
        malformed_warning_limit: Maximum malformed-email examples logged at WARNING.
        log_level: Logging level name.
    """

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    raw_emails_filename: str = "emails.csv"
    parsed_parquet_filename: str = "parsed_emails.parquet"
    parsed_csv_filename: str = "parsed_emails.csv"
    report_filename: str = "preprocessing_report.txt"
    sample_sizes: tuple[int, ...] = field(default_factory=lambda: (5_000, 20_000, 50_000))
    random_state: int = 42
    chunk_size: int = 25_000
    min_body_chars: int = 30
    malformed_warning_limit: int = 100
    log_level: str = "INFO"

    @classmethod
    def default(cls) -> "PreprocessingConfig":
        """Create a config using paths relative to this repository."""

        project_root = Path(__file__).resolve().parents[2]
        datasets_dir = project_root / "datasets"
        return cls(
            project_root=project_root,
            raw_dir=datasets_dir / "raw",
            processed_dir=datasets_dir / "processed",
        )

    @property
    def raw_emails_path(self) -> Path:
        """Path to the raw Enron emails CSV file."""

        return self.raw_dir / self.raw_emails_filename

    @property
    def parsed_parquet_path(self) -> Path:
        """Path to the parsed Parquet output."""

        return self.processed_dir / self.parsed_parquet_filename

    @property
    def parsed_csv_path(self) -> Path:
        """Path to the parsed CSV output."""

        return self.processed_dir / self.parsed_csv_filename

    @property
    def report_path(self) -> Path:
        """Path to the preprocessing report output."""

        return self.processed_dir / self.report_filename

    def sample_path(self, size: int) -> Path:
        """Return the output path for a sampled Parquet dataset.

        Args:
            size: Requested sample size.

        Returns:
            Path for the sample file.
        """

        label = f"{size // 1_000}k" if size % 1_000 == 0 else str(size)
        return self.processed_dir / f"sample_{label}.parquet"

    def ensure_directories(self) -> None:
        """Create required output directories if they do not exist."""

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

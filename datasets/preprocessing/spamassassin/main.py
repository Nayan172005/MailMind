"""Run the complete SpamAssassin preprocessing pipeline for MailMind."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from .cleaner import PUBLIC_COLUMNS, SpamAssassinCleaner
    from .config import SpamAssassinConfig
    from .parser import FailedEmailRecord, SpamAssassinEmailParser
    from .statistics import SpamAssassinStatistics
    from .utils import read_spamassassin_csv, require_pyarrow, setup_logging
except ImportError:  # pragma: no cover - supports direct script execution.
    from cleaner import PUBLIC_COLUMNS, SpamAssassinCleaner
    from config import SpamAssassinConfig
    from parser import FailedEmailRecord, SpamAssassinEmailParser
    from statistics import SpamAssassinStatistics
    from utils import read_spamassassin_csv, require_pyarrow, setup_logging


logger = logging.getLogger(__name__)


class ProcessedOutputWriter:
    """Incrementally write classifier-ready records to Parquet and CSV."""

    def __init__(self, config: SpamAssassinConfig) -> None:
        self.config = config
        self._parquet_writer: Any | None = None
        self._csv_header_written = False
        self._rows_written = 0

    @property
    def rows_written(self) -> int:
        """Return the number of saved output records."""

        return self._rows_written

    def prepare(self) -> None:
        """Remove prior pipeline outputs before starting a new run."""

        for path in (self.config.parquet_path, self.config.csv_path):
            if path.exists():
                path.unlink()

    def write_chunk(self, frame: pd.DataFrame) -> None:
        """Append one cleaned chunk to both output formats."""

        if frame.empty:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(self.config.parquet_path, table.schema)
        self._parquet_writer.write_table(table)

        frame.to_csv(
            self.config.csv_path,
            mode="a",
            header=not self._csv_header_written,
            index=False,
        )
        self._csv_header_written = True
        self._rows_written += len(frame)

    def close(self) -> None:
        """Close writers and create valid empty outputs when nothing was saved."""

        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None

        if self._rows_written == 0:
            empty = pd.DataFrame(columns=PUBLIC_COLUMNS)
            empty.to_parquet(self.config.parquet_path, index=False, engine="pyarrow")
            empty.to_csv(self.config.csv_path, index=False)


class SpamAssassinPreprocessingPipeline:
    """End-to-end bounded-memory pipeline for SpamAssassin emails."""

    def __init__(self, config: SpamAssassinConfig) -> None:
        self.config = config
        self.parser = SpamAssassinEmailParser(config.malformed_warning_limit)
        self.cleaner = SpamAssassinCleaner()
        self.statistics = SpamAssassinStatistics()
        self.writer = ProcessedOutputWriter(config)

    def run(self) -> None:
        """Execute parsing, cleaning, output writing, and report generation."""

        require_pyarrow()
        self.config.ensure_directories()
        self.writer.prepare()

        all_failed: list[FailedEmailRecord] = []

        logger.info("Reading SpamAssassin data from %s", self.config.raw_path)
        chunks_seen = 0
        try:
            chunks = read_spamassassin_csv(self.config.raw_path, self.config.chunk_size)
            for chunk_number, chunk in enumerate(
                tqdm(chunks, desc="Reading SpamAssassin CSV chunks"),
                start=1,
            ):
                chunks_seen += 1
                parsed_chunk, parse_report, failed_records = self.parser.parse_dataframe(
                    chunk, chunk_number,
                )
                all_failed.extend(failed_records)
                cleaned_chunk, cleaning_report = self.cleaner.clean_chunk(parsed_chunk)
                self.writer.write_chunk(cleaned_chunk)
                self.statistics.update(cleaned_chunk, parse_report, cleaning_report)
                logger.info(
                    "SpamAssassin chunk %s complete: raw=%s saved=%s total_saved=%s",
                    chunk_number,
                    parse_report.total,
                    cleaning_report.output_count,
                    self.writer.rows_written,
                )
        finally:
            self.writer.close()

        if chunks_seen == 0:
            raise RuntimeError("No emails were read from the raw SpamAssassin dataset.")

        # Write failed_emails.csv for debugging.
        self._write_failed_emails(all_failed)

        self.statistics.set_saved_paths(
            [
                self._display_path(self.config.parquet_path),
                self._display_path(self.config.csv_path),
                self._display_path(self.config.failed_emails_path),
            ]
        )
        self.statistics.print()
        self.statistics.write(self.config.report_path)

    def _write_failed_emails(self, failed: list[FailedEmailRecord]) -> None:
        """Write the failed-emails debugging CSV."""

        df = pd.DataFrame(
            [
                {
                    "original_row": record.original_row,
                    "reason": record.reason,
                    "preview": record.preview,
                }
                for record in failed
            ],
            columns=["original_row", "reason", "preview"],
        )
        df.to_csv(self.config.failed_emails_path, index=False, encoding="utf-8")
        logger.info(
            "Wrote %s failed email record(s) to %s",
            len(df),
            self.config.failed_emails_path,
        )

    def _display_path(self, path: Path) -> Path:
        """Return a project-relative path when one can be calculated."""

        try:
            return path.relative_to(self.config.project_root)
        except ValueError:
            return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Preprocess SpamAssassin emails for MailMind spam classification."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        help="Optional raw CSV path. Defaults to datasets/raw/spam_assassin.csv.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="Optional output directory. Defaults to datasets/processed.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Number of CSV rows to process in a chunk.",
    )
    parser.add_argument(
        "--malformed-warning-limit",
        type=int,
        help="Maximum malformed email warnings emitted during processing.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level, such as INFO or DEBUG.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SpamAssassinConfig:
    """Create a configuration object from parsed CLI arguments."""

    config = SpamAssassinConfig.default()
    if args.raw_path is not None:
        raw_path = args.raw_path.resolve()
        config = replace(config, raw_dir=raw_path.parent, raw_filename=raw_path.name)
    if args.processed_dir is not None:
        config = replace(config, processed_dir=args.processed_dir.resolve())
    if args.chunk_size is not None:
        if args.chunk_size <= 0:
            raise ValueError("--chunk-size must be a positive integer.")
        config = replace(config, chunk_size=args.chunk_size)
    if args.malformed_warning_limit is not None:
        config = replace(config, malformed_warning_limit=args.malformed_warning_limit)
    if args.log_level is not None:
        config = replace(config, log_level=args.log_level)
    return config


def main() -> None:
    """Run the SpamAssassin preprocessing CLI."""

    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    setup_logging(config.log_level)
    try:
        SpamAssassinPreprocessingPipeline(config).run()
    except Exception:
        logger.exception("SpamAssassin preprocessing failed")
        raise


if __name__ == "__main__":
    main()

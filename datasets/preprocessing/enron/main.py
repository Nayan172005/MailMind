"""Run the complete Enron preprocessing pipeline for MailMind."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from .cleaner import EmailCleaner, PUBLIC_COLUMNS
    from .config import PreprocessingConfig
    from .parser import EnronEmailParser
    from .sampler import DatasetSampler
    from .statistics import StatisticsGenerator
    from .utils import read_email_csv, require_pyarrow, setup_logging
except ImportError:  # pragma: no cover - supports direct script execution.
    from cleaner import EmailCleaner, PUBLIC_COLUMNS
    from config import PreprocessingConfig
    from parser import EnronEmailParser
    from sampler import DatasetSampler
    from statistics import StatisticsGenerator
    from utils import read_email_csv, require_pyarrow, setup_logging


logger = logging.getLogger(__name__)


class ProcessedOutputWriter:
    """Incrementally write cleaned records to Parquet and CSV."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._parquet_writer: object | None = None
        self._csv_header_written = False
        self._rows_written = 0

    @property
    def rows_written(self) -> int:
        """Number of cleaned rows written."""

        return self._rows_written

    def prepare(self) -> None:
        """Prepare output files for a fresh pipeline run."""

        for path in [self.config.parsed_parquet_path, self.config.parsed_csv_path]:
            if path.exists():
                path.unlink()

    def write_chunk(self, frame: pd.DataFrame) -> None:
        """Write one cleaned chunk to Parquet and CSV."""

        if frame.empty:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(
                self.config.parsed_parquet_path,
                table.schema,
            )
        self._parquet_writer.write_table(table)

        frame.to_csv(
            self.config.parsed_csv_path,
            mode="a",
            header=not self._csv_header_written,
            index=False,
        )
        self._csv_header_written = True
        self._rows_written += len(frame)

    def close(self) -> None:
        """Close writers and ensure empty outputs exist if no rows were saved."""

        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None

        if self._rows_written == 0:
            empty = pd.DataFrame(columns=PUBLIC_COLUMNS)
            empty.to_parquet(self.config.parsed_parquet_path, index=False, engine="pyarrow")
            empty.to_csv(self.config.parsed_csv_path, index=False)


class PreprocessingPipeline:
    """End-to-end streaming preprocessing pipeline for the Enron dataset."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.parser = EnronEmailParser(warning_limit=config.malformed_warning_limit)
        self.cleaner = EmailCleaner(config)
        self.sampler = DatasetSampler(config)
        self.statistics = StatisticsGenerator()
        self.writer = ProcessedOutputWriter(config)

    def run(self) -> None:
        """Execute parsing, cleaning, saving, statistics, sampling, and reporting."""

        require_pyarrow()
        self.config.ensure_directories()
        self.writer.prepare()

        logger.info("Reading raw emails from %s", self.config.raw_emails_path)
        chunks_seen = 0

        try:
            for chunk_number, chunk in enumerate(
                tqdm(
                    read_email_csv(self.config.raw_emails_path, self.config.chunk_size),
                    desc="Reading CSV chunks",
                ),
                start=1,
            ):
                chunks_seen += 1
                parsed_chunk, parse_report = self.parser.parse_dataframe(chunk, chunk_number)
                cleaned_chunk, cleaning_report = self.cleaner.clean_chunk(parsed_chunk)

                self.writer.write_chunk(cleaned_chunk)
                self.sampler.update(cleaned_chunk)
                self.statistics.update(cleaned_chunk, parse_report, cleaning_report)

                logger.info(
                    "Chunk %s complete: raw=%s cleaned=%s total_saved=%s",
                    chunk_number,
                    parse_report.total,
                    cleaning_report.output_count,
                    self.writer.rows_written,
                )
        finally:
            self.writer.close()

        if chunks_seen == 0:
            raise RuntimeError("No emails were read from the raw dataset.")

        sample_rows = self.sampler.write_samples()
        saved_paths = [
            self._display_path(self.config.parsed_parquet_path),
            self._display_path(self.config.parsed_csv_path),
        ]
        saved_paths.extend(self._display_path(self.config.sample_path(size)) for size in sample_rows)

        self.statistics.set_saved_paths(saved_paths)
        self.statistics.print()
        self.statistics.write(self.config.report_path)

    def _display_path(self, path: Path) -> Path:
        """Return a project-relative display path when possible."""

        try:
            return path.relative_to(self.config.project_root)
        except ValueError:
            return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Preprocess the Enron emails.csv dataset for MailMind."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        help="Optional path to raw emails.csv. Defaults to datasets/raw/emails.csv.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="Optional output directory. Defaults to datasets/processed.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Number of raw CSV rows to parse per chunk.",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        help="Sample sizes to generate, such as: 5000 20000 50000.",
    )
    parser.add_argument(
        "--min-body-chars",
        type=int,
        help="Minimum cleaned body length to keep.",
    )
    parser.add_argument(
        "--malformed-warning-limit",
        type=int,
        help="Maximum malformed-email examples logged at WARNING.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level, such as INFO or DEBUG.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PreprocessingConfig:
    """Create a config object from command-line arguments."""

    config = PreprocessingConfig.default()

    if args.raw_path is not None:
        raw_path = args.raw_path.resolve()
        config = replace(
            config,
            raw_dir=raw_path.parent,
            raw_emails_filename=raw_path.name,
        )
    if args.processed_dir is not None:
        config = replace(config, processed_dir=args.processed_dir.resolve())
    if args.chunk_size is not None:
        config = replace(config, chunk_size=args.chunk_size)
    if args.sample_sizes is not None:
        config = replace(config, sample_sizes=tuple(args.sample_sizes))
    if args.min_body_chars is not None:
        config = replace(config, min_body_chars=args.min_body_chars)
    if args.malformed_warning_limit is not None:
        config = replace(config, malformed_warning_limit=args.malformed_warning_limit)
    if args.log_level is not None:
        config = replace(config, log_level=args.log_level)

    return config


def main() -> None:
    """CLI entry point."""

    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    setup_logging(config.log_level)

    try:
        PreprocessingPipeline(config).run()
    except Exception:
        logger.exception("Preprocessing failed")
        raise


if __name__ == "__main__":
    main()

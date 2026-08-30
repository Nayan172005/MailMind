"""
MailMind RAG — Email Loader

Loads emails from parquet, JSON, or CSV files.
Returns raw records as list of dicts preserving the original schema.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from rag.config import settings
from rag.utils import get_logger

logger = get_logger("loader")


def load_emails(path: Optional[str] = None) -> list[dict]:
    """
    Load emails from a dataset file.

    Supports: .parquet, .json, .csv
    Returns a list of dicts, one per email row.
    """
    filepath = Path(path) if path else settings.dataset_abs_path

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            f"Expected the processed email dataset at this location."
        )

    suffix = filepath.suffix.lower()
    logger.info(f"Loading emails from {filepath.name} ({suffix})")

    if suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif suffix == ".json":
        df = pd.read_json(filepath)
    elif suffix == ".csv":
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Convert list/array columns properly
    records = df.to_dict(orient="records")

    logger.info(f"Loaded {len(records)} emails with columns: {list(df.columns)}")
    return records

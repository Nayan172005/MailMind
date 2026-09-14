"""
MailMind RAG — Shared Utilities

Text processing helpers, token counting, and logging setup.
"""

import logging
import os
import re


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(f"mailmind.{name}")
    level_name = os.getenv("MAILMIND_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def count_words(text: str) -> int:
    """Approximate word count."""
    return len(text.split())


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text for display / logging, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def clean_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces/newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def safe_str(value) -> str:
    """Convert a value to string safely, handling None and NaN."""
    if value is None:
        return ""
    if isinstance(value, float):
        import math
        if math.isnan(value):
            return ""
    return str(value).strip()

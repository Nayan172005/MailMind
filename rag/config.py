"""
MailMind RAG — Centralized Configuration

All configurable parameters are loaded from environment variables / .env file
with sensible defaults for a student project setup.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings
from dotenv import load_dotenv


# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """RAG pipeline configuration."""

    # --- Project paths ---
    PROJECT_ROOT: Path = _PROJECT_ROOT
    DATASET_PATH: str = "datasets/processed/emails/emails_cleaned.parquet"
    INDEX_DIR: str = "data/rag/index"

    # --- Embedding ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 output dim

    # --- LLM ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # --- Retrieval ---
    TOP_K: int = 5
    INITIAL_RETRIEVAL_K: int = 20
    FINAL_RETRIEVAL_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.25
    MIN_RERANK_SCORE: float = 0.30
    RERANK_SEMANTIC_WEIGHT: float = 0.65
    RERANK_SUBJECT_WEIGHT: float = 0.15
    RERANK_BODY_WEIGHT: float = 0.05
    RERANK_SENDER_WEIGHT: float = 0.05
    RERANK_ENTITY_WEIGHT: float = 0.07
    RERANK_TEMPORAL_WEIGHT: float = 0.08

    # --- Chunking ---
    CHUNK_SIZE: int = 400   # approximate words per chunk
    CHUNK_OVERLAP: int = 50  # overlap in words

    # --- Conversation ---
    MAX_CONVERSATION_TURNS: int = 5

    @property
    def dataset_abs_path(self) -> Path:
        """Absolute path to the email dataset."""
        p = Path(self.DATASET_PATH)
        if p.is_absolute():
            return p
        return self.PROJECT_ROOT / p

    @property
    def index_abs_dir(self) -> Path:
        """Absolute path to the FAISS index directory."""
        p = Path(self.INDEX_DIR)
        if p.is_absolute():
            return p
        return self.PROJECT_ROOT / p

    @property
    def faiss_index_path(self) -> Path:
        return self.index_abs_dir / "emails.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_abs_dir / "metadata.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings instance
settings = Settings()

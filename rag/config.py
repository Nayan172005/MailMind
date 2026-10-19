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
    INITIAL_RETRIEVAL_K: int = 100
    FINAL_RETRIEVAL_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.25
    MIN_RERANK_SCORE: float = 0.30
    ENABLE_HYBRID_RETRIEVAL: bool = True
    HYBRID_DENSE_WEIGHT: float = 1.0
    HYBRID_LEXICAL_WEIGHT: float = 0.8
    RERANK_SEMANTIC_WEIGHT: float = 0.45
    RERANK_SUBJECT_WEIGHT: float = 0.20
    RERANK_BODY_WEIGHT: float = 0.10
    RERANK_SENDER_WEIGHT: float = 0.15
    RERANK_ENTITY_WEIGHT: float = 0.10
    RERANK_TEMPORAL_WEIGHT: float = 0.30
    RERANK_FRESHNESS_WEIGHT: float = 0.10

    # --- Chunking ---
    CHUNK_SIZE: int = 400   # approximate words per chunk
    CHUNK_OVERLAP: int = 50  # overlap in words

    # --- Conversation ---
    MAX_CONVERSATION_TURNS: int = 5

    # --- Gmail Integration ---
    GMAIL_CLIENT_ID: str = ""
    GMAIL_CLIENT_SECRET: str = ""
    GMAIL_REDIRECT_URI: str = "http://localhost:8000/gmail/oauth/callback"
    GMAIL_PUBSUB_TOPIC: str = ""  # e.g. projects/<project>/topics/<topic>
    GMAIL_PUBSUB_PROJECT: str = ""
    GMAIL_PUBSUB_AUDIENCE: str = ""  # Expected OIDC audience for Pub/Sub push
    GMAIL_PUBSUB_SERVICE_ACCOUNT: str = ""  # Expected Pub/Sub service-account email
    GMAIL_PUBSUB_VERIFY_TOKEN: bool = True  # Enforce OIDC token verification
    GMAIL_SCOPES: str = "https://www.googleapis.com/auth/gmail.readonly"
    GMAIL_INITIAL_SYNC_DAYS: int = 7
    GMAIL_INITIAL_SYNC_MAX: int = 50
    GMAIL_WATCH_EXPIRATION_BUFFER: int = 3600  # seconds before expiry to renew
    GMAIL_TOKEN_PATH: str = "data/gmail/token.json"
    GMAIL_STATE_PATH: str = "data/gmail/state.json"
    GMAIL_POLL_INTERVAL: int = 60  # polling fallback interval in seconds
    GMAIL_ENABLED: bool = False  # set True when credentials are configured

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

    @property
    def gmail_token_abs_path(self) -> Path:
        p = Path(self.GMAIL_TOKEN_PATH)
        if p.is_absolute():
            return p
        return self.PROJECT_ROOT / p

    @property
    def gmail_state_abs_path(self) -> Path:
        p = Path(self.GMAIL_STATE_PATH)
        if p.is_absolute():
            return p
        return self.PROJECT_ROOT / p

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings instance
settings = Settings()

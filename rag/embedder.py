"""
MailMind RAG — Embedding Module

Uses sentence-transformers for local embedding generation.
Embeds both document chunks (for indexing) and queries (for retrieval).
The model runs entirely locally — no API calls needed.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import settings
from rag.utils import get_logger

logger = get_logger("embedder")

# Module-level model cache
_model: SentenceTransformer | None = None


def get_model(model_name: str | None = None) -> SentenceTransformer:
    """Load or return cached sentence-transformer model."""
    global _model
    model_name = model_name or settings.EMBEDDING_MODEL

    if _model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info(
            f"Model loaded. Embedding dimension: "
            f"{_model.get_sentence_embedding_dimension()}"
        )

    return _model


def embed_texts(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of strings to embed.
        model_name: Override the default embedding model.
        batch_size: Batch size for encoding.
        show_progress: Show progress bar during encoding.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    model = get_model(model_name)
    logger.info(f"Embedding {len(texts)} texts...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def embed_query(
    query: str,
    model_name: str | None = None,
) -> np.ndarray:
    """
    Generate embedding for a single query string.

    Returns:
        numpy array of shape (1, embedding_dim).
    """
    model = get_model(model_name)
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding

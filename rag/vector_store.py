"""
MailMind RAG — FAISS Vector Store

Persistent FAISS index with a JSON metadata store.
Supports building, saving, loading, updating, and searching the index.

Index layout on disk:
    data/rag/index/
    ├── emails.faiss       # FAISS flat index (Inner Product for normalized vectors)
    └── metadata.json      # vector_id -> chunk metadata mapping
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from rag.config import settings
from rag.schemas import DocumentChunk
from rag.embedder import embed_texts
from rag.utils import get_logger

logger = get_logger("vector_store")


class VectorStore:
    """FAISS-backed vector store with metadata."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []  # list indexed by FAISS vector ID
        self._indexed_message_ids: set[str] = set()

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and len(self.metadata) > 0

    @property
    def total_vectors(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    @property
    def total_emails(self) -> int:
        return len(self._indexed_message_ids)

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """
        Build a FAISS index from document chunks.

        Embeds all chunk texts, creates a flat inner-product index
        (works as cosine similarity since embeddings are L2-normalized),
        and stores chunk metadata for retrieval.
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        logger.info(f"Building index from {len(chunks)} chunks...")

        # Embed all chunk texts
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(texts)

        # Create FAISS index (Inner Product = cosine sim for normalized vectors)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        self.metadata = []
        self._indexed_message_ids = set()

        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "vector_id": i,
                "message_id": chunk.message_id,
                "chunk_id": chunk.chunk_id,
                "subject": chunk.subject,
                "sender_name": chunk.sender_name,
                "sender_email": chunk.sender_email,
                "date": chunk.date,
                "text": chunk.text,
                "body_snippet": chunk.body_snippet,
            })
            self._indexed_message_ids.add(chunk.message_id)

        logger.info(
            f"Index built: {self.index.ntotal} vectors, "
            f"{len(self._indexed_message_ids)} unique emails"
        )

    def update_index(self, new_chunks: list[DocumentChunk]) -> int:
        """
        Add new chunks to an existing index, skipping already-indexed message_ids.

        Returns the number of new chunks added.
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded. Call build_index() or load() first.")

        # Filter out already-indexed emails
        novel_chunks = [
            c for c in new_chunks
            if c.message_id not in self._indexed_message_ids
        ]

        if not novel_chunks:
            logger.info("No new emails to index.")
            return 0

        logger.info(f"Adding {len(novel_chunks)} new chunks to index...")

        texts = [chunk.text for chunk in novel_chunks]
        embeddings = embed_texts(texts)

        base_id = len(self.metadata)
        self.index.add(embeddings.astype(np.float32))

        for i, chunk in enumerate(novel_chunks):
            self.metadata.append({
                "vector_id": base_id + i,
                "message_id": chunk.message_id,
                "chunk_id": chunk.chunk_id,
                "subject": chunk.subject,
                "sender_name": chunk.sender_name,
                "sender_email": chunk.sender_email,
                "date": chunk.date,
                "text": chunk.text,
                "body_snippet": chunk.body_snippet,
            })
            self._indexed_message_ids.add(chunk.message_id)

        logger.info(f"Index updated: {self.index.ntotal} total vectors")
        return len(novel_chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Search the index for nearest neighbors.

        Args:
            query_embedding: Shape (1, dim) numpy array.
            top_k: Number of results to return.

        Returns:
            List of dicts with metadata + 'score' field, sorted by descending score.
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded.")

        top_k = top_k or settings.TOP_K
        top_k = min(top_k, self.index.ntotal)  # can't return more than we have

        scores, indices = self.index.search(
            query_embedding.astype(np.float32), top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self, index_dir: str | Path | None = None) -> None:
        """Persist the FAISS index and metadata to disk."""
        if not self.is_loaded:
            raise RuntimeError("No index to save.")

        index_dir = Path(index_dir) if index_dir else settings.index_abs_dir
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = index_dir / "emails.faiss"
        meta_path = index_dir / "metadata.json"

        faiss.write_index(self.index, str(faiss_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        logger.info(f"Index saved to {index_dir}")

    def load(self, index_dir: str | Path | None = None) -> None:
        """Load a persisted FAISS index and metadata from disk."""
        index_dir = Path(index_dir) if index_dir else settings.index_abs_dir

        faiss_path = index_dir / "emails.faiss"
        meta_path = index_dir / "metadata.json"

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        self.index = faiss.read_index(str(faiss_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self._indexed_message_ids = {m["message_id"] for m in self.metadata}

        logger.info(
            f"Index loaded: {self.index.ntotal} vectors, "
            f"{len(self._indexed_message_ids)} unique emails"
        )

    def get_chunks_by_message_id(self, message_id: str) -> list[dict]:
        """Retrieve all chunks for a given message_id."""
        return [m for m in self.metadata if m["message_id"] == message_id]

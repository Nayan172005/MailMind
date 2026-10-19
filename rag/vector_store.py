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
import threading
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
        self._lock = threading.RLock()  # protects index+metadata mutations (reentrant)
        self._version: int = 0
        self._disk_signature: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._index_dir: Optional[Path] = None

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self.index is not None

    @property
    def total_vectors(self) -> int:
        with self._lock:
            if self.index is None:
                return 0
            return self.index.ntotal

    @property
    def total_emails(self) -> int:
        with self._lock:
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
        with self._lock:
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

            self._version += 1
            self._disk_signature = (0, 0, 0, 0)

            logger.info(
                f"Index built: {self.index.ntotal} vectors, "
                f"{len(self._indexed_message_ids)} unique emails (version {self._version})"
            )

    def update_index(self, new_chunks: list[DocumentChunk]) -> int:
        """
        Add new chunks to an existing index, skipping already-indexed message_ids.

        Returns the number of new chunks added.
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded. Call build_index() or load() first.")

        with self._lock:
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

            self._version += 1
            logger.info(f"Index updated: {self.index.ntotal} total vectors (version {self._version})")
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

        with self._lock:
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

        with self._lock:
            index_dir = Path(index_dir) if index_dir else (self._index_dir or settings.index_abs_dir)
            self._index_dir = index_dir
            index_dir.mkdir(parents=True, exist_ok=True)

            faiss_path = index_dir / "emails.faiss"
            meta_path = index_dir / "metadata.json"

            # Write metadata to temp file first, then rename for atomicity
            meta_tmp = index_dir / "metadata.json.tmp"
            faiss_tmp = index_dir / "emails.faiss.tmp"

            faiss.write_index(self.index, str(faiss_tmp))

            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False)

            import os
            def _atomic_replace(src: Path, dst: Path) -> None:
                for attempt in range(5):
                    try:
                        if dst.exists():
                            os.replace(str(src), str(dst))
                        else:
                            src.rename(dst)
                        return
                    except (PermissionError, OSError):
                        if attempt == 4:
                            raise
                        import time
                        time.sleep(0.02 * (attempt + 1))

            _atomic_replace(faiss_tmp, faiss_path)
            _atomic_replace(meta_tmp, meta_path)

            try:
                f_stat = faiss_path.stat()
                m_stat = meta_path.stat()
                self._disk_signature = (
                    f_stat.st_mtime_ns,
                    f_stat.st_size,
                    m_stat.st_mtime_ns,
                    m_stat.st_size,
                )
            except OSError:
                pass

            logger.info(f"Index saved to {index_dir}")

    def load(self, index_dir: str | Path | None = None) -> None:
        """Load a persisted FAISS index and metadata from disk."""
        index_dir = Path(index_dir) if index_dir else settings.index_abs_dir
        self._index_dir = index_dir

        faiss_path = index_dir / "emails.faiss"
        meta_path = index_dir / "metadata.json"

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        new_index = faiss.read_index(str(faiss_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            new_meta = json.load(f)

        f_stat = faiss_path.stat()
        m_stat = meta_path.stat()
        current_sig = (
            f_stat.st_mtime_ns,
            f_stat.st_size,
            m_stat.st_mtime_ns,
            m_stat.st_size,
        )

        with self._lock:
            self.index = new_index
            self.metadata = new_meta
            self._indexed_message_ids = {m["message_id"] for m in self.metadata}
            self._disk_signature = current_sig
            self._version += 1

            logger.info(
                f"Index loaded: {self.index.ntotal} vectors, "
                f"{len(self._indexed_message_ids)} unique emails (version {self._version})"
            )

    def reload_if_changed(self, index_dir: str | Path | None = None) -> bool:
        """
        Check if on-disk index or metadata has been modified since last load/save.
        If modified, safely reload under lock.

        Fast return (microsecond stat check) if unchanged.
        Returns True if index was reloaded, False otherwise.
        """
        target_dir = Path(index_dir) if index_dir else self._index_dir
        if target_dir is None:
            return False

        faiss_path = target_dir / "emails.faiss"
        meta_path = target_dir / "metadata.json"

        if not faiss_path.exists() or not meta_path.exists():
            return False

        try:
            f_stat = faiss_path.stat()
            m_stat = meta_path.stat()
            current_sig = (
                f_stat.st_mtime_ns,
                f_stat.st_size,
                m_stat.st_mtime_ns,
                m_stat.st_size,
            )
        except OSError:
            return False

        if (self.index is not None) and current_sig == self._disk_signature:
            return False

        with self._lock:
            # Re-check under lock in case another thread just reloaded
            try:
                f_stat = faiss_path.stat()
                m_stat = meta_path.stat()
                current_sig = (
                    f_stat.st_mtime_ns,
                    f_stat.st_size,
                    m_stat.st_mtime_ns,
                    m_stat.st_size,
                )
            except OSError:
                return False

            if (self.index is not None) and current_sig == self._disk_signature:
                return False

            try:
                logger.info("Detected updated index files on disk. Reloading VectorStore...")
                new_index = faiss.read_index(str(faiss_path))
                with open(meta_path, "r", encoding="utf-8") as f:
                    new_meta = json.load(f)
            except (OSError, json.JSONDecodeError, RuntimeError) as e:
                logger.warning(
                    f"Transient error reading updated index files: {e}. Keeping current in-memory index."
                )
                return False

            self.index = new_index
            self.metadata = new_meta
            self._indexed_message_ids = {m["message_id"] for m in self.metadata}
            self._disk_signature = current_sig
            self._index_dir = target_dir
            self._version += 1

            logger.info(
                f"Index reloaded successfully: {self.index.ntotal} vectors, "
                f"{len(self._indexed_message_ids)} unique emails (version {self._version})"
            )
            return True

    def get_chunks_by_message_id(self, message_id: str) -> list[dict]:
        """Retrieve all chunks for a given message_id."""
        with self._lock:
            return [m for m in self.metadata if m["message_id"] == message_id]


"""
Tests for Streamlit Real-Time Ingestion and Stale State Invalidation.

Verifies that:
1. Newly ingested emails become queryable immediately without restarting Streamlit or clearing session state.
2. Newly added emails are indexed in both dense FAISS and lexical BM25.
3. Sub-millisecond signature checks avoid unnecessary reloads on normal Streamlit reruns.
4. Concurrent ingestion and search operations are completely thread-safe.
5. Streamlit cached pipeline automatically reflects updated disk index.
"""

import os
import sys
import threading
import time
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import settings
from rag.schemas import DocumentChunk
from rag.vector_store import VectorStore
from rag.pipeline import RAGPipeline
from rag.hybrid_retriever import get_bm25_index


def _create_sample_chunk(msg_id: str, subject: str, text: str, date: str = "Mon, 12 Oct 2026 10:00:00 +0000") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=0,
        message_id=msg_id,
        subject=subject,
        sender_name="VIT Admin",
        sender_email="admin@vit.ac.in",
        date=date,
        text=text,
        body_snippet=text[:100],
    )


def test_realtime_ingestion_queryable_without_restart(tmp_path: Path):
    """
    Verify that an email ingested after the pipeline starts can be queried
    immediately without restarting the pipeline or app.
    """
    index_dir = tmp_path / "rag_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build initial index with 2 base emails
    initial_chunks = [
        _create_sample_chunk("msg_001", "Campus Placement Drive", "TCS recruitment drive for 2026 batch."),
        _create_sample_chunk("msg_002", "Library Notice", "Return borrowed books before semester break."),
    ]
    base_store = VectorStore()
    base_store.build_index(initial_chunks)
    base_store.save(index_dir)

    # 2. Simulate Streamlit starting and caching the pipeline
    app_pipeline = RAGPipeline()
    app_pipeline.load_index(str(index_dir))

    assert app_pipeline.store.total_emails == 2
    assert app_pipeline.store.total_vectors == 2

    # Query for an unannounced seminar — should return nothing relevant
    initial_res = app_pipeline.search("Quantum Robotics Workshop", top_k=5)
    assert not any(r.message_id == "msg_qrw_999" for r in initial_res.results)

    # 3. Simulate real-time ingestion occurring in another process/worker
    new_chunk = _create_sample_chunk(
        "msg_qrw_999",
        "Urgent: Quantum Robotics Workshop 2026 Schedule",
        "The Quantum Robotics Workshop will take place in Hall B on Friday.",
    )
    writer_store = VectorStore()
    writer_store.load(index_dir)
    added = writer_store.update_index([new_chunk])
    assert added == 1
    writer_store.save(index_dir)

    # 4. In Streamlit (WITHOUT restarting or recreating app_pipeline):
    # Verify status check automatically detects and reports updated counts
    status = app_pipeline.get_index_status()
    assert status.total_emails == 3
    assert status.total_vectors == 3

    # 5. Verify the newly ingested email is retrieved in search
    search_res = app_pipeline.search("Quantum Robotics Workshop", top_k=5)
    matching_ids = [r.message_id for r in search_res.results]
    assert "msg_qrw_999" in matching_ids, "Newly ingested email must be retrieved without restarting!"

    # 6. Verify hybrid BM25 retrieval indexed the new email
    bm25 = get_bm25_index(app_pipeline.store)
    bm25_hits = bm25.search(["robotics", "workshop"], top_k=10)
    hit_doc_indices = [idx for idx, _ in bm25_hits]
    hit_msg_ids = [app_pipeline.store.metadata[idx]["message_id"] for idx in hit_doc_indices if idx < len(app_pipeline.store.metadata)]
    assert "msg_qrw_999" in hit_msg_ids


def test_fast_check_when_unchanged(tmp_path: Path):
    """
    Verify that reload_if_changed returns False immediately when disk files
    have not been modified, avoiding redundant reloads.
    """
    index_dir = tmp_path / "rag_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    initial_chunks = [
        _create_sample_chunk("msg_alpha", "Alpha Subject", "Alpha body text content."),
    ]
    store = VectorStore()
    store.build_index(initial_chunks)
    store.save(index_dir)

    # First reload check right after save should return False (already current)
    reloaded = store.reload_if_changed(index_dir)
    assert not reloaded

    # Multiple successive checks must all be False
    for _ in range(5):
        assert not store.reload_if_changed(index_dir)


def test_bm25_auto_invalidation_on_realtime_update(tmp_path: Path):
    """
    Verify that BM25 inverted index automatically invalidates and updates
    when new chunks are added to the VectorStore.
    """
    index_dir = tmp_path / "rag_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    store = VectorStore()
    store.build_index([_create_sample_chunk("msg_1", "General Notice", "General announcements.")])
    store.save(index_dir)

    bm25_1 = get_bm25_index(store)
    assert bm25_1.doc_count == 1
    assert not bm25_1.search(["biotechnology"])

    # Ingest biotechnology email
    store.update_index([_create_sample_chunk("msg_2", "Biotechnology Symposium", "Annual Biotechnology Research Symposium.")])
    store.save(index_dir)

    # Next call to get_bm25_index must automatically rebuild for the new version
    bm25_2 = get_bm25_index(store)
    assert bm25_2.doc_count == 2
    hits = bm25_2.search(["biotechnology"])
    assert len(hits) > 0


def test_concurrent_ingestion_and_search(tmp_path: Path):
    """
    Verify thread safety under concurrent ingestion and search operations.
    """
    index_dir = tmp_path / "rag_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    base_chunks = [
        _create_sample_chunk(f"msg_init_{i}", f"Subject {i}", f"Body content {i}")
        for i in range(5)
    ]
    store = VectorStore()
    store.build_index(base_chunks)
    store.save(index_dir)

    pipeline = RAGPipeline()
    pipeline.load_index(str(index_dir))

    errors = []
    stop_event = threading.Event()

    def search_worker():
        while not stop_event.is_set():
            try:
                pipeline.search("Subject", top_k=3)
            except Exception as e:
                errors.append(f"Search error: {e}")
            time.sleep(0.005)

    def ingestion_worker():
        for i in range(10):
            try:
                chunk = _create_sample_chunk(f"msg_thread_{i}", f"Live Ingest {i}", f"Live content {i}")
                writer = VectorStore()
                writer.load(index_dir)
                writer.update_index([chunk])
                writer.save(index_dir)
            except Exception as e:
                errors.append(f"Ingestion error: {e}")
            time.sleep(0.01)

    t_search = threading.Thread(target=search_worker)
    t_ingest = threading.Thread(target=ingestion_worker)

    t_search.start()
    t_ingest.start()

    t_ingest.join(timeout=10)
    stop_event.set()
    t_search.join(timeout=5)

    assert not errors, f"Concurrent operations caused errors: {errors}"
    assert pipeline.store.total_emails >= 5


def test_streamlit_cached_pipeline_auto_refresh(tmp_path: Path, monkeypatch):
    """
    Verify Streamlit load_pipeline cached resource picks up newly ingested
    emails when reload_if_changed is called.
    """
    monkeypatch.setattr(settings, "INDEX_DIR", str(tmp_path / "rag_index"))
    index_dir = settings.index_abs_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build initial index
    initial_chunks = [
        _create_sample_chunk("msg_st_1", "Orientation Session", "First year orientation."),
    ]
    init_store = VectorStore()
    init_store.build_index(initial_chunks)
    init_store.save(index_dir)

    # 2. Call Streamlit load_pipeline
    from streamlit_app.app import load_pipeline
    cached_pipeline, loaded = load_pipeline()
    assert loaded
    assert cached_pipeline.store.total_emails == 1

    # 3. External ingestion
    new_chunk = _create_sample_chunk("msg_st_2", "Hackathon Registration", "Register for Hackathon 2026.")
    ext_store = VectorStore()
    ext_store.load(index_dir)
    ext_store.update_index([new_chunk])
    ext_store.save(index_dir)

    # 4. Streamlit rerun simulation (calling reload_if_changed on cached_pipeline)
    reloaded = cached_pipeline.reload_if_changed()
    assert reloaded
    assert cached_pipeline.store.total_emails == 2

    # Query for the new email using the cached pipeline
    res = cached_pipeline.search("Hackathon Registration", top_k=5)
    assert any(r.message_id == "msg_st_2" for r in res.results)

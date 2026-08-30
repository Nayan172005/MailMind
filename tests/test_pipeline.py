"""Tests for rag.pipeline (integration tests)"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline
from rag.config import settings


@pytest.fixture(scope="module")
def pipeline():
    """Build a pipeline with the real dataset."""
    if not settings.dataset_abs_path.exists():
        pytest.skip("Dataset not found")

    p = RAGPipeline()

    # Try to load existing index, or build one
    try:
        p.load_index()
    except FileNotFoundError:
        p.build_index()

    return p


def test_pipeline_is_ready(pipeline):
    """Test that the pipeline is ready after loading."""
    assert pipeline.is_ready


def test_search(pipeline):
    """Test semantic search."""
    response = pipeline.search("placement interviews")
    assert response.total > 0
    assert len(response.results) > 0


def test_search_empty(pipeline):
    """Test search with empty query."""
    response = pipeline.search("")
    assert response.total == 0


def test_index_status(pipeline):
    """Test index status."""
    status = pipeline.get_index_status()
    assert status.indexed is True
    assert status.total_vectors > 0
    assert status.total_emails > 0

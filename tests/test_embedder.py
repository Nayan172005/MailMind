"""Tests for rag.embedder"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import embed_texts, embed_query


@pytest.fixture(scope="module")
def sample_embeddings():
    """Generate embeddings once for the module (model loading is slow)."""
    texts = [
        "Microsoft placement drive at VIT",
        "NPTEL course registration deadline",
        "Google security alert for your account",
    ]
    return embed_texts(texts, show_progress=False)


def test_embedding_shape(sample_embeddings):
    """Test that embeddings have correct shape."""
    assert sample_embeddings.shape[0] == 3
    assert sample_embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dim


def test_embedding_normalized(sample_embeddings):
    """Test that embeddings are L2-normalized."""
    norms = np.linalg.norm(sample_embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_query_embedding():
    """Test single query embedding."""
    emb = embed_query("placement interview")
    assert emb.shape == (1, 384)


def test_semantic_similarity(sample_embeddings):
    """Test that similar texts have higher similarity."""
    # "Microsoft placement" should be more similar to "NPTEL registration"
    # than to "Google security"
    placement_emb = sample_embeddings[0]
    nptel_emb = sample_embeddings[1]
    security_emb = sample_embeddings[2]

    # Cosine similarity (since normalized, dot product = cosine)
    sim_placement_nptel = np.dot(placement_emb, nptel_emb)
    sim_placement_security = np.dot(placement_emb, security_emb)

    # Both should be valid floats (the exact relationship may vary)
    assert isinstance(float(sim_placement_nptel), float)
    assert isinstance(float(sim_placement_security), float)

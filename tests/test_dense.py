# tests/test_dense.py
import pytest
import numpy as np
from retrieval.dense_retriever import load_chunks, build_dense_index, dense_search

CHUNKS_PATH = "data/sample_chunks.json"


@pytest.fixture(scope="module")
def setup_dense():
    """Load chunks and build dense index once for all tests."""
    chunks = load_chunks(CHUNKS_PATH)
    model, embeddings = build_dense_index(chunks)
    return chunks, model, embeddings


def test_embeddings_shape(setup_dense):
    """Embeddings matrix has correct shape (num_chunks x 384)."""
    chunks, _, embeddings = setup_dense
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384


def test_embeddings_normalized(setup_dense):
    """All embedding vectors have length ~1.0 after normalization."""
    _, _, embeddings = setup_dense
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_dense_returns_k_results(setup_dense):
    """Dense search returns exactly k results."""
    chunks, model, embeddings = setup_dense
    results = dense_search("attention mechanism", chunks, model, embeddings, k=5)
    assert len(results) == 5


def test_dense_scores_descending(setup_dense):
    """Results are sorted by score highest to lowest."""
    chunks, model, embeddings = setup_dense
    results = dense_search("BERT language model", chunks, model, embeddings, k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_scores_range(setup_dense):
    """Cosine similarity scores are between -1 and 1."""
    chunks, model, embeddings = setup_dense
    results = dense_search("transformer model", chunks, model, embeddings, k=10)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0


def test_dense_semantic_match(setup_dense):
    """Dense finds attention paper even without the word attention in query."""
    chunks, model, embeddings = setup_dense
    # query uses different words but same meaning as attention
    results = dense_search(
        "how do models focus on relevant parts of input",
        chunks, model, embeddings, k=5
    )
    retrieved_ids = [r["chunk_id"] for r in results]
    # chunks 0 or 1 are about attention — dense should find them
    assert any(cid in retrieved_ids for cid in [0, 1])


def test_dense_score_field_exists(setup_dense):
    """Every result has a score field."""
    chunks, model, embeddings = setup_dense
    results = dense_search("language model", chunks, model, embeddings, k=3)
    for r in results:
        assert "score" in r
        assert isinstance(r["score"], float)
        
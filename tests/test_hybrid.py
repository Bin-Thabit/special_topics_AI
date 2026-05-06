# tests/test_hybrid.py
import pytest
from retrieval.bm25_retriever import load_chunks, build_bm25_index
from retrieval.dense_retriever import build_dense_index
from retrieval.hybrid_retriever import hybrid_search, normalize_scores

CHUNKS_PATH = "data/sample_chunks.json"


@pytest.fixture(scope="module")
def setup_hybrid():
    """Build all indexes once for all tests."""
    chunks = load_chunks(CHUNKS_PATH)
    bm25_index = build_bm25_index(chunks)
    dense_model, dense_embeddings = build_dense_index(chunks)
    return chunks, bm25_index, dense_model, dense_embeddings


def test_hybrid_returns_k_results(setup_hybrid):
    """Hybrid search returns exactly k results."""
    chunks, bm25_index, dense_model, dense_embeddings = setup_hybrid
    results = hybrid_search(
        "BERT masked language modeling",
        chunks, bm25_index, dense_model, dense_embeddings,
        k=5, alpha=0.5
    )
    assert len(results) == 5


def test_hybrid_scores_descending(setup_hybrid):
    """Results are sorted by hybrid score highest to lowest."""
    chunks, bm25_index, dense_model, dense_embeddings = setup_hybrid
    results = hybrid_search(
        "transformer attention",
        chunks, bm25_index, dense_model, dense_embeddings,
        k=5, alpha=0.5
    )
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_has_score_fields(setup_hybrid):
    """Every result has hybrid, bm25, and dense score fields."""
    chunks, bm25_index, dense_model, dense_embeddings = setup_hybrid
    results = hybrid_search(
        "retrieval augmented generation",
        chunks, bm25_index, dense_model, dense_embeddings,
        k=3, alpha=0.5
    )
    for r in results:
        assert "score" in r
        assert "bm25_score" in r
        assert "dense_score" in r


def test_hybrid_alpha_zero_equals_dense(setup_hybrid):
    """Alpha=0.0 should rank purely by dense scores."""
    chunks, bm25_index, dense_model, dense_embeddings = setup_hybrid
    query = "how do neural networks learn representations"

    hybrid_results = hybrid_search(
        query, chunks, bm25_index, dense_model, dense_embeddings,
        k=5, alpha=0.0
    )
    # with alpha=0, hybrid_score = dense_score
    # so ranking should be purely by dense score
    scores = [r["score"] for r in hybrid_results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_alpha_one_equals_bm25(setup_hybrid):
    """Alpha=1.0 should rank purely by BM25 scores."""
    chunks, bm25_index, dense_model, dense_embeddings = setup_hybrid
    query = "BERT masked language modeling"

    hybrid_results = hybrid_search(
        query, chunks, bm25_index, dense_model, dense_embeddings,
        k=5, alpha=1.0
    )
    scores = [r["score"] for r in hybrid_results]
    assert scores == sorted(scores, reverse=True)


def test_normalize_scores_range():
    """Normalized scores should be between 0 and 1."""
    fake_results = [
        {"chunk_id": 0, "score": 4.5},
        {"chunk_id": 1, "score": 2.1},
        {"chunk_id": 2, "score": 0.0},
    ]
    normalized = normalize_scores(fake_results)
    for r in normalized:
        assert 0.0 <= r["score"] <= 1.0


def test_normalize_scores_max_is_one():
    """After normalization the highest score should be 1.0."""
    fake_results = [
        {"chunk_id": 0, "score": 8.2},
        {"chunk_id": 1, "score": 3.1},
        {"chunk_id": 2, "score": 1.0},
    ]
    normalized = normalize_scores(fake_results)
    max_score = max(r["score"] for r in normalized)
    assert max_score == 1.0
    
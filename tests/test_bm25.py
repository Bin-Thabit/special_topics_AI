# tests/test_bm25.py
import pytest
from retrieval.bm25_retriever import load_chunks, build_bm25_index, bm25_search

CHUNKS_PATH = "data/sample_chunks.json"


@pytest.fixture
def setup_bm25():
    """Load chunks and build BM25 index once for all tests."""
    chunks = load_chunks(CHUNKS_PATH)
    index = build_bm25_index(chunks)
    return chunks, index


def test_load_chunks(setup_bm25):
    """Chunks load correctly and have required fields."""
    chunks, _ = setup_bm25
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "paper_id" in chunk
        assert "text" in chunk
        assert "page_num" in chunk


def test_bm25_returns_k_results(setup_bm25):
    """BM25 returns exactly k results."""
    chunks, index = setup_bm25
    results = bm25_search("attention mechanism", chunks, index, k=5)
    assert len(results) == 5


def test_bm25_scores_descending(setup_bm25):
    """Results are sorted by score highest to lowest."""
    chunks, index = setup_bm25
    results = bm25_search("BERT masked language modeling", chunks, index, k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_relevant_result(setup_bm25):
    """BM25 finds the correct chunk for an exact keyword query."""
    chunks, index = setup_bm25
    results = bm25_search("BERT masked language modeling", chunks, index, k=5)
    retrieved_ids = [r["chunk_id"] for r in results]
    assert 6 in retrieved_ids  # chunk 6 is about BERT masked LM


def test_bm25_score_field_exists(setup_bm25):
    """Every result has a score field."""
    chunks, index = setup_bm25
    results = bm25_search("transformer attention", chunks, index, k=3)
    for r in results:
        assert "score" in r
        assert isinstance(r["score"], float)


def test_bm25_unrelated_query(setup_bm25):
    """Unrelated query still returns k results with low scores."""
    chunks, index = setup_bm25
    results = bm25_search("xyzxyzxyz notaword", chunks, index, k=5)
    assert len(results) == 5
    assert results[0]["score"] == 0.0  # no matches → all zeros
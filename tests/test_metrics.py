# tests/test_metrics.py
import pytest
from evaluation.metrics import recall_at_k, ndcg_at_k, evaluate_retriever
from retrieval.bm25_retriever import load_chunks, build_bm25_index, bm25_search

CHUNKS_PATH = "data/sample_chunks.json"
GOLD_PATH   = "data/gold_qa.json"


# ── unit tests for recall ──────────────────────────────────────────

def test_recall_perfect():
    """Correct chunk is at position 1 → recall = 1.0."""
    assert recall_at_k([6, 5, 8, 3, 1], [6], k=5) == 1.0


def test_recall_found_last():
    """Correct chunk is at last position → still recall = 1.0."""
    assert recall_at_k([1, 2, 3, 4, 6], [6], k=5) == 1.0


def test_recall_not_found():
    """Correct chunk not in results → recall = 0.0."""
    assert recall_at_k([1, 2, 3, 4, 5], [6], k=5) == 0.0


def test_recall_multiple_relevant():
    """Any one of multiple relevant chunks found → recall = 1.0."""
    assert recall_at_k([1, 2, 3, 4, 5], [3, 7, 9], k=5) == 1.0


def test_recall_k_cutoff():
    """Correct chunk outside top-k window → recall = 0.0."""
    assert recall_at_k([1, 2, 3, 6, 5], [6], k=3) == 0.0


# ── unit tests for ndcg ───────────────────────────────────────────

def test_ndcg_perfect():
    """Correct chunk at position 1 → NDCG = 1.0."""
    assert ndcg_at_k([6, 5, 8, 3, 1], [6], k=5) == 1.0


def test_ndcg_not_found():
    """Correct chunk not found → NDCG = 0.0."""
    assert ndcg_at_k([1, 2, 3, 4, 5], [6], k=5) == 0.0


def test_ndcg_lower_when_ranked_later():
    """NDCG decreases when correct chunk is ranked lower."""
    ndcg_pos1 = ndcg_at_k([6, 1, 2, 3, 4], [6], k=5)
    ndcg_pos3 = ndcg_at_k([1, 2, 6, 3, 4], [6], k=5)
    ndcg_pos5 = ndcg_at_k([1, 2, 3, 4, 6], [6], k=5)
    assert ndcg_pos1 > ndcg_pos3 > ndcg_pos5


def test_ndcg_between_zero_and_one():
    """NDCG is always between 0 and 1."""
    score = ndcg_at_k([6, 5, 8, 3, 1], [6, 5], k=5)
    assert 0.0 <= score <= 1.0


# ── integration test ──────────────────────────────────────────────

def test_evaluate_retriever_keys():
    """evaluate_retriever returns correct keys."""
    chunks = load_chunks(CHUNKS_PATH)
    index = build_bm25_index(chunks)

    def search_fn(query, k):
        return bm25_search(query, chunks, index, k)

    metrics = evaluate_retriever(search_fn, gold_qa_path=GOLD_PATH, k=5)
    assert "recall@5" in metrics
    assert "ndcg@5" in metrics
    assert metrics["k"] == 5


def test_evaluate_retriever_range():
    """Recall and NDCG scores are between 0 and 1."""
    chunks = load_chunks(CHUNKS_PATH)
    index = build_bm25_index(chunks)

    def search_fn(query, k):
        return bm25_search(query, chunks, index, k)

    metrics = evaluate_retriever(search_fn, gold_qa_path=GOLD_PATH, k=5)
    assert 0.0 <= metrics["recall@5"] <= 1.0
    assert 0.0 <= metrics["ndcg@5"] <= 1.0
    
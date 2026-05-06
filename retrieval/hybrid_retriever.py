import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval.bm25_retriever import load_chunks, build_bm25_index, bm25_search
from retrieval.dense_retriever import build_dense_index, dense_search


def normalize_scores(results: list[dict]) -> list[dict]:
    """
    Normalizes scores to [0, 1] so BM25 and dense scores
    are on the same scale before combining.
    """
    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)

    if max_s - min_s == 0:
        return [{**r, "score": 0.0} for r in results]

    return [
        {**r, "score": (r["score"] - min_s) / (max_s - min_s)}
        for r in results
    ]


def hybrid_search(
    query: str,
    chunks: list[dict],
    bm25_index: BM25Okapi,
    dense_model: SentenceTransformer,
    dense_embeddings: np.ndarray,
    k: int = 5,
    alpha: float = 0.5,
    query_vector: np.ndarray | None = None  # ← pre-projected vector from Optuna
) -> list[dict]:
    """
    Combines BM25 and dense search into hybrid retrieval.

    Formula:
      hybrid_score = alpha × bm25_score + (1 - alpha) × dense_score

    query_vector: if provided, passed straight to dense_search so SVD
                  projection and normalization aren't re-applied.
    """
    # --- Step 1: BM25 results for ALL chunks ---
    bm25_results = bm25_search(query, chunks, bm25_index, k=len(chunks))

    # --- Step 2: Dense results for ALL chunks ---
    # Pass query_vector through — dense_search will use it directly if set
    dense_results = dense_search(
        query, chunks, dense_model, dense_embeddings,
        k=len(chunks),
        query_vector=query_vector
    )

    # --- Step 3: Normalize both score lists to [0, 1] ---
    bm25_norm  = normalize_scores(bm25_results)
    dense_norm = normalize_scores(dense_results)

    # --- Step 4: Build lookup dicts by chunk_id ---
    bm25_by_id  = {r["chunk_id"]: r["score"] for r in bm25_norm}
    dense_by_id = {r["chunk_id"]: r["score"] for r in dense_norm}

    # --- Step 5: Combine scores for every chunk ---
    hybrid_results = []
    for chunk in chunks:
        cid     = chunk["chunk_id"]
        b_score = bm25_by_id.get(cid, 0.0)
        d_score = dense_by_id.get(cid, 0.0)

        hybrid_score = alpha * b_score + (1 - alpha) * d_score
        hybrid_results.append({
            **chunk,
            "score":       hybrid_score,
            "bm25_score":  b_score,
            "dense_score": d_score
        })

    # --- Step 6: Sort and return top-k ---
    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)[:k]
    
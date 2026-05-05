# retrieval/hybrid_retriever.py
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# import our two retrievers
from retrieval.bm25_retriever import load_chunks, build_bm25_index, bm25_search
from retrieval.dense_retriever import build_dense_index, dense_search


def normalize_scores(results: list[dict]) -> list[dict]:
    """
    Normalizes scores to range [0, 1] so BM25 and dense
    scores are on the same scale before combining.

    Why? BM25 scores can be 0-10+, dense scores are 0-1.
    We can't fairly combine them without normalizing first.

    Example:
      BM25  scores: [4.4, 1.2, 0.7, 0.0, 0.0]
      After norm  : [1.0, 0.27, 0.16, 0.0, 0.0]

      Dense scores: [0.88, 0.87, 0.86, 0.85, 0.84]
      After norm  : [1.0, 0.96, 0.93, 0.90, 0.88]

      Now both are on same scale → fair to combine!
    """
    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)

    # avoid division by zero if all scores are equal
    if max_s - min_s == 0:
        return [{**r, "score": 0.0} for r in results]

    normalized = [
        {**r, "score": (r["score"] - min_s) / (max_s - min_s)}
        for r in results
    ]
    return normalized


def hybrid_search(
    query: str,
    chunks: list[dict],
    bm25_index: BM25Okapi,
    dense_model: SentenceTransformer,
    dense_embeddings: np.ndarray,
    k: int = 5,
    alpha: float = 0.5
) -> list[dict]:
    """
    Combines BM25 and dense search into hybrid retrieval.

    Formula:
      hybrid_score = alpha × bm25_score + (1 - alpha) × dense_score

    Alpha controls the balance:
      alpha = 1.0 → pure BM25  (keyword only)
      alpha = 0.5 → equal mix  (default)
      alpha = 0.0 → pure dense (semantic only)

    Alpha is what Optuna will tune later to find the best value!
    """
    # --- Step 1: get BM25 results for ALL chunks ---
    bm25_results = bm25_search(query, chunks, bm25_index, k=len(chunks))

    # --- Step 2: get Dense results for ALL chunks ---
    dense_results = dense_search(
        query, chunks, dense_model, dense_embeddings, k=len(chunks)
    )

    # --- Step 3: normalize both score lists to [0,1] ---
    bm25_norm = normalize_scores(bm25_results)
    dense_norm = normalize_scores(dense_results)

    # --- Step 4: build lookup dicts by chunk_id ---
    # so we can quickly find score for any chunk
    bm25_by_id = {r["chunk_id"]: r["score"] for r in bm25_norm}
    dense_by_id = {r["chunk_id"]: r["score"] for r in dense_norm}

    # --- Step 5: combine scores for every chunk ---
    hybrid_results = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        b_score = bm25_by_id.get(cid, 0.0)
        d_score = dense_by_id.get(cid, 0.0)

        # THE core hybrid formula
        hybrid_score = alpha * b_score + (1 - alpha) * d_score

        hybrid_results.append({
            **chunk,
            "score": hybrid_score,
            "bm25_score": b_score,    # keep for debugging
            "dense_score": d_score    # keep for debugging
        })

    # --- Step 6: sort by hybrid score, return top-k ---
    results = sorted(
        hybrid_results,
        key=lambda x: x["score"],
        reverse=True
    )[:k]

    return results


# evaluation/metrics.py
import json
import numpy as np
from retrieval.bm25_retriever import load_chunks, build_bm25_index, bm25_search
from retrieval.dense_retriever import build_dense_index, dense_search
from retrieval.hybrid_retriever import hybrid_search

def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """
    Checks if ANY relevant chunk appears in the top-k retrieved chunks.

    Returns 1.0 if found, 0.0 if not found.

    Example:
      retrieved_ids = [6, 5, 8, 29, 12]  (top 5 results)
      relevant_ids  = [6]                 (correct answer)
      k = 5
      → chunk 6 is in top 5 → return 1.0
    """
    # only look at top-k results
    top_k = retrieved_ids[:k]

    # check if any relevant chunk is in top-k
    found = any(rid in top_k for rid in relevant_ids)
    return 1.0 if found else 0.0


def dcg_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """
    Computes Discounted Cumulative Gain at k.

    Rewards finding relevant chunks AND finding them early.

    Formula:
      DCG = sum of (1 / log2(position + 1)) for each relevant chunk found

    Position 1 → 1/log2(2) = 1.000  ← full score
    Position 2 → 1/log2(3) = 0.630  ← less
    Position 3 → 1/log2(4) = 0.500  ← even less
    Position 5 → 1/log2(6) = 0.387  ← much less
    """
    dcg = 0.0
    for position, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / np.log2(position + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """
    Normalized DCG — divides DCG by the IDEAL DCG.

    Ideal DCG = what score you'd get if all relevant chunks
                were ranked at the very top (positions 1, 2, 3...)

    This normalizes the score to [0, 1] regardless of
    how many relevant chunks exist.

    Example:
      relevant_ids = [6, 7]  (2 correct answers)

      Ideal ranking: chunk 6 at pos 1, chunk 7 at pos 2
      IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63

      Your ranking: chunk 6 at pos 1, chunk 7 at pos 4
      DCG  = 1/log2(2) + 1/log2(5) = 1.0 + 0.43 = 1.43

      NDCG = 1.43 / 1.63 = 0.877
    """
    # actual DCG
    actual_dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    # ideal DCG — pretend relevant chunks are at positions 1, 2, 3...
    ideal_retrieved = relevant_ids[:k]
    ideal_dcg = dcg_at_k(ideal_retrieved, relevant_ids, k)

    # avoid division by zero
    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_retriever(
    search_fn,           # function that takes query → returns results
    gold_qa_path: str = "data/gold_qa.json",
    k: int = 5
) -> dict:
    """
    Runs the retriever on ALL questions in gold_qa.json
    and computes average Recall@k and NDCG@k.

    search_fn must return a list of dicts with "chunk_id" field.

    Returns:
      {
        "recall@k": 0.xx,
        "ndcg@k":   0.xx,
        "k":        5
      }
    """
    # load gold Q&A pairs
    with open(gold_qa_path, "r") as f:
        gold_qa = json.load(f)

    recall_scores = []
    ndcg_scores = []

    for qa in gold_qa:
        query = qa["question"]
        relevant_ids = qa["relevant_chunk_ids"]

        # run the retriever
        results = search_fn(query, k=k)

        # extract just the chunk_ids from results
        retrieved_ids = [r["chunk_id"] for r in results]

        # compute metrics for this question
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)

        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

    return {
        f"recall@{k}": round(np.mean(recall_scores), 4),
        f"ndcg@{k}":   round(np.mean(ndcg_scores), 4),
        "k": k
    }

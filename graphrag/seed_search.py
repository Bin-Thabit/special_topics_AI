"""
graphrag/seed_search.py
-----------------------
Gets anchor paper_ids for select_subgraph() by doing a fast
full-corpus hybrid search and deduplicating to the top N papers.

Why this exists:
    select_subgraph() (Step 1) needs a short list of "seed" paper_ids
    to start graph traversal from.  Rather than searching the graph
    blind, we first do a normal hybrid vector+BM25 search over ALL
    chunks to find which papers are most likely relevant, then hand
    those paper_ids to the graph so it can expand from there.

Usage (standalone smoke-test):
    python -m graphrag.seed_search
"""

from __future__ import annotations


def seed_search(
    query: str,
    state,
    alpha: float,
    query_vector,
    n_seed_papers: int = 5,
) -> list[str]:
    """
    Run a broad hybrid search and return the top-N unique paper_ids.

    Parameters
    ----------
    query        : the raw question string from the user
    state        : FastAPI app state (holds model, chunks, bm25, embeddings)
    alpha        : hybrid weight (0 = pure BM25, 1 = pure dense)
    query_vector : pre-computed query embedding (np.ndarray) — reuse so
                   we don't embed the query twice
    n_seed_papers: how many unique papers to return (default 5)

    Returns
    -------
    list of paper_id strings, e.g. ["2301.07041", "2210.11610", ...]
    """
    from retrieval.hybrid_retriever import hybrid_search

    # Search broadly — k=30 gives enough chunks to find 5 distinct papers
    hits = hybrid_search(
        query=query,
        chunks=state.all_chunks,
        bm25_index=state.bm25_index,
        dense_model=state.model,
        dense_embeddings=state.dense_embeddings,
        k=30,
        alpha=alpha,
        query_vector=query_vector,   # reuse — don't re-embed
    )

    # dict.fromkeys preserves insertion order while deduplicating
    # (first time a paper_id appears = its best-ranked chunk)
    seen = list(dict.fromkeys(h["paper_id"] for h in hits))

    return seen[:n_seed_papers]


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("seed_search smoke test")
    print("----------------------")
    print("This function needs a live app state to run.")
    print("It will be called automatically inside /ask.")
    print()
    print("What it does:")
    print("  1. Runs hybrid_search over all chunks (k=30)")
    print("  2. Extracts paper_id from each hit")
    print("  3. Deduplicates (keeps first/best occurrence per paper)")
    print("  4. Returns top-5 paper_ids as seed for select_subgraph()")
    print()
    print("Example output shape:  ['2301.07041', '2210.11610', '2305.00050', ...]")
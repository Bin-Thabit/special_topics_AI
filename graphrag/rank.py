"""
graphrag/rank.py — GraphRAG Step 3: hybrid blend + optional rerank (Option B).

Step 2 hands us a candidate pool (every chunk of the graph-selected papers).
This step RANKS that pool with Abdullah's hybrid_search() and returns the
final top-k chunks for the answer.

Option B = rank the pool itself, not the whole corpus. To do that correctly,
hybrid_search() needs three POSITIONALLY-ALIGNED inputs (chunks[i] <-> bm25 doc
i <-> dense_embeddings[i]), so we:
  1. build a FRESH BM25 over just the pool (also sidesteps the startup quirk
     where state.bm25_index is built before _build_dense_from_qdrant trims
     state.all_chunks — we never depend on that alignment),
  2. take the pool's dense vectors (sliced from the already-loaded
     state.dense_embeddings, or pulled from Qdrant for standalone runs),
  3. embed the query exactly like /search (BGE prefix + normalize),
  4. call hybrid_search() with alpha sourced from the same HybridWeightAdapter.

Ablation levers for D3:
  graph-guided : pool = graph papers (this module)
  full hybrid  : pool = state.all_chunks (the /search baseline)
  vector-only  : alpha = 0.0
"""
from __future__ import annotations

import uuid

import numpy as np

from retrieval.bm25_retriever import build_bm25_index
from retrieval.hybrid_retriever import hybrid_search
from ingestion.embedder import BGE_QUERY_PREFIX
from stores.qdrant_store import COLLECTION_NAME


# ---------------------------------------------------------------------------
# Alpha + query embedding — sourced identically to api/main.py /search
# ---------------------------------------------------------------------------

def adaptive_alpha(hybrid_adapter, default: float = 0.5) -> tuple[float, dict]:
    """Return (alpha, weights) from the D1 adapter, mirroring /search exactly."""
    if hybrid_adapter is not None:
        weights = hybrid_adapter.get_weights()
        return weights.get("dense_weight", default), weights
    return default, {"dense_weight": default, "bm25_weight": default}


def embed_query(model, query: str) -> np.ndarray:
    """Same as api.main.embed_query — BGE query prefix, normalized."""
    return model.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Get the pool's dense vectors, ALIGNED to the pool chunk order
# ---------------------------------------------------------------------------

def pool_embeddings_from_state(
    pool_chunks: list[dict],
    all_chunks: list[dict],
    dense_embeddings: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """Fast path for the live API: slice rows out of state.dense_embeddings."""
    idx = {c["chunk_id"]: i for i, c in enumerate(all_chunks)}
    kept, rows = [], []
    for c in pool_chunks:
        i = idx.get(c["chunk_id"])
        if i is not None:
            kept.append(c)
            rows.append(dense_embeddings[i])
    return kept, (np.array(rows) if rows else np.empty((0,)))


def pool_embeddings_from_qdrant(
    pool_chunks: list[dict],
    qdrant,
) -> tuple[list[dict], np.ndarray]:
    """
    Standalone path (tests / offline): pull the pool's vectors from Qdrant in a
    single batched retrieve. Uses the same uuid5 point-id scheme as
    api.main._build_dense_from_qdrant. Drops any chunk missing a vector.
    """
    point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])) for c in pool_chunks]
    records = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=point_ids,
        with_vectors=True,
    )
    vec_by_id = {str(r.id): r.vector for r in records}
    kept, rows = [], []
    for c, pid in zip(pool_chunks, point_ids):
        v = vec_by_id.get(pid)
        if v is not None:
            kept.append(c)
            rows.append(v)
    return kept, (np.array(rows) if rows else np.empty((0,)))


# ---------------------------------------------------------------------------
# Step 3 core: rank the pool
# ---------------------------------------------------------------------------

def rank_pool(
    query: str,
    pool_chunks: list[dict],
    model,
    pool_embeddings: np.ndarray,
    alpha: float = 0.5,
    k: int = 8,
    query_vector: np.ndarray | None = None,
    graph_weight: float = 0.0,
) -> list[dict]:
    """
    Hybrid-rank the candidate pool and return the top-k chunks.

    pool_chunks / pool_embeddings MUST be aligned (row i == chunk i) — use one
    of the pool_embeddings_* helpers to guarantee that.

    graph_weight (0..1): optionally fold the Step-1 graph score into the final
    ranking — final = (1-w)*hybrid + w*graph_norm. Default 0 = pure hybrid.
    Each result keeps `score` (final), `hybrid_score`, `bm25_score`,
    `dense_score`, and the `graph_score`/`graph_reasons` carried from Step 2.
    """
    if not pool_chunks or pool_embeddings.size == 0:
        return []

    pool_bm25 = build_bm25_index(pool_chunks)
    if query_vector is None:
        query_vector = embed_query(model, query)

    # rank the WHOLE pool (slice to k after the optional graph blend)
    ranked = hybrid_search(
        query=query,
        chunks=pool_chunks,
        bm25_index=pool_bm25,
        dense_model=model,
        dense_embeddings=pool_embeddings,
        k=len(pool_chunks),
        alpha=alpha,
        query_vector=query_vector,
    )

    if graph_weight > 0:
        gs = [r.get("graph_score", 0) for r in ranked]
        lo, hi = min(gs), max(gs)
        span = (hi - lo) or 1
        for r in ranked:
            r["hybrid_score"] = r["score"]
            g_norm = (r.get("graph_score", 0) - lo) / span
            r["score"] = (1 - graph_weight) * r["score"] + graph_weight * g_norm
        ranked.sort(key=lambda r: r["score"], reverse=True)

    return ranked[:k]


# ---------------------------------------------------------------------------
# Optional cross-encoder rerank (the "+ optional rerank" in the D3 spec)
# ---------------------------------------------------------------------------

def rerank_cross_encoder(
    query: str,
    results: list[dict],
    top_n: int = 8,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cross_encoder=None,
) -> list[dict]:
    """
    Re-score the top hybrid results with a cross-encoder, which reads
    (query, chunk) together and is sharper than bi-encoder similarity (but
    slower — run it only on the ~20-30 you already shortlisted). Off by default;
    call it explicitly. Pass a preloaded CrossEncoder to avoid reloading.
    Adds `rerank_score` to each result.
    """
    if not results:
        return results
    from sentence_transformers import CrossEncoder

    ce = cross_encoder or CrossEncoder(model_name)
    scores = ce.predict([(query, r["text"]) for r in results])
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)[:top_n]


if __name__ == "__main__":
    # Integration smoke test: Step 1 -> Step 2 -> Step 3, standalone (no API).
    from stores.neo4j_store import Neo4jStore
    from stores.mongo_store import get_mongo_client, get_mongo_db
    from stores.qdrant_store import get_qdrant_client
    from ingestion.embedder import load_model
    from graphrag.expand import expand_to_chunks

    query = "How does reinforcement learning handle long-horizon planning?"

    with Neo4jStore() as store:
        seed = store.run_query("MATCH (p:Paper) RETURN p.paper_id AS id LIMIT 1")[0]["id"]
        subgraph = store.select_subgraph(seed_paper_ids=[seed], max_papers=10)
    print(f"Step 1: {len(subgraph)} papers")

    db = get_mongo_db(get_mongo_client())
    pool = expand_to_chunks(db, subgraph)
    print(f"Step 2: {len(pool)} candidate chunks")

    model = load_model()
    qdrant = get_qdrant_client()
    pool, pool_emb = pool_embeddings_from_qdrant(pool, qdrant)
    print(f"        {len(pool)} chunks have vectors")

    top = rank_pool(query, pool, model, pool_emb, alpha=0.2149, k=8)  # alpha = your run_card best_alpha
    print(f"Step 3: top {len(top)} for query: {query!r}")
    for r in top:
        print(f"  {r['score']:.3f}  p{r['page_num']}  {r['paper_id']}  {r['text'][:60]}")
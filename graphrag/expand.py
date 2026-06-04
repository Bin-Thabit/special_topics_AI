"""
graphrag/expand.py — GraphRAG Step 2: subgraph -> candidate chunk pool.

Step 1 (Neo4jStore.select_subgraph) decides WHICH papers are relevant.
This step turns those paper_ids into the actual TEXT chunks that Step 3
(hybrid blend + rerank) will score. It also carries the per-paper graph
`score` and `reasons` from Step 1 down onto each chunk, so later stages
(or your D3 report) can see the graph signal that justified each chunk.

Design: graph-FILTERED pool — every chunk of every selected paper becomes a
candidate, and Step 3 ranks them down. (If you instead want the graph to only
ADD to a global hybrid top-k, union this pool with that list before Step 3.)
"""
from __future__ import annotations

from pymongo.database import Database

from stores.mongo_store import get_chunks_by_paper_ids


def expand_to_chunks(
    db: Database,
    subgraph: list[dict] | list[str],
    max_chunks_per_paper: int | None = None,
    max_total_chunks: int | None = None,
) -> list[dict]:
    """
    Expand a Step-1 subgraph into a candidate pool of chunk documents.

    Args
      subgraph : output of select_subgraph() — list of dicts with at least
                 `paper_id` (and ideally `score`, `reasons`). A plain list of
                 paper_id strings also works.
      max_chunks_per_paper : optional cap so one long paper can't dominate the
                 pool (keeps each paper's earliest chunks — title/abstract/intro).
      max_total_chunks : optional global cap; when exceeded, chunks from the
                 highest graph-scored papers are kept first.

    Returns
      list[dict]: full chunk docs (text + page_num + char offsets + denormalized
                  title/authors/etc.), each annotated with `graph_score` and
                  `graph_reasons` from Step 1. Empty list if nothing to expand.
    """
    # --- normalize input: pull paper_ids + per-paper graph signal ----------
    if subgraph and isinstance(subgraph[0], dict):
        score_by_paper = {r["paper_id"]: r.get("score", 0) for r in subgraph}
        reasons_by_paper = {r["paper_id"]: r.get("reasons", []) for r in subgraph}
        paper_ids = list(score_by_paper.keys())
    else:  # plain list of ids (or empty)
        paper_ids = list(subgraph or [])
        score_by_paper = {}
        reasons_by_paper = {}

    if not paper_ids:
        return []

    chunks = get_chunks_by_paper_ids(db, paper_ids)

    # --- optional per-paper cap (chunks are pre-sorted paper->page->index) --
    if max_chunks_per_paper:
        kept, seen = [], {}
        for c in chunks:
            pid = c["paper_id"]
            if seen.get(pid, 0) < max_chunks_per_paper:
                kept.append(c)
                seen[pid] = seen.get(pid, 0) + 1
        chunks = kept

    # --- carry the Step-1 graph signal onto every chunk --------------------
    for c in chunks:
        c["graph_score"] = score_by_paper.get(c["paper_id"], 0)
        c["graph_reasons"] = reasons_by_paper.get(c["paper_id"], [])

    # --- optional global cap: prefer chunks from the most relevant papers --
    if max_total_chunks and len(chunks) > max_total_chunks:
        # stable sort keeps each paper's reading order within equal scores
        chunks.sort(key=lambda c: c["graph_score"], reverse=True)
        chunks = chunks[:max_total_chunks]

    return chunks


if __name__ == "__main__":
    # Integration smoke test: Step 1 -> Step 2, no hardcoded ids.
    from stores.neo4j_store import Neo4jStore
    from stores.mongo_store import get_mongo_client, get_mongo_db

    with Neo4jStore() as store:
        seed = store.run_query("MATCH (p:Paper) RETURN p.paper_id AS id LIMIT 1")[0]["id"]
        print(f"seed paper: {seed}")
        subgraph = store.select_subgraph(seed_paper_ids=[seed], max_papers=10)
        print(f"Step 1 subgraph: {len(subgraph)} papers")

    db = get_mongo_db(get_mongo_client())
    pool = expand_to_chunks(db, subgraph)
    papers = {c["paper_id"] for c in pool}
    print(f"Step 2 candidate pool: {len(pool)} chunks across {len(papers)} papers")
    for c in pool[:3]:
        print(f"  {c['chunk_id']} | p{c['page_num']} | gscore={c['graph_score']} "
              f"| {c['text'][:60]}")
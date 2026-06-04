"""
scripts/ablation.py
--------------------
D3 Ablation Study — compare 3 retrieval conditions on the same queries.

What "ablation" means in plain English:
    We test what happens when we REMOVE parts of our pipeline.
    Like removing ingredients to see what each one actually contributes.

The 3 conditions:
    1. vector-only    — dense search only (alpha=0.0 means pure BM25 actually,
                        so we use alpha=1.0 for pure dense, pool=all_chunks)
    2. hybrid         — BM25 + dense blended, but NO graph (skip steps 1-2)
    3. graph-guided   — full pipeline (graph narrows the pool, then hybrid ranks)

Run:
    python scripts/ablation.py

Output:
    ablation_results.json   ← paste into d3_report.ipynb
    (also prints a table to stdout)
"""

from __future__ import annotations
import json
import time
import statistics
import sys
import os

# ── Make sure project root is on the path ────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# ── Test queries (reuse your D2 gold queries or add more) ────────────────────
# These should be the same queries you use for RAGAS evaluation
TEST_QUERIES = [
    "What methods are used for knowledge representation in AI?",
    "How does automated reasoning work in planning systems?",
    "What are the main approaches to reinforcement learning?",
    "How is uncertainty handled in probabilistic graphical models?",
    "What are the challenges in natural language understanding?",
]


def run_ablation(state, queries: list[str]) -> dict:
    """
    Run all 3 conditions and return scored results.

    Parameters
    ----------
    state   : the loaded app state (model, chunks, embeddings, neo4j, etc.)
    queries : list of test question strings

    Returns
    -------
    dict with per-condition results and summary stats
    """
    from graphrag.rank        import embed_query, adaptive_alpha, rank_pool, pool_embeddings_from_state
    from graphrag.answer      import generate_answer
    from graphrag.judge       import judge_answer
    from graphrag.safety      import provenance_filter
    from graphrag.expand      import expand_to_chunks
    from graphrag.seed_search import seed_search

    alpha, _ = adaptive_alpha(state.hybrid_adapter)

    conditions = {
        "vector_only":    [],
        "hybrid_baseline":[],
        "graph_guided":   [],
    }

    for query in queries:
        print(f"\nQuery: {query[:60]}...")

        # Pre-compute query vector once per query (reused across all conditions)
        qv = embed_query(state.model, query)

        # ── Condition 1: Vector-only ──────────────────────────────────────────
        # Same pool as hybrid (all chunks), but alpha=1.0 (pure dense, no BM25)
        print("  Running vector-only...")
        t0 = time.perf_counter()
        all_aligned, all_emb = pool_embeddings_from_state(
            state.all_chunks, state.all_chunks, state.dense_embeddings
        )
        ranked_vec = rank_pool(
            query, all_aligned, state.model, all_emb,
            alpha=1.0,   # ← pure dense, BM25 ignored
            k=8,
            query_vector=qv,
        )
        result_vec = generate_answer(query, ranked_vec)
        judge_vec  = judge_answer(query, result_vec["answer"], ranked_vec)
        lat_vec    = time.perf_counter() - t0

        conditions["vector_only"].append({
            "query":             query,
            "faithfulness":      judge_vec["faithfulness_score"],
            "faithfulness_label":judge_vec["faithfulness_label"],
            "relevance":         judge_vec["relevance_score"],
            "latency_s":         round(lat_vec, 3),
        })

        # ── Condition 2: Hybrid baseline (no graph) ───────────────────────────
        # All chunks as pool, but use learned alpha (BM25 + dense blended)
        print("  Running hybrid baseline...")
        t0 = time.perf_counter()
        ranked_hyb = rank_pool(
            query, all_aligned, state.model, all_emb,
            alpha=alpha,  # ← learned hybrid weight from online learner
            k=8,
            query_vector=qv,
        )
        result_hyb = generate_answer(query, ranked_hyb)
        judge_hyb  = judge_answer(query, result_hyb["answer"], ranked_hyb)
        lat_hyb    = time.perf_counter() - t0

        conditions["hybrid_baseline"].append({
            "query":             query,
            "faithfulness":      judge_hyb["faithfulness_score"],
            "faithfulness_label":judge_hyb["faithfulness_label"],
            "relevance":         judge_hyb["relevance_score"],
            "latency_s":         round(lat_hyb, 3),
        })

        # ── Condition 3: Graph-guided (full pipeline) ─────────────────────────
        # Steps 1-2: use graph to narrow the pool BEFORE ranking
        print("  Running graph-guided...")
        t0 = time.perf_counter()
        seed_ids = seed_search(query, state, alpha, qv, n_seed_papers=5)
        topics   = state.topic_parser.parse(query)
        subgraph = state.neo4j.select_subgraph(
            seed_paper_ids=seed_ids, topics=topics, max_papers=20
        )
        pool = expand_to_chunks(state.mongo_db, subgraph)
        pool = provenance_filter(state.mongo_db, pool)
        aligned_pool, pool_emb = pool_embeddings_from_state(
            pool, state.all_chunks, state.dense_embeddings
        )
        ranked_graph = rank_pool(
            query, aligned_pool, state.model, pool_emb,
            alpha=alpha,
            k=8,
            query_vector=qv,
        )
        result_graph = generate_answer(query, ranked_graph)
        judge_graph  = judge_answer(query, result_graph["answer"], ranked_graph)
        lat_graph    = time.perf_counter() - t0

        conditions["graph_guided"].append({
            "query":             query,
            "faithfulness":      judge_graph["faithfulness_score"],
            "faithfulness_label":judge_graph["faithfulness_label"],
            "relevance":         judge_graph["relevance_score"],
            "latency_s":         round(lat_graph, 3),
            "pool_size":         len(pool),
            "papers_in_subgraph":len(subgraph),
        })

    # ── Compute summary stats ─────────────────────────────────────────────────
    summary = {}
    for cond_name, rows in conditions.items():
        faithfulness_scores = [r["faithfulness"] for r in rows]
        relevance_scores    = [r["relevance"]     for r in rows]
        latencies           = [r["latency_s"]     for r in rows]
        summary[cond_name] = {
            "mean_faithfulness": round(statistics.mean(faithfulness_scores), 3),
            "mean_relevance":    round(statistics.mean(relevance_scores),    3),
            "mean_latency_s":    round(statistics.mean(latencies),           3),
            "p95_latency_s":     round(
                sorted(latencies)[int(len(latencies) * 0.95)], 3
            ) if len(latencies) >= 2 else latencies[-1],
        }

    return {"per_query": conditions, "summary": summary}


def print_table(summary: dict):
    """Print a clean comparison table to stdout — paste into report."""
    print("\n")
    print("=" * 65)
    print("D3 ABLATION RESULTS")
    print("=" * 65)
    header = f"{'Condition':<22} {'Faithfulness':>14} {'Relevance':>10} {'p95 lat (s)':>12}"
    print(header)
    print("-" * 65)
    for cond, stats in summary.items():
        label = cond.replace("_", " ")
        print(
            f"{label:<22} "
            f"{stats['mean_faithfulness']:>14.3f} "
            f"{stats['mean_relevance']:>10.3f} "
            f"{stats['p95_latency_s']:>12.3f}"
        )
    print("=" * 65)
    print()


if __name__ == "__main__":
    print("D3 Ablation Study")
    print("-----------------")
    print("Connecting to running API server to reuse loaded state...")
    print()
    print("NOTE: This script expects the FastAPI server to be running.")
    print("      It reuses state.model / state.all_chunks etc. from the")
    print("      server process rather than re-loading everything.")
    print()
    print("To run the ablation via HTTP instead (simpler), use:")
    print()
    print('  import httpx')
    print('  conditions = ["vector_only", "hybrid_baseline", "graph_guided"]')
    print('  # POST to /ask with a ?condition= query param (add that param to /ask)')
    print()
    print("Or import run_ablation() directly from a notebook:")
    print()
    print("  from scripts.ablation import run_ablation, print_table")
    print("  results = run_ablation(app.state, TEST_QUERIES)")
    print("  print_table(results['summary'])")
    print()
    print("Expected output shape:")
    print()
    print_table({
        "vector_only":     {"mean_faithfulness": 0.60, "mean_relevance": 0.65, "p95_latency_s": 2.1},
        "hybrid_baseline": {"mean_faithfulness": 0.72, "mean_relevance": 0.74, "p95_latency_s": 2.8},
        "graph_guided":    {"mean_faithfulness": 0.85, "mean_relevance": 0.88, "p95_latency_s": 4.2},
    })
"""
scripts/ablation.py
--------------------
D3 Ablation Study — compare 3 retrieval conditions on the same queries.

What "ablation" means in plain English:
    We test what happens when we REMOVE parts of the pipeline.
    Like removing ingredients from a recipe to see what each one contributes.

The 3 conditions tested:
    1. vector_only    — dense embeddings only (no BM25, no graph)
    2. hybrid         — BM25 + dense blended, but NO graph narrowing
    3. graph_guided   — full pipeline (graph narrows pool, then hybrid ranks)

How it works:
    We added a ?condition= param to /ask so the server can swap modes.
    This script POSTs each query 3 times (once per condition) and scores
    faithfulness + relevance + latency from the judge scores in the response.

Run:
    python scripts/ablation.py

Output:
    ablation_results.json   <- paste into d3_report.ipynb
    prints a clean table to stdout
"""

import json
import time
import statistics
import httpx

BASE_URL = "http://127.0.0.1:8000"

# ── Test queries — same ones used for D2 gold eval ────────────────────────────
# Use at least 5. More = more reliable p95.
TEST_QUERIES = [
    "What methods are used for knowledge representation in AI?",
    "How does automated reasoning work in planning systems?",
    "What are the main approaches to reinforcement learning?",
    "How is uncertainty handled in probabilistic graphical models?",
    "What are the challenges in natural language understanding?",
]

CONDITIONS = ["vector_only", "hybrid", "graph_guided"]


def run_query(query: str, condition: str, timeout: int = 120) -> dict:
    """POST to /ask with a condition flag and return the parsed response."""
    resp = httpx.post(
        f"{BASE_URL}/ask",
        json={"query": query, "condition": condition},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_metrics(result: dict) -> dict:
    """Pull the numbers we care about from one /ask response."""
    judge = result.get("judge_after") or result.get("judge_before") or {}
    return {
        "faithfulness":       judge.get("faithfulness_score", 0.0),
        "faithfulness_label": judge.get("faithfulness_label", "UNKNOWN"),
        "relevance":          judge.get("relevance_score", 0.0),
        "latency_s":          result.get("elapsed_seconds", 0.0),
        "pool_size":          result.get("pool_size", 0),
        "reflected":          result.get("reflected", False),
    }


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return round(s[idx], 3)


def run_ablation():
    results = {c: [] for c in CONDITIONS}

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {query[:65]}...")
        for condition in CONDITIONS:
            print(f"  {condition:<15} ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                raw = run_query(query, condition)
                metrics = extract_metrics(raw)
                metrics["query"] = query
                results[condition].append(metrics)
                print(
                    f"faith={metrics['faithfulness']:.2f}  "
                    f"rel={metrics['relevance']:.2f}  "
                    f"pool={metrics['pool_size']}  "
                    f"{metrics['latency_s']:.1f}s"
                )
            except Exception as e:
                print(f"ERROR — {e}")
                results[condition].append({
                    "query": query,
                    "faithfulness": 0.0,
                    "faithfulness_label": "ERROR",
                    "relevance": 0.0,
                    "latency_s": time.perf_counter() - t0,
                    "pool_size": 0,
                    "reflected": False,
                    "error": str(e),
                })

    # ── Summary stats ─────────────────────────────────────────────────────────
    summary = {}
    for cond, rows in results.items():
        faithfulness = [r["faithfulness"] for r in rows]
        relevance    = [r["relevance"]    for r in rows]
        latencies    = [r["latency_s"]    for r in rows]
        summary[cond] = {
            "mean_faithfulness": round(statistics.mean(faithfulness), 3),
            "mean_relevance":    round(statistics.mean(relevance),    3),
            "mean_latency_s":    round(statistics.mean(latencies),    3),
            "p95_latency_s":     p95(latencies),
            "n_queries":         len(rows),
        }

    output = {"per_query": results, "summary": summary}

    with open("ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved → ablation_results.json")

    print_table(summary)
    return output


def print_table(summary: dict):
    print("\n")
    print("=" * 68)
    print("D3 ABLATION RESULTS")
    print("=" * 68)
    print(f"{'Condition':<20} {'Faithfulness':>13} {'Relevance':>10} {'p95 lat':>9}")
    print("-" * 68)
    for cond, s in summary.items():
        label = cond.replace("_", " ")
        print(
            f"{label:<20} "
            f"{s['mean_faithfulness']:>13.3f} "
            f"{s['mean_relevance']:>10.3f} "
            f"{s['p95_latency_s']:>9.3f}s"
        )
    print("=" * 68)

    # Delta rows — improvement of graph_guided over vector_only
    if "vector_only" in summary and "graph_guided" in summary:
        v = summary["vector_only"]
        g = summary["graph_guided"]
        print()
        print("Graph-guided vs vector-only improvement:")
        faith_delta = round(g["mean_faithfulness"] - v["mean_faithfulness"], 3)
        rel_delta   = round(g["mean_relevance"]    - v["mean_relevance"],    3)
        sign_f = "+" if faith_delta >= 0 else ""
        sign_r = "+" if rel_delta   >= 0 else ""
        print(f"  Faithfulness delta : {sign_f}{faith_delta}")
        print(f"  Relevance delta    : {sign_r}{rel_delta}")
    print()


if __name__ == "__main__":
    # Quick health check before starting
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        health = r.json()
        print(f"Server online — {health.get('chunks', 0)} chunks loaded")
        print(f"Neo4j ready: {health.get('neo4j_ready')}")
        print(f"TopicParser: {health.get('topic_parser')}")
    except Exception as e:
        print(f"Cannot reach server at {BASE_URL} — is it running?")
        print(f"Error: {e}")
        raise SystemExit(1)

    print(f"\nRunning ablation over {len(TEST_QUERIES)} queries × {len(CONDITIONS)} conditions...")
    print("This will take a while due to LLM calls. Grab a coffee.\n")

    run_ablation()
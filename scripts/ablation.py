"""
scripts/ablation.py
--------------------
D3 Ablation Study — corpus-specific gold queries from D2 eval set.

Each query targets one specific paper in the corpus.
This makes the ablation traceable: we know exactly which paper
the correct answer should come from.

Run:
    python scripts/ablation.py

Output:
    ablation_results.json
    prints summary table to stdout
"""

import json
import time
import statistics
import httpx

BASE_URL = "http://127.0.0.1:8000"

# ── Gold queries — each tied to one paper_id ─────────────────────────────────
TEST_QUERIES = [
    {
        "query":    "How does ActCam control camera motion without training?",
        "paper_id": "2605.06667v1",
    },
    {
        "query":    "What is the main computational drawback of the RLPD algorithm?",
        "paper_id": "2605.05863v1",
    },
    {
        "query":    "What does the acronym VARS-FL stand for, and what is its primary objective in federated learning?",
        "paper_id": "2605.05896v1",
    },
    {
        "query":    "What does the acronym Wisteria stand for, and what inspired its name in the context of DNA language models?",
        "paper_id": "2605.05913v1",
    },
    {
        "query":    "What does the acronym LOD KG stand for, and why is it considered a viable tool for mitigating the digital divide between high- and low-resource languages?",
        "paper_id": "2605.05929v1",
    },
    {
        "query":    "What does the acronym HMW stand for, and what are the two core limitations in planner-facing latents it addresses?",
        "paper_id": "2605.05951v1",
    },
    {
        "query":    "What does the acronym Cola DLM stand for, and how does its hierarchical latent-variable paradigm decompose text generation to avoid token-level left-to-right serialization constraints?",
        "paper_id": "2605.06548v1",
    },
]

CONDITIONS = ["vector_only", "hybrid", "graph_guided"]


def run_query(query: str, condition: str, timeout: int = 180) -> dict:
    resp = httpx.post(
        f"{BASE_URL}/ask",
        json={"query": query, "condition": condition},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_metrics(result: dict, expected_paper_id: str) -> dict:
    """
    Pull metrics from one /ask response.
    Also checks whether the expected paper_id appears in the citations —
    this is our ground-truth hit check (did the graph find the right paper?).
    """
    judge = result.get("judge_after") or result.get("judge_before") or {}

    # Check if the correct paper appears in citations
    citations    = result.get("citations", [])
    cited_papers = {c.get("paper_id") for c in citations}
    correct_hit  = expected_paper_id in cited_papers

    # Also check seed_papers (graph may have found it even if not cited)
    seed_papers  = result.get("seed_papers", [])
    in_seed      = expected_paper_id in seed_papers

    return {
        "faithfulness":       judge.get("faithfulness_score", 0.0),
        "faithfulness_label": judge.get("faithfulness_label", "UNKNOWN"),
        "relevance":          judge.get("relevance_score", 0.0),
        "latency_s":          result.get("elapsed_seconds", 0.0),
        "pool_size":          result.get("pool_size", 0),
        "reflected":          result.get("reflected", False),
        "correct_paper_cited": correct_hit,   # gold paper appeared in citations
        "correct_paper_seeded": in_seed,      # gold paper found in seed search
    }


def p95(values: list) -> float:
    if not values:
        return 0.0
    s   = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return round(s[idx], 3)


def run_ablation():
    results = {c: [] for c in CONDITIONS}

    for i, item in enumerate(TEST_QUERIES, 1):
        query    = item["query"]
        paper_id = item["paper_id"]
        print(f"\n[{i}/{len(TEST_QUERIES)}] {query[:70]}...")
        print(f"  Expected paper: {paper_id}")

        for condition in CONDITIONS:
            print(f"  {condition:<15} ", end="", flush=True)
            try:
                raw = run_query(query, condition)
                m   = extract_metrics(raw, paper_id)
                m["query"]    = query
                m["paper_id"] = paper_id
                results[condition].append(m)
                hit = "✓ hit" if m["correct_paper_cited"] else "✗ miss"
                print(
                    f"faith={m['faithfulness']:.2f}  "
                    f"rel={m['relevance']:.2f}  "
                    f"pool={m['pool_size']}  "
                    f"{m['latency_s']:.1f}s  "
                    f"{hit}  "
                    f"{'reflected' if m['reflected'] else 'clean'}"
                )
            except Exception as e:
                print(f"ERROR — {e}")
                results[condition].append({
                    "query":    query,
                    "paper_id": paper_id,
                    "faithfulness": 0.0,
                    "faithfulness_label": "ERROR",
                    "relevance": 0.0,
                    "latency_s": 0.0,
                    "pool_size": 0,
                    "reflected": False,
                    "correct_paper_cited":  False,
                    "correct_paper_seeded": False,
                    "error": str(e),
                })

    # ── Summary stats ─────────────────────────────────────────────────────────
    summary = {}
    for cond, rows in results.items():
        faith   = [r["faithfulness"] for r in rows]
        rel     = [r["relevance"]    for r in rows]
        lats    = [r["latency_s"]    for r in rows]
        hits    = [r["correct_paper_cited"] for r in rows]
        summary[cond] = {
            "mean_faithfulness": round(statistics.mean(faith), 3),
            "mean_relevance":    round(statistics.mean(rel),   3),
            "mean_latency_s":    round(statistics.mean(lats),  3),
            "p95_latency_s":     p95(lats),
            "citation_hit_rate": round(sum(hits) / len(hits), 3),  # how often correct paper cited
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
    print("=" * 78)
    print("D3 ABLATION RESULTS")
    print("=" * 78)
    print(f"{'Condition':<20} {'Faithfulness':>13} {'Relevance':>10} {'Hit Rate':>10} {'p95 lat':>9}")
    print("-" * 78)
    for cond, s in summary.items():
        label = cond.replace("_", " ")
        print(
            f"{label:<20} "
            f"{s['mean_faithfulness']:>13.3f} "
            f"{s['mean_relevance']:>10.3f} "
            f"{s['citation_hit_rate']:>10.3f} "
            f"{s['p95_latency_s']:>9.3f}s"
        )
    print("=" * 78)

    # Delta rows
    if "vector_only" in summary and "graph_guided" in summary:
        v = summary["vector_only"]
        g = summary["graph_guided"]
        print()
        print("Graph-guided vs vector-only improvement:")
        fd = round(g["mean_faithfulness"]  - v["mean_faithfulness"],  3)
        rd = round(g["mean_relevance"]     - v["mean_relevance"],     3)
        hd = round(g["citation_hit_rate"]  - v["citation_hit_rate"],  3)
        print(f"  Faithfulness delta : {'+' if fd >= 0 else ''}{fd}")
        print(f"  Relevance delta    : {'+' if rd >= 0 else ''}{rd}")
        print(f"  Hit rate delta     : {'+' if hd >= 0 else ''}{hd}")
    print()


if __name__ == "__main__":
    try:
        r      = httpx.get(f"{BASE_URL}/health", timeout=5)
        health = r.json()
        print(f"Server online — {health.get('chunks', 0)} chunks loaded")
        print(f"Neo4j ready  : {health.get('neo4j_ready')}")
        print(f"TopicParser  : {health.get('topic_parser')}")
    except Exception as e:
        print(f"Cannot reach server at {BASE_URL} — is it running?\n{e}")
        raise SystemExit(1)

    print(f"\nRunning {len(TEST_QUERIES)} queries × {len(CONDITIONS)} conditions")
    print("Estimated time: ~30 minutes total\n")
    run_ablation()
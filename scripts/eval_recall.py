# scripts/eval_recall.py
"""
scripts/eval_recall.py
-----------------------
Measures Recall@k and MRR on a small gold query set.
Target: Recall@5 >= 0.60

Usage:
    python scripts/eval_recall.py
    python scripts/eval_recall.py --k 5
    python scripts/eval_recall.py --k 10
"""

import argparse
import json
import time
import httpx

# ---------------------------------------------------------------------------
# Gold query set — 10 queries with known relevant paper_id
# Edit these with queries from your actual corpus
# ---------------------------------------------------------------------------

GOLD_QUERIES = [
    {
        "query":    "how does ActCam control camera motion without training?",
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
        "query":    "What is the fundamental barrier to using standard Reinforcement Learning (RL) in real-world power grids?",
        "paper_id": "2604.14032v1",
    },
    {
        "query":    "What are the promising future research directions identified in the literature review for AI-driven BPMN model generation?",
        "paper_id": "2604.14034v1",
    },
    {
        "query":    "What is Value Gradient Flow (VGF) and how does it approach behavior-regularized reinforcement learning?",
        "paper_id": "2604.14265v1",
    },
    {
        "query":    "What unique fine-tuning methodology does the DharmaOCR paper introduce to handle text degeneration in structured OCR?",
        "paper_id": "2604.14314v1",
    },
]

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

API_URL    = "http://localhost:8000"
SEARCH_URL = f"{API_URL}/search"


def recall_at_k(results: list[dict], paper_id: str, k: int) -> int:
    """Return 1 if paper_id appears in top-k results, else 0."""
    top_k = results[:k]
    return int(any(r["paper_id"] == paper_id for r in top_k))


def reciprocal_rank(results: list[dict], paper_id: str) -> float:
    """Return 1/rank if paper_id found, else 0."""
    for i, r in enumerate(results, start=1):
        if r["paper_id"] == paper_id:
            return 1.0 / i
    return 0.0


def run_eval(k: int) -> None:
    print(f"Evaluating Recall@{k} and MRR on {len(GOLD_QUERIES)} queries\n")
    print(f"{'Query':<55} {'Hit@'+str(k):<8} {'RR':<8} {'Latency'}")
    print("-" * 85)

    recall_scores = []
    rr_scores     = []
    latencies     = []

    all_results   = []

    for item in GOLD_QUERIES:
        query    = item["query"]
        paper_id = item["paper_id"]

        start = time.time()
        try:
            response = httpx.post(
                SEARCH_URL,
                json    ={"query": query, "top_k": max(k, 10)},
                timeout =30,
            )
            elapsed = round(time.time() - start, 3)
            data    = response.json()
            results = data.get("results", [])

            hit = recall_at_k(results, paper_id, k)
            rr  = reciprocal_rank(results, paper_id)

            recall_scores.append(hit)
            rr_scores.append(rr)
            latencies.append(elapsed)

            all_results.append({
                "query":    query,
                "paper_id": paper_id,
                "hit":      hit,
                "rr":       round(rr, 4),
                "latency":  elapsed,
                "top_results": [
                    {
                        "paper_id": r["paper_id"],
                        "page_num": r["page_num"],
                        "score":    r["score"],
                        "text":     r["text"][:120],
                    }
                    for r in results[:3]
                ],
            })

            print(
                f"  {query[:53]:<55} "
                f"{'✅' if hit else '❌':<8} "
                f"{rr:<8.4f} "
                f"{elapsed}s"
            )

        except Exception as e:
            print(f"  {query[:53]:<55} ERROR: {e}")
            recall_scores.append(0)
            rr_scores.append(0.0)
            latencies.append(0.0)

    # Summary
    recall_at_k_score = sum(recall_scores) / len(recall_scores)
    mrr               = sum(rr_scores)     / len(rr_scores)
    p95_latency       = sorted(latencies)[int(len(latencies) * 0.95)]
    avg_latency       = sum(latencies)     / len(latencies)

    print("\n" + "=" * 85)
    print(f"  Recall@{k}       : {recall_at_k_score:.2f}   (target >= 0.60)")
    print(f"  MRR             : {mrr:.4f}")
    print(f"  Avg latency     : {avg_latency:.3f}s")
    print(f"  p95 latency     : {p95_latency:.3f}s  (target <= 2s)")
    print("=" * 85)

    # Pass/fail
    print(f"\n  Recall@{k} {'✅ PASS' if recall_at_k_score >= 0.60 else '❌ FAIL — below 0.60 target'}")
    print(f"  p95 latency {'✅ PASS' if p95_latency <= 2.0 else '❌ FAIL — above 2s target'}")

    # Save results
    output = {
        f"recall@{k}":  round(recall_at_k_score, 4),
        "mrr":          round(mrr, 4),
        "avg_latency":  round(avg_latency, 4),
        "p95_latency":  round(p95_latency, 4),
        "queries":      all_results,
    }
    with open("data/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to data/eval_results.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Recall@k (default: 5)")
    args = parser.parse_args()
    run_eval(k=args.k)
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
        "query": "What does the acronym Cola DLM stand for, and how does its hierarchical latent-variable paradigm decompose text generation to avoid token-level left-to-right serialization constraints?",
        "paper_id": "2605.06548v1",
    },
    {
        "query": "What does the acronym STAT stand for, and what specific process-level diagnostics does it introduce to supplement outcome-level rewards in multi-agent reinforcement learning evaluation?",
        "paper_id": "2605.06557v1",
    },
    {
        "query": "What does the acronym GONO stand for, and how does it utilize the consecutive cosine similarity signal (cc_t) to dynamically adapt Adam's momentum coefficient?",
        "paper_id": "2605.06575v1",
    },
    {
        "query": "What is the direction-loss decoupling phenomenon identified in deep learning optimization, and why do standard magnitude-based criteria fail to distinguish it?",
        "paper_id": "2605.06575v1",
    },
    {
        "query": "How does the proposed deterministic adjoint matching framework formulate human preference alignment for flow models, and what computational benefit does its truncated adjoint scheme provide?",
        "paper_id": "2605.06583v1",
    },
    {
        "query": "What does the acronym NeuroAgent stand for, and what are the primary core modules in its hierarchical multi-agent architecture used to automate multimodal neuroimaging preprocessing?",
        "paper_id": "2605.06584v1",
    },
    {
        "query": "Explain the mechanics and purpose of the feedback-driven 'Generate-Execute-Validate' engine within the context of automated neuroimaging error recovery pipelines.",
        "paper_id": "2605.06584v1",
    },
    {
        "query": "What does the acronym ConAXps stand for, and how does it utilize concept erasure to verify causal necessity compared to standard feature sensitivity approaches?",
        "paper_id": "2605.06640v1",
    },
    {
        "query": "Explain the difference between the NaiveEnum and XpSatEnum algorithms when extracting concept-based explanations across a given behavior set of images.",
        "paper_id": "2605.06640v1",
    },
    {
        "query": "What are the two sequential steps used by the GlazyBench framework to connect raw material formulation parameters to final perceptible visual tiles?",
        "paper_id": "2605.06641v1",
    },
    {
        "query": "What are the four core post-firing visual and physical prediction tasks supported by the GlazyBench benchmark dataset?",
        "paper_id": "2605.06641v1",
    },
    {
        "query": "What does the acronym StraTA stand for, and how does its explicit trajectory-level guidance fix the short-sighted exploration of purely reactive agents?",
        "paper_id": "2605.06642v1",
    },
    {
        "query": "How does the StraTA framework construct hierarchical groups over GRPO-style rollouts to optimize strategy generation and action execution jointly?",
        "paper_id": "2605.06642v1",
    },
    {
        "query": "What does the acronym RAO stand for, and how does it apply a local node reward structure with a delegation bonus to avoid the quantity-over-quality spawning problem?",
        "paper_id": "2605.06639v1",
    },
    {
        "query": "How does the RAO framework utilize depth-level inverse-frequency weighting to prevent heavily populated execution tree depths from dominating policy learning?",
        "paper_id": "2605.06639v1",
    },
    {
        "query": "What does the acronym MMDG-Bench stand for, and what are the three core task families it unifies to standardize evaluation practices?",
        "paper_id": "2605.06643v1",
    },
    {
        "query": "What is the rank inversion phenomenon identified in MMDG-Bench under input corruption, and why does clean benchmark performance fail to predict robustness?",
        "paper_id": "2605.06643v1",
    },
    {
        "query": "What does the acronym LOD KG stand for, and what are the primary multilingual knowledge graphs analyzed to categorize low-resource languages on the Semantic Web?",
        "paper_id": "2605.05929v1",
    },
    {
        "query": "What key structural variables are identified to characterize the uneven distribution of languages across Open Access Data (OAD) frameworks?",
        "paper_id": "2605.05929v1",
    },
    {
        "query": "What two core linguistic strategies are proposed to optimize cross-lingual transfer candidate selection for multilingual knowledge graph completion?",
        "paper_id": "2605.05931v1",
    },
    {
        "query": "How can analogical reasoning based on language proximity be utilized to address the digital invisibility of low-resource languages in Linked Open Data?",
        "paper_id": "2605.05931v1",
    },
    {
        "query": "What does the acronym CUAs stand for, and how are these circuit blocks integrated into pre-trained large language models to enable quantum enhancement?",
        "paper_id": "2605.05914v1",
    },
    {
        "query": "What fundamental constraint of classical memory allocation do Cayley-parameterized unitary adapters address when executing models on real quantum hardware?",
        "paper_id": "2605.05914v1",
    },
    {
        "query": "What does the acronym SECDA-DSE stand for, and how does it integrate large language models to automate the design space exploration of FPGA-based accelerators?",
        "paper_id": "2605.05920v1",
    },
    {
        "query": "How does the SECDA-DSE framework combine a structured explorer module with an LLM Stack to optimize memory hierarchies and dataflow strategies?",
        "paper_id": "2605.05920v1",
    },
    {
        "query": "What are the core mechanics of the 'Intentmaking' and 'Sensemaking' loops during human-AI interaction in AI-guided mathematical discovery?",
        "paper_id": "2605.05921v1",
    },
    {
        "query": "What challenges do domain experts face when trying to formulate mathematical constraints and interpret algorithmic outputs using interactive AI stack interfaces?",
        "paper_id": "2605.05921v1",
    },
    {
        "query": "What does the acronym AuxPath-FM stand for, and how does it generalize conditional flow matching by incorporating non-Gaussian distributions into the probability path?",
        "paper_id": "2605.06364v1",
    },
    {
        "query": "How does the AuxPath-FM framework implement trajectory-level classifier-free guidance (CFG) via auxiliary variables using only a single backbone evaluation?",
        "paper_id": "2605.06364v1",
    },
    {
        "query": "What is execution lineage in AI-native workflows, and how does it represent agentic work as a directed acyclic graph (DAG) to support identity-based replay?",
        "paper_id": "2605.06365v1",
    },
    {
        "query": "Explain how execution-lineage replay prevents unrelated-branch contamination and churn during policy-memo update tasks compared to loop-centric baselines.",
        "paper_id": "2605.06365v1",
    },
    {
        "query": "What does the acronym eX2L stand for, and how does it decorrelate confounding features by penalizing similarity between primary and confounder Grad-CAM activation maps?",
        "paper_id": "2605.06368v1",
    },
    {
        "query": "What performance gains does the eX2L framework achieve over Empirical Risk Minimization (ERM) on the Spawrious Many-to-Many Hard Challenge benchmark?",
        "paper_id": "2605.06368v1",
    },
    {
        "query": "How is persistent homology applied to the embedding matrices of modular arithmetic models to reveal a consistent topological signature of grokking?",
        "paper_id": "2605.06352v1",
    },
    {
        "query": "What sharp transitions in maximum and total persistence of first homology (H1) distinguish generalization from memorization in the data regimes of grokking?",
        "paper_id": "2605.06352v1",
    },
    {
        "query": "What does the acronym MEFA stand for, and how does it utilize gradient checkpointing to enable exact end-to-end full-gradient attacks through long purification trajectories?",
        "paper_id": "2605.06357v1",
    },
    {
        "query": "How does the SoftLeaky Relu activation function help alleviate the exponential decay of gradients in deep architectures evaluated under the MEFA framework?",
        "paper_id": "2605.06357v1",
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
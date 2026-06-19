"""
scripts/ablation_d4.py
-----------------------
D4 ablation harness — extends D3's 3-condition study with an LLM axis.

Runs 6 gold queries × 3 conditions × 2 LLMs = 36 total /ask calls.

Conditions  : graph_guided | hybrid | bm25_only (was vector_only - see note)
LLMs        : openrouter (~70B free tier)  |  tuned_local (QLoRA Qwen2.5-1.5B)

NOTE on bm25_only naming:
    With our hybrid_score = alpha*BM25 + (1-alpha)*dense formula, setting
    alpha=1.0 actually gives PURE BM25, not pure dense. We renamed the
    condition from "vector_only" to "bm25_only" to reflect this honestly.

NOTE on gold queries:
    These are 6 content-grounded questions tied to specific chunks (not the
    acronym-definition queries used previously). Each query has a target
    chunk_id that the answer should ideally cite, allowing us to measure
    retrieval precision at the chunk level (not just the paper level).
    All queries are from the HELD-OUT EVAL SET — they were NOT seen by the
    tuned model during QLoRA training, ensuring fair comparison.

Saves per-query results to ablation_d4_results.json and prints a summary table.

Usage:
    # Start the server first (no --reload):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    # Then in another shell:
    python scripts/ablation_d4.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL     = "http://127.0.0.1:8000/ask"
TIMEOUT     = 600
OUTPUT_PATH = Path("ablation_d4_results.json")

CONDITIONS = ["graph_guided", "hybrid", "bm25_only"]
LLMS       = ["openrouter", "tuned_local"]

# 6 content-grounded gold queries — each tied to one specific chunk + paper
# All from the held-out eval set; not seen during QLoRA training.
GOLD_QUERIES = [
    {
        "query":     "What does NeurIPS require submissions to provide for reproducibility?",
        "target":    "2605.06187v1",
        "chunk_id":  "2605.06187v1_p28_c1",
    },
    {
        "query":     "What does Recursive Agent Optimization (RAO) train across all nodes of the recursively generated execution tree?",
        "target":    "2605.06639v1",
        "chunk_id":  "2605.06639v1_p12_c0",
    },
    {
        "query":     "How many unique environment-dataset combinations are described in the dataset setup?",
        "target":    "2605.05863v1",
        "chunk_id":  "2605.05863v1_p5_c1",
    },
    {
        "query":     "How many more tokens does Petri use overall compared to SimpleAudit, and which role contributes most to this difference?",
        "target":    "2605.06652v1",
        "chunk_id":  "2605.06652v1_p19_c0",
    },
    {
        "query":     "What is the purpose of PersonaKit (PK) as described in the abstract?",
        "target":    "2605.06007v1",
        "chunk_id":  "2605.06007v1_p1_c0",
    },
    {
        "query":     "What is the nest parameter value used in the nested-logit choice model?",
        "target":    "2605.06529v1",
        "chunk_id":  "2605.06529v1_p2_c1",
    },
    {
        "query":     "At what time does the baby remain looking left while the person continues to lean on the walker?",
        "target":    "2605.06094v1",
        "chunk_id":  "2605.06094v1_p29_c1",
    },
    {
        "query":     "What test mean absolute error (MAE) did the proposed model achieve on the DIVA-HisDB benchmark?",
        "target":    "2605.06475v1",
        "chunk_id":  "2605.06475v1_p1_c0",
    },
    {
        "query":     "What are the two stages of the ProCompNav framework?",
        "target":    "2605.06223v1",
        "chunk_id":  "2605.06223v1_p1_c0",
    },
    {
        "query":     "In On-Policy Self-Distillation, what additional information does the teacher have compared to the student?",
        "target":    "2605.06188v1",
        "chunk_id":  "2605.06188v1_p3_c0",
    },
    {
        "query":     "What does Proposition 3 provide intuition for?",
        "target":    "2605.06474v1",
        "chunk_id":  "2605.06474v1_p8_c0",
    },
    {
        "query":     "What is the FID of the U-Net + P-Guide model on ImageNet-1k (256×256) at guidance scale w = 1.1?",
        "target":    "2605.06124v1",
        "chunk_id":  "2605.06124v1_p9_c0",
    },
    {
        "query":     "What does the strict policy state for category (4) regarding sharing user data with a third party?",
        "target":    "2605.06161v1",
        "chunk_id":  "2605.06161v1_p19_c1",
    },
    {
        "query":     "What are the two types of triggers described for backdoor attacks in deep reinforcement learning?",
        "target":    "2605.05977v1",
        "chunk_id":  "2605.05977v1_p2_c1",
    },
]


# ---------------------------------------------------------------------------
# Run one /ask call
# ---------------------------------------------------------------------------

def run_one(query: str, condition: str, llm: str, target: str,
            target_chunk: str) -> dict:
    """Hit /ask once, return a flat row of metrics."""
    payload = {"query": query, "condition": condition, "llm": llm}
    t0 = time.time()

    try:
        r = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return {
            "query":     query,
            "condition": condition,
            "llm":       llm,
            "target":    target,
            "target_chunk": target_chunk,
            "error":     str(e),
            "elapsed":   round(time.time() - t0, 2),
        }

    citations       = data.get("citations", [])
    cited_paper_ids = {c.get("paper_id") for c in citations}
    cited_chunk_ids = {c.get("chunk_id") for c in citations}
    correct_paper   = target       in cited_paper_ids
    # NEW for D4: did we cite the EXACT gold chunk, not just the right paper?
    correct_chunk   = target_chunk in cited_chunk_ids

    seed_ids       = data.get("seed_papers", [])
    correct_seeded = target in seed_ids if condition == "graph_guided" else None

    j_before = data.get("judge_before", {}) or {}
    j_after  = data.get("judge_after",  {}) or {}
    improve  = data.get("improvement",  {}) or {}

    return {
        "query":      query,
        "condition":  condition,
        "llm":        llm,
        "target":     target,
        "target_chunk": target_chunk,

        # quality after reflection
        "faithfulness":     j_after.get("faithfulness_score"),
        "faithfulness_lbl": j_after.get("faithfulness_label"),
        "relevance":        j_after.get("relevance_score"),
        "relevance_lbl":    j_after.get("relevance"),

        # before reflection (shows reflection lift)
        "faithfulness_before": j_before.get("faithfulness_score"),
        "claims_fixed":        improve.get("claims_fixed", 0),
        "reflected":           bool(data.get("reflected")),

        # retrieval metrics
        "pool_size":            data.get("pool_size"),
        "papers_in_subgraph":   data.get("papers_in_subgraph"),
        "correct_paper_cited":  correct_paper,
        "correct_chunk_cited":  correct_chunk,                # NEW
        "correct_paper_seeded": correct_seeded,

        # safety
        "safety_clean":  (data.get("safety") or {}).get("is_clean"),
        "out_of_range":  len((data.get("safety") or {}).get("out_of_range", [])),
        "uncited_sents": len((data.get("safety") or {}).get("uncited_sentences", [])),

        # cost
        "elapsed":     data.get("elapsed_seconds"),
        "llm_model":   data.get("llm_model"),

        # raw answer for human inspection
        "answer":      data.get("answer", "")[:500],
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("Checking server health...")
    try:
        h = requests.get(API_URL.replace("/ask", "/health"), timeout=5).json()
    except Exception as e:
        raise SystemExit(f"Server not reachable at {API_URL}. ({e})")

    print(f"  version:          {h.get('version')}")
    print(f"  chunks:           {h.get('chunks')}")
    print(f"  neo4j_ready:      {h.get('neo4j_ready')}")
    print(f"  local_llm_ready:  {h.get('local_llm_ready')}")
    print(f"  local_llm_model:  {h.get('local_llm_model')}")

    if not h.get("local_llm_ready"):
        print("\n⚠️  Local LLM not loaded. Ablation will skip tuned_local runs.")
        llms = ["openrouter"]
    else:
        llms = LLMS

    total = len(GOLD_QUERIES) * len(CONDITIONS) * len(llms)
    print(f"\nRunning {total} /ask calls "
          f"({len(GOLD_QUERIES)} queries × {len(CONDITIONS)} conditions × {len(llms)} llms)")
    print("Allow 30-50 minutes total.\n")

    rows = []
    n = 0
    for gq in GOLD_QUERIES:
        for cond in CONDITIONS:
            for llm in llms:
                n += 1
                print(f"[{n:>2}/{total}] cond={cond:13s} llm={llm:12s} "
                      f"q={gq['query'][:55]!r}")
                row = run_one(gq["query"], cond, llm, gq["target"], gq["chunk_id"])
                rows.append(row)

                # Incremental save — survives crashes
                OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")

                if "error" in row:
                    print(f"        ⚠️  error: {row['error'][:80]}")
                else:
                    print(f"        faith={row['faithfulness']:.2f} "
                          f"rel={row['relevance']:.2f} "
                          f"paper={row['correct_paper_cited']} "
                          f"chunk={row['correct_chunk_cited']} "
                          f"elapsed={row['elapsed']}s")

    print_summary(rows)
    print(f"\n✅ Full results saved to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def safe_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 3) if xs else None

def p95(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    k = max(0, int(round(0.95 * (len(xs) - 1))))
    return round(xs[k], 2)

def print_summary(rows):
    print("\n" + "=" * 95)
    print("D4 ABLATION SUMMARY")
    print("=" * 95)

    groups = {}
    for r in rows:
        key = (r["condition"], r["llm"])
        groups.setdefault(key, []).append(r)

    header = ("Condition       LLM            Faith  Rel   PaperHit  ChunkHit  Refl  ClmFix  p95(s)")
    print(header)
    print("-" * len(header))

    for (cond, llm), gs in sorted(groups.items()):
        faith    = safe_mean([g.get("faithfulness") for g in gs])
        rel      = safe_mean([g.get("relevance")    for g in gs])
        paper    = safe_mean([1 if g.get("correct_paper_cited") else 0 for g in gs])
        chunk    = safe_mean([1 if g.get("correct_chunk_cited") else 0 for g in gs])
        refl     = safe_mean([1 if g.get("reflected")            else 0 for g in gs])
        fixed    = safe_mean([g.get("claims_fixed")  for g in gs])
        p95_lat  = p95([g.get("elapsed") for g in gs])

        print(f"{cond:15s} {llm:14s} "
              f"{faith if faith is not None else '?':<6} "
              f"{rel   if rel   is not None else '?':<5} "
              f"{paper if paper is not None else '?':<9} "
              f"{chunk if chunk is not None else '?':<9} "
              f"{refl  if refl  is not None else '?':<5} "
              f"{fixed if fixed is not None else '?':<7} "
              f"{p95_lat if p95_lat is not None else '?'}")
    print("=" * 95)


if __name__ == "__main__":
    main()
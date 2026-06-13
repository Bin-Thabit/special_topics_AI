"""
scripts/analyze_ablation.py
----------------------------
Re-processes ablation_d4_results.json to add a third, judge-independent
quality metric: `contains_gold_answer` (substring match against expected
answer keywords).

This addresses the judge's documented limitation when verifying numerical
or tabular claims — even when the model correctly says "87.92%", the judge
sometimes can't link that claim to a dense tabular chunk and marks it
NOT_GROUNDED. The substring check is objective ground-truth.

Run AFTER your ablation produces ablation_d4_results.json. Outputs:
  - ablation_d4_analyzed.json   (original rows + contains_gold_answer field)
  - prints comparison table (faithfulness vs paper hit vs gold-answer hit)

Usage:
    python scripts/analyze_ablation.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


INPUT_PATH  = Path("ablation_d4_results.json")
OUTPUT_PATH = Path("ablation_d4_analyzed.json")

# ---------------------------------------------------------------------------
# Gold answer keywords per query
# ---------------------------------------------------------------------------
# For each query, list the distinctive substrings that MUST appear in any
# correct answer. Match is case-insensitive. ANY keyword present = correct.
#
# Keep them tight: a 4-digit number, a domain term, an acronym — anything
# that would only appear if the model genuinely answered the question.
# ---------------------------------------------------------------------------

GOLD_KEYWORDS: dict[str, list[str]] = {

    # 1. Learning rate query
    # Chunk shows "Learning rate 104" — likely rendering of 10⁻⁴.
    # Any of these forms is a correct answer.
    "What learning rate was used in the experiments described in the chunk?": [
        "1e-4", "10⁻⁴", "10−4", "10^-4", "0.0001",
        "10-4",                # ASCII fallback
    ],

    # 2. Q-Patch AUROC
    "What area under the receiver operating characteristic curve (AUROC) did Q-Patch achieve on the audio spoofing detection task?": [
        "0.87",
    ],

    # 3. Implementer share in scenario A
    "What percentage of the total active time does the Implementer role consume in scenario A?": [
        "53.4",
    ],

    # 4. NOVA context ratio
    "What does the context ratio control in the NOVA Context-Conditioned Video Generation algorithm?": [
        "fraction of steps",
        "reference",
        "ground-truth",
        "ground truth",
    ],

    # 5. NFA Gram matrices claim
    "What does the Neural Feature Ansatz (NFA) claim about Gram matrices of weights in a neural network layer?": [
        "AGOP",
        "Average Gradient Outer Product",
        "proportional",
    ],

    # 6. Training hours on RTX Pro 6000
    "How many hours does it take to train Llama3.1-8B-Instruct on GSM8K on a single RTX Pro 6000 GPU?": [
        "20 hour",
        "20-hour",
        "20hr",
        "approximately 20",
    ],

    # 7. Old XDecomposer query (in case it shows up in older runs)
    "What Top-1 accuracy does the Proposed XDecomposer achieve for K=2 in the baseline comparison?": [
        "87.92",
    ],

    # 8. Old z_t interpolation query (in case it shows up in older runs)
    "What effect does interpolating z_t towards zero have on digit identities and motion trajectories?": [
        "8s", "9s",
        "8 and 9", "8's and 9's", "8'\u2019s",
        "morph",
        "preserv",
    ],

    # 9. Old structural alignment query
    "Which metric measures the alignment between the sink key and the principal direction of W_Q extracted via SVD?": [
        "structural alignment",
        "SVD",
    ],

    # 10. Old UniSD datasets query
    "On which datasets does UniSD provide the largest gains according to the component-level results?": [
        "MBPP", "GPQA",
    ],

    # 11. Old PragLocker portability query
    "What relative mean portability does PragLocker achieve compared to no protection?": [
        "0.2",
    ],

    # 12. Old certified lower bounds query
    "What is the goal of certified lower bounds via histogram binning as described in the text?": [
        "high probability",
        "calibration",
        "without any parametric",
        "Markov assumption",
    ],
}


# ---------------------------------------------------------------------------
# Per-row check
# ---------------------------------------------------------------------------

def contains_gold(answer: str, keywords: list[str]) -> bool:
    if not answer or not keywords:
        return False
    a = answer.lower()
    return any(kw.lower() in a for kw in keywords)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Missing {INPUT_PATH} — run the ablation first.")

    rows = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(rows)} rows.\n")

    unmatched_queries = set()
    for r in rows:
        q = r["query"]
        keywords = GOLD_KEYWORDS.get(q)
        if keywords is None:
            unmatched_queries.add(q)
            r["contains_gold_answer"] = None
            r["gold_keywords"]        = []
            continue
        r["gold_keywords"]        = keywords
        r["contains_gold_answer"] = contains_gold(r.get("answer", ""), keywords)

    if unmatched_queries:
        print("⚠️  These queries had no gold keywords defined "
              "(add them to GOLD_KEYWORDS in this script):")
        for q in sorted(unmatched_queries):
            print(f"  - {q[:80]}")
        print()

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"✅ Wrote {OUTPUT_PATH}\n")

    print_three_metric_table(rows)
    print_per_query_breakdown(rows)
    print_judge_vs_gold_disagreements(rows)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def safe_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 3) if xs else None


def print_three_metric_table(rows):
    print("=" * 100)
    print("THREE-METRIC COMPARISON  (per condition × LLM)")
    print("=" * 100)
    print(f"{'Condition':<15} {'LLM':<14} "
          f"{'Faith (judge)':<14} {'Relevance':<11} "
          f"{'PaperHit':<10} {'ChunkHit':<10} {'GoldAns':<10}")
    print("-" * 100)

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["condition"], r["llm"]), []).append(r)

    for (cond, llm), gs in sorted(groups.items()):
        faith    = safe_mean([g.get("faithfulness")  for g in gs])
        rel      = safe_mean([g.get("relevance")     for g in gs])
        paper    = safe_mean([1 if g.get("correct_paper_cited") else 0 for g in gs])
        chunk    = safe_mean([1 if g.get("correct_chunk_cited") else 0 for g in gs])
        gold     = safe_mean(
            [1 if g.get("contains_gold_answer") else 0
             for g in gs if g.get("contains_gold_answer") is not None]
        )
        print(f"{cond:<15} {llm:<14} "
              f"{faith if faith is not None else '?':<14} "
              f"{rel   if rel   is not None else '?':<11} "
              f"{paper if paper is not None else '?':<10} "
              f"{chunk if chunk is not None else '?':<10} "
              f"{gold  if gold  is not None else '?':<10}")
    print("=" * 100)
    print()


def print_per_query_breakdown(rows):
    print("=" * 100)
    print("PER-QUERY BREAKDOWN  (rows per query × 6 = condition×LLM)")
    print("=" * 100)
    print(f"{'Query (truncated)':<50} {'#runs':<6} {'avgFaith':<9} "
          f"{'paperHit':<9} {'chunkHit':<9} {'goldAns':<9}")
    print("-" * 100)

    queries: dict[str, list[dict]] = {}
    for r in rows:
        queries.setdefault(r["query"], []).append(r)

    for q, gs in queries.items():
        faith = safe_mean([g.get("faithfulness") for g in gs])
        paper = safe_mean([1 if g.get("correct_paper_cited") else 0 for g in gs])
        chunk = safe_mean([1 if g.get("correct_chunk_cited") else 0 for g in gs])
        gold  = safe_mean([1 if g.get("contains_gold_answer") else 0
                           for g in gs if g.get("contains_gold_answer") is not None])

        q_short = q[:48] + ".." if len(q) > 50 else q
        print(f"{q_short:<50} {len(gs):<6} "
              f"{faith if faith is not None else '?':<9} "
              f"{paper if paper is not None else '?':<9} "
              f"{chunk if chunk is not None else '?':<9} "
              f"{gold  if gold  is not None else '?':<9}")
    print("=" * 100)
    print()


def print_judge_vs_gold_disagreements(rows):
    """Find rows where the judge says NOT_GROUNDED but the gold check says CORRECT."""
    print("=" * 100)
    print("JUDGE VS GOLD-ANSWER DISAGREEMENTS")
    print("(judge says NOT_GROUNDED but answer contains correct fact)")
    print("=" * 100)

    disagreements = [
        r for r in rows
        if r.get("contains_gold_answer") is True
        and (r.get("faithfulness") or 0) == 0.0
    ]

    if not disagreements:
        print("(none — judge and gold-check agree on every row)")
        print("=" * 100)
        return

    print(f"\nFound {len(disagreements)} rows where judge=0.0 but answer is correct.")
    print(f"This documents the LLM-judge's table/numeric blindness:\n")

    by_query: dict[str, int] = {}
    for r in disagreements:
        by_query[r["query"]] = by_query.get(r["query"], 0) + 1

    for q, n in sorted(by_query.items(), key=lambda kv: -kv[1]):
        print(f"  [{n} runs flagged wrongly]  {q[:80]}")

    sample = disagreements[0]
    print("\nExample disagreement:")
    print(f"  Query:  {sample['query']}")
    print(f"  Cond:   {sample['condition']}  LLM: {sample['llm']}")
    print(f"  Judge:  faithfulness=0.0  ({sample.get('faithfulness_lbl')})")
    print(f"  Gold:   contains_gold_answer=True  "
          f"keywords={sample.get('gold_keywords')}")
    print(f"  Answer: {sample.get('answer', '')[:200]}")
    print("=" * 100)


if __name__ == "__main__":
    main()
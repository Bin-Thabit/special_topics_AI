"""
scripts/build_qa_dataset.py — D4 Phase 1

Builds a raw Q/A dataset from the MongoDB chunk corpus for QLoRA fine-tuning.

Strategy:
  1. Stratified sample ~2 chunks per paper across all papers (min length 400 chars).
  2. For each chunk, prompt a teacher LLM (default: openai/gpt-oss-120b:free)
     to produce ONE factual question + a grounded 1-3 sentence answer ending in [1].
  3. Strict null-rejection: teacher returns null for unsuitable chunks
     (math-only, references, captions, too short, non-extractive).
  4. Save raw output to data/qa_dataset_raw.jsonl for hand-curation (Faisal, Phase 1b).

Output schema (one JSON per line):
  {
    "question":  str,
    "context":   str,    # the chunk text the question was generated from
    "answer":    str,    # ends with [1]
    "paper_id":  str,
    "chunk_id":  str,
    "page":      int
  }

Run:
  python scripts/build_qa_dataset.py
  python scripts/build_qa_dataset.py --n-papers 150 --per-paper 2 --min-chars 400
  python scripts/build_qa_dataset.py --model openai/gpt-oss-120b:free --resume

Env (.env):
  MONGO_URI, MONGO_DB                  — MongoDB connection
  OPENROUTER_API_KEY                   — required
  OPENROUTER_QA_MODEL                  — optional, overrides --model
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

DEFAULT_MODEL    = os.getenv("OPENROUTER_QA_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY")
MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB         = os.getenv("MONGO_DB",  "papers_db")

OUT_PATH         = Path("data/qa_dataset_raw.jsonl")
SEED             = 42

# ──────────────────────────────────────────────────────────────────────────────
# Prompt — kept strict to keep junk out of the dataset
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You generate factual question-answer pairs for fine-tuning a small \
language model on scientific-paper question answering. You ALWAYS return strict JSON \
and nothing else."""

USER_PROMPT_TEMPLATE = """Read the chunk below. Your task:

1. Decide if this chunk is suitable for an extractive Q/A pair. UNSUITABLE chunks:
   - mostly math/equations with little prose
   - reference lists / bibliographies
   - figure or table captions in isolation
   - acknowledgements, author lists, affiliations
   - too short or fragmentary to support a factual question
   - text that depends on figures/tables not shown

2. If UNSUITABLE → return exactly:
   {{"suitable": false}}

3. If SUITABLE → write ONE factual question whose answer is fully and explicitly \
contained in this chunk, then write a 1-3 sentence gold answer ending with the citation [1]. \
The answer MUST be grounded ONLY in this chunk; do not add outside knowledge.

   Return exactly:
   {{"suitable": true, "question": "...", "answer": "... [1]"}}

Rules for good questions:
- Specific and factual (what / how / why / which), not "what does this paper discuss?"
- Answerable in 1-3 sentences from this chunk alone.
- Avoid pronouns without antecedents ("what does it do?" is bad).

CHUNK (paper {paper_id}, page {page}):
\"\"\"
{chunk_text}
\"\"\"

Return ONLY the JSON object. No prose, no markdown, no code fences."""


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────
def stratified_sample(
    chunks_coll,
    n_papers: int,
    per_paper: int,
    min_chars: int,
    seed: int = SEED,
) -> list[dict]:
    """
    Stratified sample: up to `per_paper` chunks from each of `n_papers` papers.

    - Filters chunks shorter than min_chars (drops junk early).
    - Random shuffle within each paper for diversity (intros, methods, results).
    - Deterministic with `seed`.
    """
    rng = random.Random(seed)

    print(f"→ Scanning MongoDB for chunks with len(text) >= {min_chars}...")
    cursor = chunks_coll.find(
        {"text": {"$exists": True}},
        {"chunk_id": 1, "paper_id": 1, "text": 1, "page": 1, "_id": 0},
    )

    by_paper: dict[str, list[dict]] = defaultdict(list)
    for ch in cursor:
        text = (ch.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        if not ch.get("paper_id") or not ch.get("chunk_id"):
            continue
        ch["text"] = text
        by_paper[ch["paper_id"]].append(ch)

    print(f"  found {sum(len(v) for v in by_paper.values())} eligible chunks "
          f"across {len(by_paper)} papers")

    # Pick papers (all of them if fewer than n_papers exist)
    paper_ids = list(by_paper.keys())
    rng.shuffle(paper_ids)
    paper_ids = paper_ids[:n_papers]

    sampled: list[dict] = []
    for pid in paper_ids:
        bucket = by_paper[pid]
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_paper])

    rng.shuffle(sampled)  # final shuffle so progress bar shows mixed papers
    print(f"  sampled {len(sampled)} chunks "
          f"({per_paper}/paper × {len(paper_ids)} papers)")
    return sampled


# ──────────────────────────────────────────────────────────────────────────────
# Teacher LLM call
# ──────────────────────────────────────────────────────────────────────────────
def _strip_code_fences(s: str) -> str:
    """Some models wrap JSON in ```json ... ``` despite instructions."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def generate_qa_for_chunk(
    chunk: dict,
    model: str,
    api_key: str,
    timeout: int = 90,
    max_retries: int = 3,
) -> dict | None:
    """
    Ask the teacher LLM for one Q/A pair for this chunk.
    Returns the example dict, or None if the chunk is unsuitable / call failed.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        paper_id   = chunk["paper_id"],
        page       = chunk.get("page", "?"),
        chunk_text = chunk["text"][:4000],   # safety cap
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens":  400,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers,
                              json=payload, timeout=timeout)
            if r.status_code == 429:
                wait = 2 ** attempt
                tqdm.write(f"  rate-limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            content = _strip_code_fences(content)

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to salvage a JSON object embedded in prose
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if not match:
                    raise
                parsed = json.loads(match.group(0))

            if not parsed.get("suitable"):
                return None

            q = (parsed.get("question") or "").strip()
            a = (parsed.get("answer")   or "").strip()
            if not q or not a:
                return None
            if "[1]" not in a:
                a = a.rstrip(".") + " [1]"

            return {
                "question": q,
                "context":  chunk["text"],
                "answer":   a,
                "paper_id": chunk["paper_id"],
                "chunk_id": chunk["chunk_id"],
                "page":     chunk.get("page", 0),
            }

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    tqdm.write(f"  ✗ failed {chunk['chunk_id']}: {last_err}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-papers",  type=int, default=150,
                    help="Number of distinct papers to draw from")
    ap.add_argument("--per-paper", type=int, default=2,
                    help="Chunks per paper")
    ap.add_argument("--min-chars", type=int, default=400,
                    help="Skip chunks shorter than this many characters")
    ap.add_argument("--model",     type=str, default=DEFAULT_MODEL,
                    help="OpenRouter model id")
    ap.add_argument("--out",       type=Path, default=OUT_PATH,
                    help="Output JSONL path")
    ap.add_argument("--resume",    action="store_true",
                    help="Skip chunk_ids already present in the output file")
    ap.add_argument("--limit",     type=int, default=None,
                    help="Optional: cap total chunks processed (smoke test)")
    args = ap.parse_args()

    if not OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load already-processed chunk_ids
    done: set[str] = set()
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["chunk_id"])
                except Exception:
                    pass
        print(f"→ Resume: {len(done)} chunks already in {args.out}")

    # Mongo
    print(f"→ Connecting to MongoDB at {MONGO_URI} / {MONGO_DB}")
    client = MongoClient(MONGO_URI)
    chunks_coll = client[MONGO_DB]["chunks"]

    sampled = stratified_sample(
        chunks_coll,
        n_papers  = args.n_papers,
        per_paper = args.per_paper,
        min_chars = args.min_chars,
    )

    if args.resume:
        sampled = [c for c in sampled if c["chunk_id"] not in done]
        print(f"  after resume filter: {len(sampled)} chunks to process")

    if args.limit:
        sampled = sampled[: args.limit]
        print(f"  --limit applied: {len(sampled)} chunks")

    # Generate
    print(f"→ Generating Q/A with {args.model}")
    n_kept = 0
    n_null = 0
    with args.out.open("a", encoding="utf-8") as out_f:
        for chunk in tqdm(sampled, desc="Q/A gen"):
            example = generate_qa_for_chunk(chunk, args.model, OPENROUTER_KEY)
            if example is None:
                n_null += 1
                continue
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            out_f.flush()
            n_kept += 1

    print("\n──────────────────────────────────────────────")
    print(f"  kept:        {n_kept}")
    print(f"  unsuitable:  {n_null}")
    print(f"  total tried: {len(sampled)}")
    print(f"  output:      {args.out}")
    print("──────────────────────────────────────────────")
    print("Next: Khalid hand-curates → data/qa_dataset.jsonl → train/eval split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
graphrag/judge.py — LLM-as-Judge with claim-level faithfulness rubric.

FAITHFULNESS (claim-level):
  The model identifies every factual claim in the answer and checks each
  one against the numbered context chunks. A claim is grounded only if the
  specific fact appears in a chunk — not just the general topic.
  faithfulness_score = grounded_claims / total_claims  (computed in Python,
  not by the model, so arithmetic errors don't affect the score).

RELEVANCE (4-point rubric):
  FULLY_RELEVANT     = directly and completely answers the question
  MOSTLY_RELEVANT    = answers the main question, minor aspects missing
  PARTIALLY_RELEVANT = answers only part of the question
  NOT_RELEVANT       = does not answer the question
  Mapped to 1.0 / 0.75 / 0.5 / 0.0 for aggregation.

Why this is better than free-range scoring:
  Free-range 0.0–1.0 has no stable calibration — 0.73 means nothing
  consistent across models or calls. Claim-level rubric forces explicit
  reasoning per claim, produces a verifiable trace, and gives numbers
  computed from auditable yes/no decisions.

The ungrounded_claims list feeds directly into:
  - the before/after safety evidence table (concrete examples of what
    safety mitigations caught or prevented)
  - the reflection loop (D4): pass ungrounded claims back to regenerate

Setup: same OPENROUTER_API_KEY / OPENROUTER_MODEL as answer.py.

Run:
  python -m graphrag.judge
"""
from __future__ import annotations

import json
import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

_BASE_URL      = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
_FALLBACKS = [
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-12b-it:free",
    "qwen/qwen-2.5-7b-instruct:free",
]

_RELEVANCE_SCORES = {
    "FULLY_RELEVANT":     1.00,
    "MOSTLY_RELEVANT":    0.75,
    "PARTIALLY_RELEVANT": 0.50,
    "NOT_RELEVANT":       0.00,
}

_SYSTEM = """\
You are an evaluation judge for a Retrieval-Augmented Generation system.
Evaluate the ANSWER using the two rubrics below.

────────────────────────────────────────
RUBRIC 1 — FAITHFULNESS (claim-level)

Step 1: Identify every FACTUAL CLAIM in the answer.
  A claim IS:  a specific fact, number, finding, causal statement, or attribution.
  NOT a claim: transitional phrases ("therefore", "in summary"), hedges ("may",
               "could"), or re-statements of the question.

Step 2: For each claim decide if it is GROUNDED.
  GROUNDED     = the specific fact is explicitly stated in one of the numbered
                 context chunks. Set source_chunk to the chunk number [N].
  NOT GROUNDED = the fact comes from outside the context (model training knowledge).
                 Set source_chunk to null.

  IMPORTANT: the topic being present in a chunk is NOT enough.
  The specific fact must appear in the chunk text.

────────────────────────────────────────
RUBRIC 2 — RELEVANCE (4-point scale)

  FULLY_RELEVANT     = directly and completely answers the question
  MOSTLY_RELEVANT    = answers the main question, minor aspects missing
  PARTIALLY_RELEVANT = answers only part of the question
  NOT_RELEVANT       = does not answer the question

────────────────────────────────────────
Respond ONLY with valid JSON — no markdown fences, no text outside the JSON:
{
  "claims": [
    {"claim": "<exact claim text>", "grounded": true/false, "source_chunk": <N or null>}
  ],
  "relevance": "FULLY_RELEVANT | MOSTLY_RELEVANT | PARTIALLY_RELEVANT | NOT_RELEVANT",
  "relevance_reasoning": "<one sentence explaining the relevance score>"
}\
"""


def _fmt_authors(authors: list | None) -> str:
    if not authors:
        return "Unknown"
    return authors[0] + (" et al." if len(authors) > 1 else "")


def _build_context_block(chunks: list[dict], max_chars: int = 400) -> str:
    lines = ["CONTEXT CHUNKS", "─" * 50]
    for i, c in enumerate(chunks, 1):
        title   = c.get("title") or c.get("paper_id", "?")
        authors = _fmt_authors(c.get("authors"))
        year    = c.get("year", "n.d.")
        page    = c.get("page_num", "?")
        text    = (c.get("text") or "")[:max_chars]
        lines.append(f'[{i}] "{title}" | {authors} | {year} | p.{page}\n    {text}\n')
    return "\n".join(lines)


def _faithfulness_label(score: float | None) -> str:
    if score is None:
        return "NO_CLAIMS_FOUND"
    if score == 1.0:
        return "FULLY_GROUNDED"
    if score >= 0.75:
        return "MOSTLY_GROUNDED"
    if score >= 0.50:
        return "PARTIALLY_GROUNDED"
    return "NOT_GROUNDED"


def _parse_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON; fallback to empty structure on error."""
    text = re.sub(r"^```(?:json)?\n?", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("judge: JSON parse failed — returning raw for inspection")
        return {"_raw": raw, "_parse_error": True}


def judge_answer(
    query: str,
    answer: str,
    context_chunks: list[dict],
    model: str | None = None,
    openrouter_api_key: str | None = None,
) -> dict:
    """
    Evaluate an answer with the claim-level faithfulness rubric + relevance rubric.

    Args
      query          : the original question.
      answer         : answer produced by generate_answer().
      context_chunks : the ranked chunks passed to generate_answer() — used as
                       the ground-truth context for grounding checks.

    Returns
      faithfulness_score  : float  grounded_claims / total_claims  (0.0-1.0)
      faithfulness_label  : str    FULLY / MOSTLY / PARTIALLY / NOT _GROUNDED
      grounded_count      : int
      total_claims        : int
      claim_breakdown     : list[{claim, grounded, source_chunk}]
      ungrounded_claims   : list[str]  the non-grounded claims (for reflection/report)
      relevance           : str    categorical label
      relevance_score     : float  0.0 / 0.50 / 0.75 / 1.0
      relevance_reasoning : str
      model               : str
      error               : str | None
    """
    from openai import OpenAI, RateLimitError

    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set — add it to your .env file.")

    primary = model or _DEFAULT_MODEL
    queue   = [primary] + [m for m in _FALLBACKS if m != primary]

    context_block = _build_context_block(context_chunks)
    user_msg = (
        f"{context_block}\n\n"
        f"{'─' * 50}\n"
        f"QUESTION\n{query}\n\n"
        f"{'─' * 50}\n"
        f"ANSWER TO EVALUATE\n{answer}\n\n"
        f"Apply both rubrics to the answer above."
    )

    client = OpenAI(
        base_url=_BASE_URL,
        api_key=api_key,
        max_retries=0,
        default_headers={
            "HTTP-Referer": "https://github.com/special_topics_ai",
            "X-Title": "PDF-Papers AI Agent - D3 Judge",
        },
    )

    raw = used_model = None
    for attempt in queue:
        log.info("Judge  model=%s", attempt)
        try:
            resp = client.chat.completions.create(
                model=attempt,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw        = resp.choices[0].message.content
            used_model = attempt
            break
        except RateLimitError:
            log.warning("Judge 429 on %s — trying fallback", attempt)
        except Exception as e:
            raise RuntimeError(f"Judge API call failed on {attempt}: {e}") from e

    if raw is None:
        raise RuntimeError("All judge models rate-limited. Wait and retry.")

    parsed = _parse_response(raw)

    if parsed.get("_parse_error"):
        return {
            "faithfulness_score":  None,
            "faithfulness_label":  "PARSE_ERROR",
            "grounded_count":      None,
            "total_claims":        None,
            "claim_breakdown":     [],
            "ungrounded_claims":   [],
            "relevance":           None,
            "relevance_score":     None,
            "relevance_reasoning": "",
            "model":               used_model,
            "error":               f"JSON parse failed. Raw: {raw[:200]}",
        }

    claims          = parsed.get("claims", [])
    total           = len(claims)
    grounded_count  = sum(1 for c in claims if c.get("grounded"))
    faith_score     = round(grounded_count / total, 2) if total > 0 else None
    ungrounded      = [c["claim"] for c in claims if not c.get("grounded")]

    relevance_label = parsed.get("relevance", "PARTIALLY_RELEVANT")
    if relevance_label not in _RELEVANCE_SCORES:
        relevance_label = "PARTIALLY_RELEVANT"
    relevance_score = _RELEVANCE_SCORES[relevance_label]

    return {
        "faithfulness_score":  faith_score,
        "faithfulness_label":  _faithfulness_label(faith_score),
        "grounded_count":      grounded_count,
        "total_claims":        total,
        "claim_breakdown":     claims,
        "ungrounded_claims":   ungrounded,
        "relevance":           relevance_label,
        "relevance_score":     relevance_score,
        "relevance_reasoning": parsed.get("relevance_reasoning", ""),
        "model":               used_model,
        "error":               None,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    QUERY = "How does reinforcement learning handle long-horizon planning?"

    CHUNKS = [
        {
            "chunk_id": "c1", "paper_id": "2605.06078v1",
            "title": "Milestone-Guided Policy Learning for Long-Horizon Language Agents",
            "authors": ["Zhang", "Wang"], "year": 2025, "page_num": 3,
            "text": (
                "Over 73% of training samples receive no learning signal in standard "
                "RLHF on long-horizon tasks, and successful samples suffer from "
                "contradictory credit assignment that corrupts gradients."
            ),
        },
        {
            "chunk_id": "c2", "paper_id": "2605.06078v1",
            "title": "Milestone-Guided Policy Learning for Long-Horizon Language Agents",
            "authors": ["Zhang", "Wang"], "year": 2025, "page_num": 9,
            "text": (
                "Milestone-anchored policy learning inserts intermediate reward "
                "checkpoints along the trajectory, isolating local action quality "
                "from downstream variance and improving sample utilisation."
            ),
        },
        {
            "chunk_id": "c3", "paper_id": "2605.06094v1",
            "title": "VISD: Enhancing Video Reasoning via Structured Self-Distillation",
            "authors": ["Liu"], "year": 2025, "page_num": 6,
            "text": (
                "A curriculum strategy gradually transitions from structured "
                "self-distillation to pure RL, helping the model acquire detailed "
                "reasoning patterns before relying on sparse environmental rewards."
            ),
        },
    ]

    GOOD_ANSWER = (
        "RL struggles with long-horizon planning because over 73% of training "
        "samples receive no learning signal, making credit assignment extremely "
        "difficult [1]. Milestone-anchored methods address this by inserting "
        "intermediate checkpoints along the trajectory, isolating local action "
        "quality from downstream variance [2]. Curriculum strategies that shift "
        "from self-distillation to pure RL also help the model build reasoning "
        "patterns before relying on sparse rewards [3]."
    )

    BAD_ANSWER = (
        "Reinforcement learning handles long-horizon planning through hierarchical "
        "abstractions and option frameworks, as proposed by Sutton et al. in 1999. "
        "Modern approaches use transformer-based world models trained on internet "
        "scale data to predict future states over thousands of time steps."
    )

    SEP = "=" * 62
    for label, ans in [("A  grounded answer", GOOD_ANSWER),
                       ("B  hallucinated answer", BAD_ANSWER)]:
        result = judge_answer(QUERY, ans, CHUNKS)
        print(f"\n{SEP}")
        print(f"Scenario {label}")
        print(f"{'─' * 62}")
        print(f"Claims ({result['total_claims']} found):")
        for c in result["claim_breakdown"]:
            icon = "✓" if c["grounded"] else "✗"
            src  = f"chunk [{c['source_chunk']}]" if c["source_chunk"] else "NOT IN CONTEXT"
            print(f"  {icon}  {c['claim'][:70]}")
            print(f"      → {src}")
        print(f"{'─' * 62}")
        print(f"Faithfulness : {result['grounded_count']}/{result['total_claims']} "
              f"({result['faithfulness_score']}) — {result['faithfulness_label']}")
        print(f"Relevance    : {result['relevance']} ({result['relevance_score']})")
        print(f"             : {result['relevance_reasoning']}")
        print(f"Model        : {result['model']}")
        if result["ungrounded_claims"]:
            print(f"Ungrounded   :")
            for u in result["ungrounded_claims"]:
                print(f"  - {u[:80]}")
    print(f"\n{SEP}")
    print("Paste the claim breakdown tables into your D3 before/after evidence section.")
    print(SEP)
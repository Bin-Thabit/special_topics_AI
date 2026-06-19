"""
graphrag/reflect.py — Reflection agent (critic-revise pattern).

Flow
----
  generate_answer()  →  initial answer
  judge_answer()     →  critique  (ungrounded_claims, faithfulness score)
  reflect_answer()   →  revised answer  (fixes the critique, re-judges)
  compare            →  before/after faithfulness + relevance delta

The reflection agent receives:
  - the numbered context chunks  (the ONLY allowed sources)
  - the initial answer
  - the judge's ungrounded_claims list  (exact claims to fix)

It rewrites the answer by removing or replacing every ungrounded claim
with content actually present in the context. Claims already grounded are
kept unchanged. The revised answer keeps [N] inline citations throughout.

After revision, judge_answer() runs again so the caller gets a side-by-side
before/after comparison built into the return dict — ready for your D3
evidence table and the /ask response payload.

One reflection pass is used (not iterative). If the initial answer is already
FULLY_GROUNDED, reflection is skipped and reflection_triggered=False.

Setup: same OPENROUTER_API_KEY / OPENROUTER_MODEL as answer.py and judge.py.

Run:
  python -m graphrag.reflect
"""
from __future__ import annotations

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

_SYSTEM = """\
You are a revision agent improving a Retrieval-Augmented Generation answer.

You will receive:
  - CONTEXT CHUNKS [1]..[N]  — the only allowed sources
  - ORIGINAL ANSWER          — may contain unsupported claims
  - FAITHFULNESS CRITIQUE    — the exact claims that are NOT in the context

Your revision rules (follow strictly):
  1. For each claim listed in the critique:
       - If the context contains a related grounded fact, replace the claim
         with that fact and cite the chunk [N].
       - If no related fact exists in the context, remove the claim entirely.
  2. Every remaining factual claim MUST have an inline citation [N].
  3. Keep all claims NOT in the critique unchanged (they are already grounded).
  4. Do NOT add any information from outside the provided context chunks.
  5. If removing ungrounded claims leaves the answer unable to fully answer
     the question, say so explicitly at the end.
  6. Match the original answer's style. Write one coherent paragraph or
     short paragraph set — not a bullet list.\
"""


def _fmt_authors(authors: list | None) -> str:
    if not authors:
        return "Unknown"
    return authors[0] + (" et al." if len(authors) > 1 else "")


def _build_context_block(chunks: list[dict], max_chars: int = 500) -> str:
    lines = ["CONTEXT CHUNKS", "─" * 50]
    for i, c in enumerate(chunks, 1):
        title   = c.get("title") or c.get("paper_id", "?")
        authors = _fmt_authors(c.get("authors"))
        year    = c.get("year", "n.d.")
        page    = c.get("page_num", "?")
        text    = (c.get("text") or "")[:max_chars]
        lines.append(f'[{i}] "{title}" | {authors} | {year} | p.{page}\n    {text}\n')
    return "\n".join(lines)


def _extract_citations(answer: str, numbered_chunks: list[dict]) -> list[dict]:
    """Build citation list from inline [N] references in the revised answer."""
    cited_nums = sorted({
        int(n)
        for m in re.findall(r"\[([\d,\s]+)\]", answer)
        for n in re.findall(r"\d+", m)
    })
    num_to_chunk = {c["number"]: c for c in numbered_chunks}
    return [
        {
            "number":   n,
            "chunk_id": num_to_chunk[n]["chunk_id"],
            "paper_id": num_to_chunk[n]["paper_id"],
            "title":    num_to_chunk[n].get("title"),
            "authors":  num_to_chunk[n].get("authors", []),
            "year":     num_to_chunk[n].get("year"),
            "page_num": num_to_chunk[n].get("page_num"),
        }
        for n in cited_nums if n in num_to_chunk
    ]


def _compare(before: dict, after: dict) -> dict:
    """Compute the before/after delta for the D3 evidence table."""
    b_faith = before.get("faithfulness_score") or 0.0
    a_faith = after.get("faithfulness_score")  or 0.0
    b_rel   = before.get("relevance_score")    or 0.0
    a_rel   = after.get("relevance_score")     or 0.0
    b_ung   = before.get("ungrounded_claims",  [])
    a_ung   = after.get("ungrounded_claims",   [])
    return {
        "faithfulness_before":       b_faith,
        "faithfulness_after":        a_faith,
        "faithfulness_delta":        round(a_faith - b_faith, 2),
        "faithfulness_label_before": before.get("faithfulness_label"),
        "faithfulness_label_after":  after.get("faithfulness_label"),
        "relevance_before":          b_rel,
        "relevance_after":           a_rel,
        "relevance_delta":           round(a_rel - b_rel, 2),
        "claims_fixed":              len(b_ung) - len(a_ung),
        "ungrounded_before":         b_ung,
        "ungrounded_after":          a_ung,
    }


def reflect_answer(
    query: str,
    ranked_chunks: list[dict],
    initial_result: dict,
    judge_result: dict,
    model: str | None = None,
    re_judge: bool = True,
    openrouter_api_key: str | None = None,
) -> dict:
    """
    Revise the initial answer to fix the faithfulness issues found by the judge.

    Args
      query          : the original question.
      ranked_chunks  : the same chunks passed to generate_answer().
      initial_result : output of generate_answer()  — needs 'answer' key.
      judge_result   : output of judge_answer()     — needs 'ungrounded_claims',
                       'faithfulness_score', 'claim_breakdown'.
      re_judge       : if True (default), run judge_answer() on the revised answer
                       so the caller gets a before/after comparison.

    Returns
      {
        reflection_triggered : bool
        revised_answer       : str   (= initial answer if not triggered)
        revised_citations    : list
        initial_judge        : dict  (the judge_result passed in)
        revised_judge        : dict  (new judge scores; None if re_judge=False)
        improvement          : dict  (faithfulness_delta, claims_fixed, ...)
        model                : str
      }
    """
    from openai import OpenAI, RateLimitError

    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set — add it to your .env file.")

    initial_answer   = initial_result.get("answer", "")
    ungrounded       = judge_result.get("ungrounded_claims", [])
    faith_score      = judge_result.get("faithfulness_score")

    # ── skip reflection if already fully grounded ─────────────────────────
    if not ungrounded and faith_score == 1.0:
        log.info("reflect: answer already FULLY_GROUNDED — skipping revision")
        return {
            "reflection_triggered": False,
            "revised_answer":       initial_answer,
            "revised_citations":    initial_result.get("citations", []),
            "initial_judge":        judge_result,
            "revised_judge":        judge_result,
            "improvement":          _compare(judge_result, judge_result),
            "model":                None,
        }

    # ── build the reflection prompt ────────────────────────────────────────
    # Number the chunks the same way generate_answer does
    max_chunks = initial_result.get("chunks_used", 6)
    pool       = ranked_chunks[:max_chunks]

    context_block = _build_context_block(pool)

    critique_lines = "\n".join(f'  {i+1}. "{c}"' for i, c in enumerate(ungrounded))
    critique_block = (
        f"The following {len(ungrounded)} claim(s) are NOT supported by the context:\n"
        f"{critique_lines}"
        if ungrounded else
        "No specific ungrounded claims were listed, but faithfulness is below 1.0. "
        "Ensure every factual claim has a verifiable citation [N]."
    )

    user_msg = (
        f"{context_block}\n\n"
        f"{'─' * 50}\n"
        f"ORIGINAL QUESTION\n{query}\n\n"
        f"{'─' * 50}\n"
        f"ORIGINAL ANSWER\n{initial_answer}\n\n"
        f"{'─' * 50}\n"
        f"FAITHFULNESS CRITIQUE\n{critique_block}\n\n"
        f"Rewrite the answer above to fix the critique. "
        f"Use ONLY the context chunks as sources."
    )

    # ── call the reflection LLM ────────────────────────────────────────────
    primary = model or _DEFAULT_MODEL
    queue   = [primary] + [m for m in _FALLBACKS if m != primary]

    client = OpenAI(
        base_url=_BASE_URL,
        api_key=api_key,
        max_retries=0,
        default_headers={
            "HTTP-Referer": "https://github.com/special_topics_ai",
            "X-Title": "PDF-Papers AI Agent - D3 Reflect",
        },
    )

    revised_text = used_model = None
    for attempt in queue:
        log.info("Reflect  model=%s  ungrounded=%d", attempt, len(ungrounded))
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
            revised_text = resp.choices[0].message.content.strip()
            used_model   = attempt
            break
        except RateLimitError:
            log.warning("Reflect 429 on %s — trying fallback", attempt)
        except Exception as e:
            raise RuntimeError(f"Reflect API call failed on {attempt}: {e}") from e

    if revised_text is None:
        raise RuntimeError("All reflection models rate-limited. Wait and retry.")

    # ── extract citations from revised answer ──────────────────────────────
    numbered = [{"number": i + 1, **c} for i, c in enumerate(pool)]
    revised_citations = _extract_citations(revised_text, numbered)

    # ── re-judge the revised answer ────────────────────────────────────────
    revised_judge = None
    if re_judge:
        from agents.judge import judge_answer
        log.info("Reflect  re-judging revised answer...")
        revised_judge = judge_answer(
            query=query,
            answer=revised_text,
            context_chunks=pool,
            model=used_model,
            openrouter_api_key=api_key,
        )

    improvement = _compare(judge_result, revised_judge or judge_result)

    return {
        "reflection_triggered": True,
        "revised_answer":       revised_text,
        "revised_citations":    revised_citations,
        "initial_judge":        judge_result,
        "revised_judge":        revised_judge,
        "improvement":          improvement,
        "model":                used_model,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from agents.judge import judge_answer

    QUERY = "How does reinforcement learning handle long-horizon planning?"

    CHUNKS = [
        {
            "chunk_id": "c1", "paper_id": "2605.06078v1", "number": 1,
            "title": "Milestone-Guided Policy Learning for Long-Horizon Language Agents",
            "authors": ["Zhang", "Wang"], "year": 2025, "page_num": 3,
            "text": (
                "Over 73% of training samples receive no learning signal in standard "
                "RLHF on long-horizon tasks, and successful samples suffer from "
                "contradictory credit assignment that corrupts gradients."
            ),
        },
        {
            "chunk_id": "c2", "paper_id": "2605.06078v1", "number": 2,
            "title": "Milestone-Guided Policy Learning for Long-Horizon Language Agents",
            "authors": ["Zhang", "Wang"], "year": 2025, "page_num": 9,
            "text": (
                "Milestone-anchored policy learning inserts intermediate reward "
                "checkpoints along the trajectory, isolating local action quality "
                "from downstream variance and improving sample utilisation."
            ),
        },
        {
            "chunk_id": "c3", "paper_id": "2605.06094v1", "number": 3,
            "title": "VISD: Enhancing Video Reasoning via Structured Self-Distillation",
            "authors": ["Liu"], "year": 2025, "page_num": 6,
            "text": (
                "A curriculum strategy gradually transitions from structured "
                "self-distillation to pure RL, helping the model acquire detailed "
                "reasoning patterns before relying on sparse environmental rewards."
            ),
        },
    ]

    # Start with the hallucinated answer — worst case for the judge
    BAD_ANSWER = (
        "Reinforcement learning handles long-horizon planning through hierarchical "
        "abstractions and option frameworks, as proposed by Sutton et al. in 1999. "
        "Modern approaches use transformer-based world models trained on internet "
        "scale data to predict future states over thousands of time steps."
    )

    initial_result = {
        "answer": BAD_ANSWER,
        "citations": [],
        "chunks_used": 3,
    }

    SEP = "=" * 62
    print(f"\n{SEP}")
    print("STEP 1  Judge initial (hallucinated) answer")
    print(SEP)
    initial_judge = judge_answer(QUERY, BAD_ANSWER, CHUNKS)
    print(f"Faithfulness : {initial_judge['grounded_count']}/{initial_judge['total_claims']} "
          f"— {initial_judge['faithfulness_label']}")
    print(f"Relevance    : {initial_judge['relevance']} ({initial_judge['relevance_score']})")
    print(f"Ungrounded   : {len(initial_judge['ungrounded_claims'])} claim(s)")
    for u in initial_judge["ungrounded_claims"]:
        print(f"  - {u[:80]}")

    print(f"\n{SEP}")
    print("STEP 2  Reflect — revise the answer using the critique")
    print(SEP)
    result = reflect_answer(QUERY, CHUNKS, initial_result, initial_judge)

    print(f"\nRevised answer:\n{result['revised_answer']}\n")

    print(f"\n{SEP}")
    print("STEP 3  Before vs After comparison")
    print(SEP)
    imp = result["improvement"]
    rj  = result["revised_judge"] or {}
    print(f"{'Metric':<28} {'Before':>12} {'After':>12} {'Delta':>8}")
    print("─" * 62)
    print(f"{'Faithfulness score':<28} "
          f"{imp['faithfulness_before']:>12.2f} "
          f"{imp['faithfulness_after']:>12.2f} "
          f"{imp['faithfulness_delta']:>+8.2f}")
    print(f"{'Faithfulness label':<28} "
          f"{imp['faithfulness_label_before']:>12} "
          f"{imp['faithfulness_label_after']:>12}")
    print(f"{'Relevance score':<28} "
          f"{imp['relevance_before']:>12.2f} "
          f"{imp['relevance_after']:>12.2f} "
          f"{imp['relevance_delta']:>+8.2f}")
    print(f"{'Claims fixed':<28} {imp['claims_fixed']:>32}")
    if rj.get("ungrounded_claims"):
        print(f"\nStill ungrounded after reflection:")
        for u in rj["ungrounded_claims"]:
            print(f"  - {u[:80]}")
    else:
        print("\nAll claims grounded after reflection.")
    print(SEP)
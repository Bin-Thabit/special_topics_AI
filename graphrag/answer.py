"""
graphrag/answer.py — GraphRAG Step 4: grounded answer with inline citations.

Takes the top-k ranked chunks from Step 3, builds a numbered context block,
calls an LLM via OpenRouter, and returns a structured answer where every
factual claim is grounded with an inline [N] citation and a source list that
maps N → paper title + page number.

Setup:
    pip install openai
    Add to .env:  OPENROUTER_API_KEY=sk-or-...
                  OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free  (optional)

Run:
    python -m graphrag.answer
"""
from __future__ import annotations

import os
import re
import logging

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_BASE_URL      = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-oss-120b:free",          # OpenAI open-weight 120B, free as of June 2026
)
# Tried in order when the primary model returns 429 or 404.
# Free-tier models rotate — check openrouter.ai/models?q=free for the current list.
# `openrouter/free` is the final safety net: a meta-router that always picks
# from whatever free models are available, so it never 404s.
_DEFAULT_FALLBACKS = [
    "openrouter/owl-alpha",                      # OpenRouter's own free flagship
    "nvidia/nemotron-3-super-120b-a12b:free",    # NVIDIA 120B MoE, 1M ctx
    "openrouter/free",                           # meta-router — always works
]

_SYSTEM = """\
You are a precise research assistant answering questions about AI research papers.

Rules (follow strictly):
1. Answer ONLY using the numbered context chunks provided. No outside knowledge.
2. After every factual claim add an inline citation like [1] or [2, 3].
3. If the context does not contain enough information, say exactly:
   "The provided context does not contain enough information to answer this question."
4. Be concise — one focused paragraph per point is enough.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_authors(authors: list | None) -> str:
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} & {authors[1]}"
    return f"{authors[0]} et al."


def build_context(
    chunks: list[dict],
    max_chars_per_chunk: int = 600,
) -> tuple[str, list[dict]]:
    """
    Number chunks [1..N] and format them as a context block for the prompt.
    Returns (context_str, numbered_chunks) — numbered_chunks[i]["number"] == i+1.
    Each chunk must have at minimum: text, paper_id.
    Denormalized fields (title, authors, year, page_num) are used if present.
    """
    lines = ["CONTEXT\n" + "─" * 60]
    numbered = []
    for i, c in enumerate(chunks, 1):
        title   = c.get("title") or c.get("paper_id", "Unknown")
        authors = _fmt_authors(c.get("authors") or [])
        year    = c.get("year") or "n.d."
        page    = c.get("page_num", "?")
        text    = (c.get("text") or "")[:max_chars_per_chunk]
        lines.append(
            f'[{i}] "{title}"\n'
            f"    {authors} | {year} | page {page}\n\n"
            f"    {text}\n"
        )
        numbered.append({**c, "number": i})
    return "\n".join(lines), numbered


def _cited_numbers(answer: str) -> list[int]:
    """Extract ALL citation numbers from [N], [N, M] and 【N】 style references."""
    blocks = re.findall(r"(?:\[|【)([\d,\s]+)(?:\]|】)", answer)
    return sorted({int(n) for block in blocks for n in re.findall(r"\d+", block)})


# ---------------------------------------------------------------------------
# Step 4 core
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    ranked_chunks: list[dict],
    model: str | None = None,
    model_fallbacks: list[str] | None = None,
    max_context_chunks: int = 6,
    max_chars_per_chunk: int = 600,
    openrouter_api_key: str | None = None,
) -> dict:
    """
    Call an LLM via OpenRouter and return a grounded answer with citations.

    Args
      query            : the user's original question.
      ranked_chunks    : top-k output from rank_pool()["results"] — needs
                         text, paper_id, page_num, title, authors, year.
      model            : OpenRouter model string (overrides env / default).
      model_fallbacks  : tried in order on 429; defaults to _DEFAULT_FALLBACKS.
      max_context_chunks: how many chunks to include (6 keeps well under 8k).
      openrouter_api_key: falls back to OPENROUTER_API_KEY env var.

    Returns
      {answer, citations, sources_used, model, chunks_used, query}

      answer        : answer text with inline [N] citations
      citations     : list of {number, paper_id, title, authors, year, page_num,
                      chunk_id} — only for [N]s that actually appear in the answer
      sources_used  : sorted list of cited [N] ints
      chunks_used   : how many chunks were passed to the model
    """
    from openai import OpenAI, NotFoundError, RateLimitError

    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file: OPENROUTER_API_KEY=sk-or-..."
        )

    primary  = model or _DEFAULT_MODEL
    fallbacks = list(model_fallbacks if model_fallbacks is not None else _DEFAULT_FALLBACKS)
    # full queue: primary first, then fallbacks (skip primary if already listed)
    queue = [primary] + [m for m in fallbacks if m != primary]

    pool = ranked_chunks[:max_context_chunks]
    if not pool:
        return {
            "answer": "No context chunks were provided.",
            "citations": [], "sources_used": [],
            "model": primary, "chunks_used": 0, "query": query,
        }

    context_str, numbered = build_context(pool, max_chars_per_chunk)
    user_msg = (
        f"{context_str}\n\n"
        f"{'─' * 60}\n"
        f"QUESTION\n{query}\n\n"
        f"Answer using ONLY the context above. "
        f"Cite every factual claim inline as [N]."
    )

    client = OpenAI(
        base_url=_BASE_URL,
        api_key=api_key,
        max_retries=0,                  # we handle retries ourselves via the queue
        default_headers={
            "HTTP-Referer": "https://github.com/special_topics_ai",
            "X-Title":      "PDF-Papers AI Agent - D3",
        },
    )

    response = None
    used_model = primary
    for attempt_model in queue:
        log.info("OpenRouter  model=%s  chunks=%d", attempt_model, len(pool))
        try:
            response = client.chat.completions.create(
                model=attempt_model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            used_model = attempt_model
            break                       # success — stop trying fallbacks
        except (RateLimitError, NotFoundError):
            log.warning("429/404 on %s — trying next fallback", attempt_model)
        except Exception as e:
            raise RuntimeError(f"OpenRouter API call failed on {attempt_model}: {e}") from e

    if response is None:
        raise RuntimeError(
            "All models rate-limited. Wait a minute and retry, or add a paid key at "
            "https://openrouter.ai/settings/integrations"
        )

    answer_text = response.choices[0].message.content.strip()

    # Some models (e.g. OpenAI GPT) emit 【1†L1-L4】 instead of [1].
    # Normalise to standard [N] before extracting citations.
    answer_text = re.sub(r"【(\d+)[^】]*】", r"[\1]", answer_text)

    # Build citation list from the [N]s that actually appear in the answer
    cited    = _cited_numbers(answer_text)
    num_to_c = {c["number"]: c for c in numbered}
    citations = [
        {
            "number":   n,
            "chunk_id": num_to_c[n]["chunk_id"],
            "paper_id": num_to_c[n]["paper_id"],
            "title":    num_to_c[n].get("title"),
            "authors":  num_to_c[n].get("authors", []),
            "year":     num_to_c[n].get("year"),
            "page_num": num_to_c[n].get("page_num"),
        }
        for n in cited if n in num_to_c
    ]

    return {
        "answer":       answer_text,
        "citations":    citations,
        "sources_used": cited,
        "model":        used_model,
        "chunks_used":  len(pool),
        "query":        query,
    }


# ---------------------------------------------------------------------------
# Smoke test — full pipeline Steps 1 → 4
# Requires: all services up + OPENROUTER_API_KEY in .env
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    from stores.neo4j_store  import Neo4jStore
    from stores.mongo_store  import get_mongo_client, get_mongo_db
    from stores.qdrant_store import get_qdrant_client
    from ingestion.embedder  import load_model
    from graphrag.expand     import expand_to_chunks
    from graphrag.rank       import (
        embed_query            as _embed_q,
        pool_embeddings_from_qdrant,
        rank_pool,
    )

    QUERY = "How does reinforcement learning handle long-horizon planning?"

    # --- Step 1: subgraph ---------------------------------------------------
    with Neo4jStore() as store:
        seed     = store.run_query(
            "MATCH (p:Paper) RETURN p.paper_id AS id LIMIT 1"
        )[0]["id"]
        subgraph = store.select_subgraph(seed_paper_ids=[seed], max_papers=10)
    print(f"Step 1 : {len(subgraph)} papers")

    # --- Step 2: expand to chunks -------------------------------------------
    db   = get_mongo_db(get_mongo_client())
    pool = expand_to_chunks(db, subgraph)
    print(f"Step 2 : {len(pool)} candidate chunks")

    # --- Step 3: hybrid rank ------------------------------------------------
    model  = load_model()
    qdrant = get_qdrant_client()
    aligned_pool, pool_emb = pool_embeddings_from_qdrant(pool, qdrant)
    qv     = _embed_q(model, QUERY)
    ranked = rank_pool(
        QUERY, aligned_pool, model, pool_emb,
        alpha=0.2149, k=8, query_vector=qv,
    )
    print(f"Step 3 : top {len(ranked)} ranked chunks")

    # --- Step 4: grounded answer --------------------------------------------
    result = generate_answer(QUERY, ranked)

    print(f"\n{'═' * 60}")
    print(f"Q: {result['query']}")
    print(f"{'─' * 60}")
    print(f"A: {result['answer']}")
    print(f"\nCITATIONS ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  [{c['number']}] {c['title']}  |  p. {c['page_num']}  |  {c['paper_id']}")
    print(f"\nModel : {result['model']}")
    print(f"{'═' * 60}")
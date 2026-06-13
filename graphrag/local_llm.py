"""
graphrag/local_llm.py

Drop-in replacement for graphrag.answer.generate_answer() that calls a
locally-running Ollama instance with our QLoRA-tuned Qwen2.5-1.5B.

Same return shape as generate_answer(), so /ask only swaps one line.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests


# ---- Config (override via env vars in .env) ----------------------------------

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "pdfpapers-tuned")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# Must match the system prompt used during QLoRA training
SYSTEM_PROMPT = (
    "You answer questions from scientific papers. "
    "Use ONLY the provided context. Cite sources inline as [1], [2], etc. "
    "Be concise (1-3 sentences). If the answer is not in the context, say so."
)


# ---- Public API ---------------------------------------------------------------

def load_local_llm() -> dict[str, Any]:
    """
    Smoke-test the Ollama connection at startup. Returns a small state dict
    that /ask passes back to generate_answer_local().

    Raises RuntimeError if Ollama isn't reachable or the model isn't loaded.
    """
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            f"Is `ollama serve` running? Original error: {e}"
        )

    models = [m["name"] for m in r.json().get("models", [])]
    # Ollama tags can have ':latest' suffix
    matches = [m for m in models if m == OLLAMA_MODEL or m.startswith(f"{OLLAMA_MODEL}:")]
    if not matches:
        raise RuntimeError(
            f"Model '{OLLAMA_MODEL}' not loaded in Ollama. "
            f"Available: {models}. "
            f"Run: `ollama create {OLLAMA_MODEL} -f models/Modelfile`"
        )

    return {
        "host":  OLLAMA_HOST,
        "model": matches[0],
        "ready": True,
    }


def generate_answer_local(
    query: str,
    ranked: list[dict],
    state: dict[str, Any] | None = None,
    max_chunks: int = 8,
    max_new_tokens: int = 400,
) -> dict[str, Any]:
    """
    Drop-in replacement for graphrag.answer.generate_answer().

    Args:
        query:   user's question
        ranked:  list of chunks from rank_pool(), each with keys
                 'chunk_id', 'text', 'paper_id', 'page' (or 'page_start')
        state:   dict from load_local_llm() (ignored fields tolerated)
        max_chunks: how many top chunks to give the model

    Returns:
        {
          "answer":      str,
          "citations":   list[dict],   # [{n, chunk_id, paper_id, page}, ...]
          "chunks_used": int,
          "model":       str,
        }
    """
    if state is None:
        state = {"host": OLLAMA_HOST, "model": OLLAMA_MODEL}

    top_chunks = ranked[:max_chunks]

    # Build numbered context and citation map
    context_lines = []
    citations = []
    for i, ch in enumerate(top_chunks, start=1):
        text = ch.get("text", "").strip()
        context_lines.append(f"[{i}] {text}")
        citations.append({
            "n":        i,
            "chunk_id": ch.get("chunk_id"),
            "paper_id": ch.get("paper_id"),
            "page":     ch.get("page") or ch.get("page_start"),
        })

    user_msg = (
        f"Question: {query}\n\n"
        f"Context:\n" + "\n\n".join(context_lines)
    )

    payload = {
        "model": state["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "options": {
            "temperature":  0.2,
            "top_p":        0.9,
            "num_predict":  max_new_tokens,
        },
    }

    t0 = time.time()
    try:
        r = requests.post(
            f"{state['host']}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return {
            "answer":      f"[local_llm error: {e}]",
            "citations":   citations,
            "chunks_used": len(top_chunks),
            "model":       state["model"],
            "error":       str(e),
        }

    data = r.json()
    answer = (data.get("message") or {}).get("content", "").strip()

    return {
        "answer":      answer,
        "citations":   citations,
        "chunks_used": len(top_chunks),
        "model":       state["model"],
        "elapsed":     round(time.time() - t0, 2),
    }


# ---- CLI smoke test -----------------------------------------------------------

if __name__ == "__main__":
    print(f"Checking Ollama at {OLLAMA_HOST}...")
    state = load_local_llm()
    print(f"✅ Connected. Using model: {state['model']}")

    fake_chunks = [{
        "chunk_id": "test_chunk_1",
        "text": "RLPD requires storing the full replay buffer in memory, "
                "which scales linearly with the number of environment steps. "
                "For long-horizon tasks this becomes prohibitive, often exceeding 40GB.",
        "paper_id": "2605.05863v1",
        "page": 4,
    }]

    out = generate_answer_local(
        "What is the main computational drawback of the RLPD algorithm?",
        fake_chunks,
        state,
    )
    print("\n--- Generated answer ---")
    print(out["answer"])
    print(f"\n--- Metadata ---")
    print(json.dumps({k: v for k, v in out.items() if k != "answer"}, indent=2))
"""
ingestion/chunker.py
---------------------
Splits parsed PDF pages into overlapping text chunks for embedding.

Sliding window approach:
    - chunk_size  : 400 tokens  (fits bge-small-en 512 token limit)
    - chunk_overlap: 80 tokens  (preserves context at boundaries)

Token count is approximated as len(words) / 0.75 — fast and accurate
enough for chunking without loading a full tokenizer.

Used by: ingestion/embedder.py → stores/ → api/main.py
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE    = 500   # tokens per chunk
DEFAULT_CHUNK_OVERLAP = 80    # token overlap between consecutive chunks
MIN_CHUNK_TOKENS      = 30    # discard chunks shorter than this (noise)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:    str    # "{paper_id}_p{page_num}_c{chunk_index}"
    paper_id:    str
    page_num:    int
    chunk_index: int    # position within the page
    text:        str
    char_start:  int    # character offset in original page text
    char_end:    int    # character offset in original page text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Approximate token count using word count / 0.75."""
    return int(len(text.split()) / 0.75)


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences as the atomic unit for chunking.
    We chunk at sentence boundaries so chunks never cut mid-sentence.
    """
    # Split on period/exclamation/question followed by space + capital letter
    # Also split on double newlines (paragraph breaks)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def chunk_page(
    text:          str,
    paper_id:      str,
    page_num:      int,
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split a single page's text into overlapping chunks.

    Strategy:
        1. Split page into sentences
        2. Greedily add sentences until chunk_size is reached
        3. Slide back chunk_overlap tokens and start next chunk
        4. Track char_start / char_end via string search

    Returns a list of Chunk dataclass instances.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks      : list[Chunk] = []
    chunk_index : int         = 0
    i           : int         = 0   # sentence pointer

    while i < len(sentences):
        current_sentences : list[str] = []
        current_tokens    : int       = 0

        j = i
        # Fill chunk up to chunk_size tokens
        while j < len(sentences):
            sent_tokens = estimate_tokens(sentences[j])
            if current_tokens + sent_tokens > chunk_size and current_sentences:
                break
            current_sentences.append(sentences[j])
            current_tokens += sent_tokens
            j += 1

        # Build chunk text
        chunk_text = " ".join(current_sentences).strip()

        # Skip chunks that are too short (headers, page numbers, noise)
        if estimate_tokens(chunk_text) >= MIN_CHUNK_TOKENS:
            # Find character offsets in original page text
            char_start = text.find(current_sentences[0]) if current_sentences else 0
            char_end   = char_start + len(chunk_text)
            if char_start == -1:
                char_start = 0
                char_end   = len(chunk_text)

            chunks.append(Chunk(
                chunk_id    = f"{paper_id}_p{page_num}_c{chunk_index}",
                paper_id    = paper_id,
                page_num    = page_num,
                chunk_index = chunk_index,
                text        = chunk_text,
                char_start  = char_start,
                char_end    = min(char_end, len(text)),
            ))
            chunk_index += 1

        # Slide back by overlap — find how many sentences to backtrack
        if j >= len(sentences):
            break

        overlap_tokens = 0
        backtrack      = 0
        for k in range(len(current_sentences) - 1, -1, -1):
            overlap_tokens += estimate_tokens(current_sentences[k])
            backtrack      += 1
            if overlap_tokens >= chunk_overlap:
                break

        # Next window starts at (i + sentences_consumed - backtrack)
        sentences_consumed = j - i
        i += max(1, sentences_consumed - backtrack)

    return chunks


# ---------------------------------------------------------------------------
# Document-level chunker — main entry point
# ---------------------------------------------------------------------------

def chunk_document(
    document:      dict,
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Chunk all pages in a parsed document dict (output of pdf_parser.parse_pdf).

    Args:
        document     : dict from parse_pdf() with keys: paper_id, pages
        chunk_size   : max tokens per chunk
        chunk_overlap: overlap tokens between chunks

    Returns:
        Flat list of Chunk instances across all pages.
    """
    paper_id = document["paper_id"]
    all_chunks: list[Chunk] = []

    for page in document.get("pages", []):
        page_chunks = chunk_page(
            text          = page["text"],
            paper_id      = paper_id,
            page_num      = page["page_num"],
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
        )
        all_chunks.extend(page_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, ".")
    from ingestion.pdf_parser import parse_pdf

    csv_path = "data/papers_enriched.csv"
    df       = pd.read_csv(csv_path)
    row      = df.iloc[0]

    print(f"Parsing: {row['pdf_path']}\n")
    doc    = parse_pdf(row["pdf_path"], csv_row=row)
    chunks = chunk_document(doc)

    print(f"paper_id    : {doc['paper_id']}")
    print(f"pages       : {doc['page_count']}")
    print(f"total chunks: {len(chunks)}")
    print(f"avg per page: {len(chunks) / max(doc['page_count'], 1):.1f}")

    print(f"\n--- First 3 chunks ---")
    for c in chunks[:3]:
        print(f"\n  chunk_id  : {c.chunk_id}")
        print(f"  page_num  : {c.page_num}")
        print(f"  tokens    : ~{estimate_tokens(c.text)}")
        print(f"  char_start: {c.char_start}")
        print(f"  char_end  : {c.char_end}")
        print(f"  text      : {c.text[:120]}...")
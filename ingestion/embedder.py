"""
ingestion/embedder.py
----------------------
Embeds text chunks using BAAI/bge-small-en-v1.5.
Returns (clean_chunks, embeddings) only — storing is done by api/main.py.

Used by: api/main.py /ingest endpoint
"""

import os
import re

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from ingestion.chunker import Chunk

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL      = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH_SIZE = 64
VECTOR_DIM       = 384

# BGE prefixes
BGE_DOC_PREFIX   = "Represent this passage: "
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Boilerplate filter
# ---------------------------------------------------------------------------

BOILERPLATE_PATTERNS = [
    r"permission to make digital or hard copies",
    r"all or part of this work for personal or classroom",
    r"acm isbn",
    r"https?://doi\.org",
    r"^\s*page\s+\d+\s+of\s+\d+\s*$",
    r"^\s*\d+\s*$",
    r"arxiv:\d{4}\.\d{4,5}",
    r"preprint\.\s*do not distribute",
    r"under review",
    r"^\s*references\s*$",
    r"^\s*acknowledgements?\s*$",
]

_BOILERPLATE_RE = re.compile(
    "|".join(BOILERPLATE_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


def is_boilerplate(text: str) -> bool:
    """Return True if chunk is mostly boilerplate noise."""
    lines   = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    matches = sum(1 for l in lines if _BOILERPLATE_RE.search(l))
    return (matches / len(lines)) > 0.6


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model():
    """Load embedding model. Cached after first call."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Core — embed only, no store logic
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks : list[Chunk],
    model,
) -> tuple[list[Chunk], np.ndarray]:
    """
    Filter boilerplate then embed clean chunks in batches.

    Returns:
        clean_chunks : list[Chunk] after boilerplate filter
        embeddings   : np.ndarray shape (len(clean_chunks), 384)
    """
    clean_chunks = [c for c in chunks if not is_boilerplate(c.text)]
    skipped      = len(chunks) - len(clean_chunks)
    if skipped:
        print(f"  Boilerplate filter: skipped {skipped} chunks")

    if not clean_chunks:
        return [], np.array([])

    texts = [BGE_DOC_PREFIX + c.text for c in clean_chunks]

    all_embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="  Embedding"):
        batch      = texts[i : i + EMBED_BATCH_SIZE]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(embeddings)

    return clean_chunks, np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, ".")

    from ingestion.pdf_parser import parse_pdf
    from ingestion.chunker    import chunk_document

    csv_path = "data/papers_enriched.csv"
    df       = pd.read_csv(csv_path)
    row      = df.iloc[0]

    print(f"Parsing : {row['pdf_path']}")
    doc    = parse_pdf(row["pdf_path"], csv_row=row)
    chunks = chunk_document(doc)

    print(f"Loading model: {EMBED_MODEL}")
    model = load_model()

    clean_chunks, embeddings = embed_chunks(chunks, model)

    print(f"\npaper_id        : {doc['paper_id']}")
    print(f"chunks total    : {len(chunks)}")
    print(f"chunks embedded : {len(clean_chunks)}")
    print(f"embeddings shape: {embeddings.shape}")
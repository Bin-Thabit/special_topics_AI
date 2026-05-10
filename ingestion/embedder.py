"""
ingestion/embedder.py
----------------------
Embeds text chunks using BAAI/bge-small-en and writes to:
    - Qdrant  : vector + chunk_id  (similarity search)
    - MongoDB : full chunk text + metadata (content retrieval)

Both stores are linked by chunk_id so Qdrant search results
can be hydrated with full text from MongoDB.

Used by: api/main.py /ingest endpoint
"""

import re
import uuid
from dataclasses import asdict
from typing import Optional

import numpy as np
from tqdm import tqdm

from ingestion.chunker import Chunk
import os
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH_SIZE  = 64
VECTOR_DIM        = 384
QDRANT_COLLECTION = "chunks"

# BGE document prefix — improves retrieval accuracy for bge models
BGE_DOC_PREFIX = "Represent this passage: "

# ---------------------------------------------------------------------------
# Boilerplate filter
# ---------------------------------------------------------------------------

BOILERPLATE_PATTERNS = [
    r"permission to make digital or hard copies",
    r"all or part of this work for personal or classroom",
    r"acm isbn",
    r"https?://doi\.org",
    r"^\s*page\s+\d+\s+of\s+\d+\s*$",
    r"^\s*\d+\s*$",                          # page number only
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
    """Return True if the chunk is mostly boilerplate noise."""
    # If more than 60% of lines match boilerplate patterns → skip
    lines     = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    matches   = sum(1 for l in lines if _BOILERPLATE_RE.search(l))
    return (matches / len(lines)) > 0.6


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_model():
    """Load bge-small-en model. Cached after first call."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed_chunks(
    chunks: list[Chunk],
    model,
) -> tuple[list[Chunk], np.ndarray]:
    """
    Filter boilerplate, then embed remaining chunks in batches.

    Returns:
        clean_chunks : list of Chunk after boilerplate filter
        embeddings   : np.ndarray of shape (len(clean_chunks), 384)
    """
    # Filter boilerplate
    clean_chunks = [c for c in chunks if not is_boilerplate(c.text)]
    skipped      = len(chunks) - len(clean_chunks)
    if skipped:
        print(f"  Boilerplate filter: skipped {skipped} chunks")

    if not clean_chunks:
        return [], np.array([])

    # Add BGE document prefix to each chunk text
    texts = [BGE_DOC_PREFIX + c.text for c in clean_chunks]

    # Embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="  Embedding batches"):
        batch      = texts[i : i + EMBED_BATCH_SIZE]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(embeddings)

    return clean_chunks, np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Qdrant writer
# ---------------------------------------------------------------------------

def write_to_qdrant(
    chunks:     list[Chunk],
    embeddings: np.ndarray,
    qdrant_client,
) -> None:
    """
    Upsert vectors into Qdrant.
    Each point stores: chunk_id, paper_id, page_num as payload
    (full text stays in MongoDB to keep Qdrant lean).
    """
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(
            id      = str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)),
            vector  = embeddings[i].tolist(),
            payload = {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "page_num": c.page_num,
            },
        )
        for i, c in enumerate(chunks)
    ]

    # Upsert in batches of 256
    batch_size = 256
    for i in range(0, len(points), batch_size):
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i : i + batch_size],
        )


# ---------------------------------------------------------------------------
# MongoDB writer
# ---------------------------------------------------------------------------

def write_to_mongo(
    chunks:   list[Chunk],
    document: dict,
    mongo_db,
) -> None:
    """
    Insert chunk documents into MongoDB chunks collection.
    Each document contains full text + all metadata for citation display.
    """
    chunk_docs = [
        {
            "chunk_id":    c.chunk_id,
            "paper_id":    c.paper_id,
            "page_num":    c.page_num,
            "chunk_index": c.chunk_index,
            "text":        c.text,
            "char_start":  c.char_start,
            "char_end":    c.char_end,
            # Paper-level metadata denormalized for fast retrieval
            "title":       document.get("title"),
            "authors":     document.get("authors", []),
            "year":        document.get("year"),
            "venue":       document.get("venue"),
            "topics":      document.get("topics", []),
            "pdf_path":    document.get("pdf_path"),
        }
        for c in chunks
    ]

    if chunk_docs:
        mongo_db["chunks"].insert_many(chunk_docs, ordered=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def embed_document(
    document:      dict,
    chunks:        list[Chunk],
    model,
    qdrant_client,
    mongo_db,
) -> dict:
    """
    Full embed + store pipeline for a single document.

    Args:
        document      : parsed doc dict from pdf_parser.parse_pdf()
        chunks        : list of Chunk from chunker.chunk_document()
        model         : loaded SentenceTransformer model
        qdrant_client : connected Qdrant client
        mongo_db      : connected MongoDB database

    Returns:
        summary dict with counts
    """
    paper_id = document["paper_id"]
    print(f"\n  [{paper_id}] {len(chunks)} chunks → embedding...")

    clean_chunks, embeddings = embed_chunks(chunks, model)

    if not clean_chunks:
        print(f"  [{paper_id}] No clean chunks — skipping stores.")
        return {
            "paper_id":        paper_id,
            "chunks_total":    len(chunks),
            "chunks_embedded": 0,
            "chunks_skipped":  len(chunks),
        }

    write_to_qdrant(clean_chunks, embeddings, qdrant_client)
    write_to_mongo(clean_chunks, document, mongo_db)

    summary = {
        "paper_id":        paper_id,
        "chunks_total":    len(chunks),
        "chunks_embedded": len(clean_chunks),
        "chunks_skipped":  len(chunks) - len(clean_chunks),
    }

    print(f"  [{paper_id}] ✓ {len(clean_chunks)} embedded | "
          f"{summary['chunks_skipped']} skipped")

    return summary


# ---------------------------------------------------------------------------
# Smoke test — requires Docker Compose running
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, ".")

    from ingestion.pdf_parser  import parse_pdf
    from ingestion.chunker     import chunk_document
    from stores.qdrant_client  import get_qdrant_client, ensure_collection
    from stores.mongo_client   import get_mongo_db

    csv_path = "data/papers_enriched.csv"
    df       = pd.read_csv(csv_path)
    row      = df.iloc[0]

    print(f"Parsing : {row['pdf_path']}")
    doc    = parse_pdf(row["pdf_path"], csv_row=row)
    chunks = chunk_document(doc)

    print(f"Loading model: {EMBED_MODEL}")
    model = load_model()

    print("Connecting to stores...")
    qdrant = get_qdrant_client()
    ensure_collection(qdrant)
    mongo  = get_mongo_db()

    summary = embed_document(doc, chunks, model, qdrant, mongo)

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")
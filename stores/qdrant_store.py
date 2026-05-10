"""
stores/qdrant_client.py
------------------------
Thin wrapper around the Qdrant client.

Responsibilities:
    - Connect to Qdrant using .env config
    - Create/verify the 'chunks' collection (384-dim, cosine distance)
    - Upsert vectors
    - Search by query vector → returns top-k chunk_ids with scores

Used by: ingestion/embedder.py, api/main.py
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME   = "chunks"
VECTOR_DIM        = 384
DISTANCE          = Distance.COSINE

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    """
    Return a connected Qdrant client.
    Reads QDRANT_HOST and QDRANT_PORT from .env
    """
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create the 'chunks' collection if it doesn't exist.

    Args:
        client   : connected QdrantClient
        recreate : if True, drop and recreate (useful for dev resets)
    """
    existing = [c.name for c in client.get_collections().collections]

    if recreate and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Qdrant: dropped collection '{COLLECTION_NAME}'")
        existing = []

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size     =VECTOR_DIM,
                distance =DISTANCE,
            ),
        )
        print(f"  Qdrant: created collection '{COLLECTION_NAME}' "
              f"(dim={VECTOR_DIM}, distance={DISTANCE})")
    else:
        print(f"  Qdrant: collection '{COLLECTION_NAME}' already exists — skipping.")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def upsert_vectors(
    client:     QdrantClient,
    chunk_ids:  list[str],
    vectors:    list[list[float]],
    payloads:   list[dict],
    batch_size: int = 256,
) -> None:
    """
    Upsert vectors into Qdrant in batches.

    Args:
        client     : connected QdrantClient
        chunk_ids  : deterministic string IDs (converted to UUID internally)
        vectors    : list of embedding vectors (each len=384)
        payloads   : list of dicts — stored alongside vector (chunk_id, paper_id, page_num)
        batch_size : upsert batch size
    """
    import uuid

    points = [
        PointStruct(
            id      = str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)),
            vector  = vec,
            payload = payload,
        )
        for cid, vec, payload in zip(chunk_ids, vectors, payloads)
    ]

    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )

    print(f"  Qdrant: upserted {len(points)} vectors into '{COLLECTION_NAME}'")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_vectors(
    client:       QdrantClient,
    query_vector: list[float],
    top_k:        int = 10,
    paper_id:     str | None = None,
) -> list[dict]:
    """
    Search Qdrant for the top-k most similar chunks.

    Args:
        client       : connected QdrantClient
        query_vector : embedded query (len=384)
        top_k        : number of results to return
        paper_id     : optional filter — restrict search to one paper

    Returns:
        list of dicts: [{ chunk_id, paper_id, page_num, score }, ...]
    """
    query_filter = None
    if paper_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key   ="paper_id",
                    match =MatchValue(value=paper_id),
                )
            ]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector   =query_vector,
        limit          =top_k,
        query_filter   =query_filter,
        with_payload   =True,
    )

    return [
        {
            "chunk_id": r.payload.get("chunk_id"),
            "paper_id": r.payload.get("paper_id"),
            "page_num": r.payload.get("page_num"),
            "score":    round(r.score, 4),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Smoke test — requires Qdrant running via Docker Compose
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = get_qdrant_client()

    # Health check
    info = client.get_collections()
    print(f"Qdrant alive — {len(info.collections)} collection(s) found")

    # Create collection
    ensure_collection(client)

    # Test upsert with dummy vectors
    import numpy as np
    dummy_vectors  = np.random.rand(3, VECTOR_DIM).tolist()
    dummy_ids      = ["test_p1_c0", "test_p1_c1", "test_p1_c2"]
    dummy_payloads = [
        {"chunk_id": cid, "paper_id": "test_paper", "page_num": 1}
        for cid in dummy_ids
    ]

    upsert_vectors(client, dummy_ids, dummy_vectors, dummy_payloads)

    # Test search
    query = np.random.rand(VECTOR_DIM).tolist()
    hits  = search_vectors(client, query, top_k=3)
    print(f"\nSearch results (random query):")
    for h in hits:
        print(f"  {h}")
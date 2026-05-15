"""
stores/mongo_store.py
----------------------
Thin wrapper around pymongo.

Collections:
    chunks : one doc per chunk — full text + metadata + char offsets
    papers : one doc per paper — title, authors, year, venue, topics, abstract

Used by: ingestion/embedder.py, api/main.py
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "papers_db")

# TTL for run cards — auto-expire after 30 days
RUN_CARD_TTL_SECONDS = 30 * 24 * 60 * 60

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_mongo_client() -> MongoClient:
    """Return a connected MongoClient."""
    return MongoClient(MONGO_URI)


def get_mongo_db(client: MongoClient | None = None) -> Database:
    """
    Return the papers_db database.
    Creates a new client if none is provided.
    """
    if client is None:
        client = get_mongo_client()
    return client[MONGO_DB]


# ---------------------------------------------------------------------------
# Index setup
# ---------------------------------------------------------------------------

def ensure_indexes(db: Database) -> None:
    """
    Create all indexes on startup.
    Safe to call multiple times — MongoDB skips existing indexes.
    """
    # chunks collection
    chunks: Collection = db["chunks"]
    chunks.create_index([("chunk_id", ASCENDING)], unique=True, name="idx_chunk_id")
    chunks.create_index([("paper_id", ASCENDING)],             name="idx_paper_id")
    chunks.create_index([("page_num", ASCENDING)],             name="idx_page_num")
    chunks.create_index([("topics",   ASCENDING)],             name="idx_topics")

    # papers collection
    papers: Collection = db["papers"]
    papers.create_index([("paper_id",    ASCENDING)], unique=True, name="idx_paper_id")
    papers.create_index([("year",        ASCENDING)],             name="idx_year")
    papers.create_index([("topics",      ASCENDING)],             name="idx_topics")

    # run_cards collection — TTL index auto-expires docs after 30 days
    run_cards: Collection = db["run_cards"]
    run_cards.create_index(
        [("ingested_at", ASCENDING)],
        expireAfterSeconds=RUN_CARD_TTL_SECONDS,
        name="idx_ttl_ingested_at",
    )

    print("  MongoDB: indexes ensured on chunks, papers, run_cards")


# ---------------------------------------------------------------------------
# Papers collection
# ---------------------------------------------------------------------------

def paper_exists(db: Database, paper_id: str) -> bool:
    """Return True if the paper has already been ingested."""
    return db["papers"].count_documents({"paper_id": paper_id}, limit=1) > 0


def insert_paper(db: Database, document: dict) -> None:
    """
    Insert or replace paper-level metadata.
    Uses replace_one with upsert=True so re-ingesting is safe.
    """
    doc = {
        "paper_id":    document.get("paper_id"),
        "title":       document.get("title"),
        "authors":     document.get("authors", []),
        "year":        document.get("year"),
        "venue":       document.get("venue"),
        "doi":         document.get("doi"),
        "topics":      document.get("topics", []),
        "abstract":    document.get("abstract"),
        "pdf_path":    document.get("pdf_path"),
        "page_count":  document.get("page_count", 0),
        "ingested_at": datetime.now(timezone.utc),
    }
    db["papers"].replace_one(
        {"paper_id": document["paper_id"]},
        doc,
        upsert=True,
    )


def get_paper(db: Database, paper_id: str) -> dict | None:
    """Fetch paper metadata by paper_id."""
    return db["papers"].find_one({"paper_id": paper_id}, {"_id": 0})


# ---------------------------------------------------------------------------
# Chunks collection
# ---------------------------------------------------------------------------

def insert_chunks(
    db:       Database,
    chunks:   list,           # list of Chunk dataclass instances
    document: dict,
) -> int:
    """
    Bulk insert all chunks for a paper.
    Skips duplicates silently (ordered=False).

    Returns number of chunks inserted.
    """
    if not chunks:
        return 0

    chunk_docs = [
        {
            "chunk_id":    c.chunk_id,
            "paper_id":    c.paper_id,
            "page_num":    c.page_num,
            "chunk_index": c.chunk_index,
            "text":        c.text,
            "char_start":  c.char_start,
            "char_end":    c.char_end,
            # Denormalized paper metadata for fast retrieval
            "title":       document.get("title"),
            "authors":     document.get("authors", []),
            "year":        document.get("year"),
            "venue":       document.get("venue"),
            "topics":      document.get("topics", []),
            "pdf_path":    document.get("pdf_path"),
            "ingested_at": datetime.now(timezone.utc),
        }
        for c in chunks
    ]

    try:
        result = db["chunks"].insert_many(chunk_docs, ordered=False)
        return len(result.inserted_ids)
    except Exception as e:
        # BulkWriteError on duplicates — count successful inserts
        inserted = getattr(e, "details", {}).get("nInserted", 0)
        return inserted


def get_chunks_by_ids(
    db:        Database,
    chunk_ids: list[str],
) -> list[dict]:
    """
    Fetch full chunk documents by chunk_id list.
    Called after Qdrant search to hydrate results with full text.
    Preserves the order of chunk_ids (Qdrant score order).
    """
    docs = list(
        db["chunks"].find(
            {"chunk_id": {"$in": chunk_ids}},
            {"_id": 0},
        )
    )

    # Restore Qdrant score order
    order  = {cid: i for i, cid in enumerate(chunk_ids)}
    docs.sort(key=lambda d: order.get(d["chunk_id"], 999))
    return docs


def get_chunks_by_paper(
    db:       Database,
    paper_id: str,
) -> list[dict]:
    """Fetch all chunks for a paper ordered by page and chunk index."""
    return list(
        db["chunks"]
        .find({"paper_id": paper_id}, {"_id": 0})
        .sort([("page_num", ASCENDING), ("chunk_index", ASCENDING)])
    )


def delete_paper_chunks(db: Database, paper_id: str) -> int:
    """Delete all chunks for a paper. Used when re-ingesting."""
    result = db["chunks"].delete_many({"paper_id": paper_id})
    return result.deleted_count


# ---------------------------------------------------------------------------
# Run cards
# ---------------------------------------------------------------------------

def insert_run_card(db: Database, run_card: dict) -> None:
    """
    Insert an ingest run card for provenance tracking.
    Auto-expires after 30 days via TTL index.
    """
    run_card["ingested_at"] = datetime.now(timezone.utc)
    db["run_cards"].insert_one(run_card)


# ---------------------------------------------------------------------------
# Smoke test — requires MongoDB running via Docker Compose
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")  # add this line
    print(f"Connecting to MongoDB at {MONGO_URI}...")
    client = get_mongo_client()
    db     = get_mongo_db(client)

    # Health check
    client.admin.command("ping")
    print("MongoDB alive")

    # Ensure indexes
    ensure_indexes(db)

    # Test insert paper
    dummy_doc = {
        "paper_id":   "test_001",
        "title":      "Test Paper",
        "authors":    ["Alice", "Bob"],
        "year":       2026,
        "venue":      "arXiv",
        "topics":     ["reasoning", "planning_and_scheduling"],
        "abstract":   "A test abstract.",
        "pdf_path":   "data/pdfs/test_001.pdf",
        "page_count": 5,
    }
    insert_paper(db, dummy_doc)
    print(f"\nInserted paper: {dummy_doc['paper_id']}")
    print(f"paper_exists  : {paper_exists(db, 'test_001')}")

    # Test insert chunks
    from ingestion.chunker import Chunk
    dummy_chunks = [
        Chunk(
            chunk_id    = f"test_001_p1_c{i}",
            paper_id    = "test_001",
            page_num    = 1,
            chunk_index = i,
            text        = f"This is dummy chunk number {i} with some test content.",
            char_start  = i * 100,
            char_end    = i * 100 + 50,
        )
        for i in range(3)
    ]
    inserted = insert_chunks(db, dummy_chunks, dummy_doc)
    print(f"Inserted chunks: {inserted}")

    # Test get_chunks_by_ids
    ids    = [c.chunk_id for c in dummy_chunks]
    fetched = get_chunks_by_ids(db, ids)
    print(f"Fetched chunks : {len(fetched)}")
    for c in fetched:
        print(f"  {c['chunk_id']} | page {c['page_num']} | {c['text'][:50]}")

    # Cleanup test data
    db["chunks"].delete_many({"paper_id": "test_001"})
    db["papers"].delete_many({"paper_id": "test_001"})
    print("\nTest data cleaned up.")
    print("mongo_store smoke test passed.")
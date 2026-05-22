"""
scripts/seed_all.py
--------------------
One-command seed script that sets up all three stores from scratch.

Order:
    1. Enrich subtopics (if papers_enriched.csv doesn't exist)
    2. Ingest all PDFs → MongoDB + Qdrant
    3. Seed Neo4j graph from papers_enriched.csv
    4. Rebuild retrieval indexes
    5. Run eval to verify everything works

Usage:
    python scripts/seed_all.py
    python scripts/seed_all.py --skip-enrich   # if papers_enriched.csv exists
    python scripts/seed_all.py --limit 10      # test with 10 papers first

Requirements:
    docker compose up -d   (all three stores must be running)
    uvicorn api.main:app --host 0.0.0.0 --port 8000  (API must be running)
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def check_api() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        print(f"  API alive — chunks in store: {data.get('chunks', 0)}")
        return True
    except Exception as e:
        print(f"  API not reachable: {e}")
        print("  Start it with: uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False


def check_stores() -> bool:
    """Check MongoDB, Qdrant, Neo4j are all reachable."""
    all_ok = True

    # MongoDB
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                            serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        print("  MongoDB   ✅")
        client.close()
    except Exception as e:
        print(f"  MongoDB   ❌ {e}")
        all_ok = False

    # Qdrant
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        qc.get_collections()
        print("  Qdrant    ✅")
    except Exception as e:
        print(f"  Qdrant    ❌ {e}")
        all_ok = False

    # Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"),
                  os.getenv("NEO4J_PASSWORD", "your_password"))
        )
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        print("  Neo4j     ✅")
    except Exception as e:
        print(f"  Neo4j     ❌ {e}")
        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_enrich() -> None:
    banner("Step 1 — Enriching subtopics")
    if Path("data/papers_enriched.csv").exists():
        print("  data/papers_enriched.csv already exists — skipping.")
        return
    print("  Running enrich_subtopics.py ...")
    subprocess.run([sys.executable, "scripts/enrich_subtopics.py"], check=True)


def step_ingest(limit: int | None) -> None:
    banner("Step 2 — Batch ingesting PDFs → MongoDB + Qdrant")
    cmd = [sys.executable, "scripts/batch_ingest.py"]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)


def step_neo4j() -> None:
    banner("Step 3 — Seeding Neo4j graph")
    print("  Running graphrag/seed_neo4j.py ...")
    subprocess.run([sys.executable, "graphrag/seed_neo4j.py"], check=True)


def step_rebuild_index() -> None:
    banner("Step 4 — Rebuilding retrieval indexes")
    try:
        r = httpx.post(f"{API_URL}/rebuild-index", timeout=300)
        data = r.json()
        print(f"  Indexes rebuilt — {data.get('chunks_indexed')} chunks")
        print(f"  Elapsed: {data.get('elapsed_seconds')}s")
    except Exception as e:
        print(f"  Rebuild failed: {e}")
        print("  You can run it manually: curl -X POST http://localhost:8000/rebuild-index")


def step_eval() -> None:
    banner("Step 5 — Running eval to verify")
    subprocess.run(
        [sys.executable, "scripts/eval_recall.py", "--k", "5"],
        check=True
    )


def step_verify() -> None:
    """Print final store counts."""
    banner("Final verification")

    # MongoDB chunk count
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db     = client[os.getenv("MONGO_DB", "papers_db")]
        chunks = db["chunks"].count_documents({})
        papers = db["papers"].count_documents({})
        print(f"  MongoDB  — papers: {papers} | chunks: {chunks}")
        client.close()
    except Exception as e:
        print(f"  MongoDB check failed: {e}")

    # Qdrant vector count
    try:
        from qdrant_client import QdrantClient
        qc   = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        info = qc.get_collection("chunks")
        print(f"  Qdrant   — vectors: {info.vectors_count}")
    except Exception as e:
        print(f"  Qdrant check failed: {e}")

    # Neo4j node count
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"),
                  os.getenv("NEO4J_PASSWORD", "your_password"))
        )
        with driver.session() as session:
            papers  = session.run("MATCH (p:Paper)  RETURN count(p) AS c").single()["c"]
            authors = session.run("MATCH (a:Author) RETURN count(a) AS c").single()["c"]
            topics  = session.run("MATCH (t:Topic)  RETURN count(t) AS c").single()["c"]
        driver.close()
        print(f"  Neo4j    — papers: {papers} | authors: {authors} | topics: {topics}")
    except Exception as e:
        print(f"  Neo4j check failed: {e}")

    # API health
    try:
        r    = httpx.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        print(f"  API      — chunks indexed: {data.get('chunks')} | alpha: {data.get('alpha')}")
    except Exception as e:
        print(f"  API check failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_enrich: bool, limit: int | None, skip_eval: bool) -> None:
    start = time.time()

    banner("CSAI415 — D2 Full Store Seed")
    print("  Checking stores are running...")
    if not check_stores():
        print("\n  Start stores first: docker compose up -d")
        sys.exit(1)

    print("\n  Checking API is running...")
    if not check_api():
        sys.exit(1)

    if not skip_enrich:
        step_enrich()

    step_ingest(limit)
    step_neo4j()
    step_rebuild_index()

    if not skip_eval:
        step_eval()

    step_verify()

    elapsed = round(time.time() - start, 1)
    banner(f"Seed complete in {elapsed}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed all stores: MongoDB + Qdrant + Neo4j"
    )
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Skip subtopic enrichment (papers_enriched.csv exists)")
    parser.add_argument("--skip-eval",   action="store_true",
                        help="Skip eval_recall.py after seeding")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Limit papers for testing (default: all)")
    args = parser.parse_args()

    main(
        skip_enrich=args.skip_enrich,
        limit      =args.limit,
        skip_eval  =args.skip_eval,
    )
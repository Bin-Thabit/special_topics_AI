"""
api/main.py
------------
FastAPI application with two endpoints:

    POST /ingest  — upload PDF → parse → chunk → embed → store
    POST /search  — query → hybrid_search() → cited results

Startup:
    - Reads run_card.yaml for best_alpha, best_k, best_svd_dim
    - Loads embedding model once
    - Connects to Qdrant and MongoDB
    - Loads all chunks from MongoDB + builds BM25 + dense indexes
    - Warm-starts HybridWeightAdapter with AutoML best_alpha
"""

import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, ".")
load_dotenv()

from ingestion.chunker    import chunk_document
from ingestion.embedder   import (
    EMBED_MODEL,
    BGE_QUERY_PREFIX,
    embed_chunks,
    load_model,
)
from ingestion.pdf_parser import parse_pdf
from stores.mongo_store   import (
    ensure_indexes,
    get_chunks_by_ids,
    get_mongo_db,
    insert_chunks,
    insert_paper,
    insert_run_card,
    paper_exists,
)
from stores.qdrant_store  import (
    COLLECTION_NAME,
    VECTOR_DIM,
    ensure_collection,
    get_qdrant_client,
    upsert_vectors,
)

from retrieval.hybrid_retriever import hybrid_search

# ---------------------------------------------------------------------------
# Load run card — AutoML best hyperparameters
# ---------------------------------------------------------------------------

RUN_CARD_PATH = "run_card.yaml"

def load_run_card() -> dict:
    """
    Load AutoML run card from main folder.
    Reads best_params section only.
    Falls back to safe defaults if file not found.
    """
    defaults = {
        "best_alpha":   0.5,
        "best_k":       10,
        "best_svd_dim": 384,
    }
    if not os.path.exists(RUN_CARD_PATH):
        print(f"  run_card.yaml not found — using defaults {defaults}")
        return defaults

    with open(RUN_CARD_PATH) as f:
        card = yaml.safe_load(f) or {}

    # Read from best_params section only
    best = card.get("best_params", {})

    merged = {
        "best_alpha":   best.get("alpha",   defaults["best_alpha"]),
        "best_k":       best.get("k",       defaults["best_k"]),
        "best_svd_dim": best.get("svd_dim", defaults["best_svd_dim"]) or 384,
    }

    print(f"  run_card.yaml loaded:")
    print(f"    best_alpha   = {merged['best_alpha']}")
    print(f"    best_k       = {merged['best_k']}")
    print(f"    best_svd_dim = {merged['best_svd_dim']}")
    return merged


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    model            = None   # SentenceTransformer
    qdrant           = None   # QdrantClient
    mongo_db         = None   # MongoDB Database
    hybrid_adapter   = None   # HybridWeightAdapter (D1)
    run_card         = {}     # AutoML best hyperparameters
    # Built once at startup from MongoDB chunks
    all_chunks       = []     # list[dict] — all chunks in the store
    bm25_index       = None   # BM25Okapi index
    dense_embeddings = None   # np.ndarray (N, 384)
    default_k        = 10

state = AppState()


# ---------------------------------------------------------------------------
# Build retrieval indexes from MongoDB
# ---------------------------------------------------------------------------

def build_retrieval_indexes() -> None:
    """
    Load all chunks from MongoDB and build:
        - BM25 index (rank-bm25)
        - Dense embeddings matrix (sentence-transformers)

    Called once at startup and after every /ingest.
    """
    from rank_bm25 import BM25Okapi
    from retrieval.bm25_retriever import build_bm25_index
    from retrieval.dense_retriever import build_dense_index

    print("  Building retrieval indexes from MongoDB...")
    state.all_chunks = list(
        state.mongo_db["chunks"].find({}, {"_id": 0})
    )

    if not state.all_chunks:
        print("  No chunks in MongoDB yet — indexes empty.")
        state.bm25_index       = None
        state.dense_embeddings = None
        return

    state.bm25_index       = build_bm25_index(state.all_chunks)
    state.dense_embeddings = build_dense_index(
        state.all_chunks,
        state.model,
    )
    print(f"  Indexes built: {len(state.all_chunks)} chunks")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")

    # Load run card
    state.run_card  = load_run_card()
    state.default_k = state.run_card.get("best_k", 10)
    alpha           = state.run_card.get("best_alpha", 0.5)

    # Load embedding model
    print(f"  Loading model: {EMBED_MODEL}")
    state.model = load_model()

    # Connect to Qdrant
    state.qdrant = get_qdrant_client()
    ensure_collection(state.qdrant)

    # Connect to MongoDB
    state.mongo_db = get_mongo_db()
    ensure_indexes(state.mongo_db)

    # Warm-start HybridWeightAdapter with AutoML best_alpha
    try:
        from adaptation.online_learner import HybridWeightAdapter
        state.hybrid_adapter = HybridWeightAdapter(alpha=alpha)
        print(f"  HybridWeightAdapter warm-started with alpha={alpha}")
    except ImportError:
        print("  HybridWeightAdapter not found — using static alpha")
        state.hybrid_adapter = None

    # Build BM25 + dense retrieval indexes
    build_retrieval_indexes()

    print("Startup complete.\n")
    yield
    print("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       ="PDF-Papers AI Agent",
    description ="Hybrid Retrieval + GraphRAG with Online Learning",
    version     ="0.2.0",
    lifespan    =lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query : str
    top_k : int | None = None   # defaults to run_card best_k


class ChunkResult(BaseModel):
    chunk_id    : str
    paper_id    : str
    title       : str | None
    page_num    : int
    score       : float
    bm25_score  : float
    dense_score : float
    text        : str


class SearchResponse(BaseModel):
    query             : str
    results           : list[ChunkResult]
    retrieval_weights : dict
    top_k_used        : int
    elapsed_seconds   : float


class IngestResponse(BaseModel):
    paper_id        : str
    title           : str | None
    chunks_embedded : int
    pages_parsed    : int
    elapsed_seconds : float
    status          : str


# ---------------------------------------------------------------------------
# Helper — embed query
# ---------------------------------------------------------------------------

def embed_query(query: str) -> np.ndarray:
    return state.model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
    )


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload PDF → parse → chunk → embed → write to Qdrant + MongoDB.
    Rebuilds retrieval indexes after successful ingest.
    Skips re-ingestion if paper already exists.
    """
    import tempfile, shutil
    start    = time.time()
    tmp_path = None

    try:
        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Match to papers_enriched.csv
        import pandas as pd
        paper_id_hint = os.path.splitext(file.filename or "unknown")[0]
        csv_path      = "data/papers_enriched.csv"
        csv_row       = None

        if os.path.exists(csv_path):
            df    = pd.read_csv(csv_path)
            match = df[df["paper_id"].astype(str) == paper_id_hint]
            if not match.empty:
                csv_row = match.iloc[0]

        # Parse
        doc = parse_pdf(tmp_path, csv_row=csv_row)
        if csv_row is None:
            doc["paper_id"] = paper_id_hint

        paper_id = doc["paper_id"]

        # Skip if already ingested
        if paper_exists(state.mongo_db, paper_id):
            return IngestResponse(
                paper_id        =paper_id,
                title           =doc.get("title"),
                chunks_embedded =0,
                pages_parsed    =doc.get("page_count", 0),
                elapsed_seconds =round(time.time() - start, 2),
                status          ="skipped — already ingested",
            )

        # Chunk → embed
        chunks                   = chunk_document(doc)
        clean_chunks, embeddings = embed_chunks(chunks, state.model)

        if clean_chunks:
            # Write vectors to Qdrant
            upsert_vectors(
                client    =state.qdrant,
                chunk_ids =[c.chunk_id for c in clean_chunks],
                vectors   =[embeddings[i].tolist() for i in range(len(clean_chunks))],
                payloads  =[
                    {
                        "chunk_id": c.chunk_id,
                        "paper_id": c.paper_id,
                        "page_num": c.page_num,
                    }
                    for c in clean_chunks
                ],
            )

            # Write full text + metadata to MongoDB
            insert_chunks(state.mongo_db, clean_chunks, doc)

        # Paper metadata + provenance
        insert_paper(state.mongo_db, doc)
        insert_run_card(state.mongo_db, {
            "paper_id":        paper_id,
            "chunks_embedded": len(clean_chunks),
            "pages_parsed":    doc.get("page_count", 0),
            "model":           EMBED_MODEL,
        })

        # Rebuild retrieval indexes so new paper is searchable immediately
        build_retrieval_indexes()

        return IngestResponse(
            paper_id        =paper_id,
            title           =doc.get("title"),
            chunks_embedded =len(clean_chunks),
            pages_parsed    =doc.get("page_count", 0),
            elapsed_seconds =round(time.time() - start, 2),
            status          ="ok",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Query → hybrid_search() → cited results.
    Uses Abdullah's hybrid_search() with adaptive alpha from HybridWeightAdapter.
    """
    start = time.time()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if state.bm25_index is None or state.dense_embeddings is None:
        raise HTTPException(status_code=503, detail="No documents ingested yet.")

    # Resolve top_k — request → run_card default
    top_k = request.top_k or state.default_k

    # Get adaptive alpha from D1 HybridWeightAdapter
    if state.hybrid_adapter is not None:
        weights = state.hybrid_adapter.get_weights()
        alpha   = weights.get("dense_weight", 0.5)
    else:
        weights = {"dense_weight": 0.5, "bm25_weight": 0.5}
        alpha   = 0.5

    # Embed query
    query_vector = embed_query(request.query)

    # Call Abdullah's hybrid_search()
    results = hybrid_search(
        query            =request.query,
        chunks           =state.all_chunks,
        bm25_index       =state.bm25_index,
        dense_model      =state.model,
        dense_embeddings =state.dense_embeddings,
        k                =top_k,
        alpha            =alpha,
        query_vector     =query_vector,
    )

    # Build response
    chunk_results = [
        ChunkResult(
            chunk_id    =r["chunk_id"],
            paper_id    =r["paper_id"],
            title       =r.get("title"),
            page_num    =r["page_num"],
            score       =round(r["score"], 4),
            bm25_score  =round(r.get("bm25_score", 0.0), 4),
            dense_score =round(r.get("dense_score", 0.0), 4),
            text        =r["text"],
        )
        for r in results
    ]

    return SearchResponse(
        query             =request.query,
        results           =chunk_results,
        retrieval_weights =weights,
        top_k_used        =top_k,
        elapsed_seconds   =round(time.time() - start, 2),
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status"     : "ok",
        "model"      : EMBED_MODEL,
        "collection" : COLLECTION_NAME,
        "vector_dim" : VECTOR_DIM,
        "chunks"     : len(state.all_chunks),
        "alpha"      : state.hybrid_adapter.get_weights() if state.hybrid_adapter else 0.5,
        "top_k"      : state.default_k,
        "run_card"   : state.run_card,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
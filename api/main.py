"""
api/main.py
------------
FastAPI application with two endpoints:

    POST /ingest  — upload a PDF → parse → chunk → embed → store
    POST /search  — query string → hybrid retrieval → cited results

Startup:
    - Loads embedding model once into memory
    - Connects to Qdrant and MongoDB
    - Ensures collection and indexes exist

Used by: D3 GraphRAG, D4 final /ask endpoint
"""

import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, ".")
load_dotenv()

from ingestion.chunker    import chunk_document
from ingestion.embedder   import (
    EMBED_MODEL,
    BGE_DOC_PREFIX,
    embed_chunks,
    is_boilerplate,
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
    search_vectors,
    upsert_vectors,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOP_K        = int(os.getenv("TOP_K", 10))
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# App state — shared across requests
# ---------------------------------------------------------------------------

class AppState:
    model        = None
    qdrant       = None
    mongo_db     = None
    hybrid       = None   # HybridWeightAdapter from D1 adaptation/

state = AppState()

# ---------------------------------------------------------------------------
# Lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")

    # Load embedding model
    print(f"  Loading model: {EMBED_MODEL}")
    state.model = load_model()
    print(f"  Model loaded.")

    # Connect to Qdrant
    state.qdrant = get_qdrant_client()
    ensure_collection(state.qdrant)

    # Connect to MongoDB
    state.mongo_db = get_mongo_db()
    ensure_indexes(state.mongo_db)

    # Load HybridWeightAdapter from D1
    try:
        from adaptation.online_learner import HybridWeightAdapter
        state.hybrid = HybridWeightAdapter()
        print("  HybridWeightAdapter loaded from adaptation/")
    except ImportError:
        print("  HybridWeightAdapter not found — using default weights (0.5 / 0.5)")
        state.hybrid = None

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
    allow_origins  =["*"],
    allow_methods  =["*"],
    allow_headers  =["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query : str
    top_k : int = TOP_K


class ChunkResult(BaseModel):
    chunk_id : str
    paper_id : str
    title    : str | None
    page_num : int
    score    : float
    text     : str


class SearchResponse(BaseModel):
    query              : str
    results            : list[ChunkResult]
    retrieval_weights  : dict
    elapsed_seconds    : float


class IngestResponse(BaseModel):
    paper_id         : str
    title            : str | None
    chunks_embedded  : int
    pages_parsed     : int
    elapsed_seconds  : float
    status           : str


# ---------------------------------------------------------------------------
# Helper — embed a query string
# ---------------------------------------------------------------------------

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def embed_query(query: str) -> list[float]:
    vec = state.model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
    )
    return vec.tolist()


# ---------------------------------------------------------------------------
# Helper — BM25 search over MongoDB chunks
# ---------------------------------------------------------------------------

def bm25_search(query: str, top_k: int) -> list[dict]:
    """
    Lexical BM25 search using rank-bm25 over all chunks in MongoDB.
    Returns list of {chunk_id, score} dicts.
    """
    try:
        from rank_bm25 import BM25Okapi

        # Fetch all chunks — in production this would be an index
        all_chunks = list(
            state.mongo_db["chunks"].find({}, {"chunk_id": 1, "text": 1, "_id": 0})
        )
        if not all_chunks:
            return []

        corpus     = [c["text"].lower().split() for c in all_chunks]
        bm25       = BM25Okapi(corpus)
        scores     = bm25.get_scores(query.lower().split())
        top_idx    = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "chunk_id": all_chunks[i]["chunk_id"],
                "score":    float(scores[i]),
            }
            for i in top_idx
            if scores[i] > 0
        ]
    except Exception as e:
        print(f"  BM25 error: {e}")
        return []


# ---------------------------------------------------------------------------
# Helper — hybrid fusion (RRF + adaptive weights)
# ---------------------------------------------------------------------------

def hybrid_fusion(
    dense_hits : list[dict],
    bm25_hits  : list[dict],
    top_k      : int,
) -> tuple[list[dict], dict]:
    """
    Fuse dense and BM25 results using Reciprocal Rank Fusion (RRF)
    weighted by HybridWeightAdapter from D1.

    Returns:
        fused  : list of {chunk_id, score} sorted by fused score
        weights: {dense_weight, bm25_weight}
    """
    # Get adaptive weights from D1 HybridWeightAdapter
    if state.hybrid is not None:
        weights = state.hybrid.get_weights()
    else:
        weights = {"dense_weight": 0.5, "bm25_weight": 0.5}

    dense_w = weights.get("dense_weight", 0.5)
    bm25_w  = weights.get("bm25_weight",  0.5)

    # RRF constant
    k = 60

    scores: dict[str, float] = {}

    for rank, hit in enumerate(dense_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + dense_w * (1 / (k + rank + 1))

    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + bm25_w * (1 / (k + rank + 1))

    fused = [
        {"chunk_id": cid, "score": round(score, 4)}
        for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return fused[:top_k], weights


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF → parse → chunk → embed → store in Qdrant + MongoDB.
    Skips re-ingestion if paper already exists.
    """
    start = time.time()

    # Save upload to temp file
    import tempfile, shutil
    suffix   = ".pdf"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Parse PDF — use filename as paper_id fallback
        paper_id_hint = os.path.splitext(file.filename or "unknown")[0]

        import pandas as pd
        csv_path = "data/papers_enriched.csv"
        csv_row  = None
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            match = df[df["paper_id"].astype(str) == paper_id_hint]
            if not match.empty:
                csv_row = match.iloc[0]

        doc = parse_pdf(tmp_path, csv_row=csv_row)

        # Override paper_id with filename hint if CSV didn't match
        if csv_row is None:
            doc["paper_id"] = paper_id_hint

        paper_id = doc["paper_id"]

        # Skip if already ingested
        if paper_exists(state.mongo_db, paper_id):
            elapsed = round(time.time() - start, 2)
            return IngestResponse(
                paper_id        =paper_id,
                title           =doc.get("title"),
                chunks_embedded =0,
                pages_parsed    =doc.get("page_count", 0),
                elapsed_seconds =elapsed,
                status          ="skipped — already ingested",
            )

        # Chunk
        chunks = chunk_document(doc)

        # Embed + filter boilerplate
        clean_chunks, embeddings = embed_chunks(chunks, state.model)

        if clean_chunks:
            # Write to Qdrant
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

            # Write to MongoDB
            insert_chunks(state.mongo_db, clean_chunks, doc)

        # Insert paper metadata
        insert_paper(state.mongo_db, doc)

        # Insert run card for provenance
        insert_run_card(state.mongo_db, {
            "paper_id":        paper_id,
            "chunks_embedded": len(clean_chunks),
            "pages_parsed":    doc.get("page_count", 0),
            "model":           EMBED_MODEL,
        })

        elapsed = round(time.time() - start, 2)
        return IngestResponse(
            paper_id        =paper_id,
            title           =doc.get("title"),
            chunks_embedded =len(clean_chunks),
            pages_parsed    =doc.get("page_count", 0),
            elapsed_seconds =elapsed,
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
    Query string → hybrid retrieval → cited results.
    Fuses dense (Qdrant) + BM25 (MongoDB) using adaptive weights from D1.
    """
    start = time.time()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Embed query
    query_vector = embed_query(request.query)

    # Dense search via Qdrant
    dense_hits = search_vectors(
        client       =state.qdrant,
        query_vector =query_vector,
        top_k        =request.top_k * 2,   # over-fetch before fusion
    )

    # BM25 lexical search
    bm25_hits = bm25_search(request.query, top_k=request.top_k * 2)

    # Hybrid fusion
    fused, weights = hybrid_fusion(dense_hits, bm25_hits, request.top_k)

    if not fused:
        return SearchResponse(
            query             =request.query,
            results           =[],
            retrieval_weights =weights,
            elapsed_seconds   =round(time.time() - start, 2),
        )

    # Hydrate with full text from MongoDB
    chunk_ids = [h["chunk_id"] for h in fused]
    score_map = {h["chunk_id"]: h["score"] for h in fused}
    mongo_chunks = get_chunks_by_ids(state.mongo_db, chunk_ids)

    results = [
        ChunkResult(
            chunk_id =c["chunk_id"],
            paper_id =c["paper_id"],
            title    =c.get("title"),
            page_num =c["page_num"],
            score    =score_map.get(c["chunk_id"], 0.0),
            text     =c["text"],
        )
        for c in mongo_chunks
    ]

    elapsed = round(time.time() - start, 2)
    return SearchResponse(
        query             =request.query,
        results           =results,
        retrieval_weights =weights,
        elapsed_seconds   =elapsed,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status"    : "ok",
        "model"     : EMBED_MODEL,
        "collection": COLLECTION_NAME,
        "vector_dim": VECTOR_DIM,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
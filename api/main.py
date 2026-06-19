"""
api/main.py
------------
FastAPI application — D2 retrieval stack + D3 GraphRAG pipeline + D4 tuned LLM.

Endpoints:
    POST /ingest        — upload PDF → parse → chunk → embed → store
    POST /search        — query → hybrid_search() → cited results
    POST /ask           — query → full GraphRAG pipeline → grounded answer  [D3]
                          + optional llm switching for ablation              [D4]
    POST /rebuild-index — rebuild BM25 + dense indexes after batch ingest
    GET  /health        — liveness check

Startup:
    - Reads run_card.yaml for best_alpha, best_k, best_svd_dim
    - Loads embedding model once
    - Connects to Qdrant, MongoDB, Neo4j                                    [D3]
    - Loads all chunks from MongoDB + builds BM25 + dense indexes
    - Warm-starts HybridWeightAdapter with AutoML best_alpha
    - Initialises TopicParser for query → Neo4j topic mapping               [D3]
    - Connects to local Ollama (tuned QLoRA model)                          [D4]
"""

import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

sys.path.insert(0, ".")
load_dotenv()

# ── D2 imports (unchanged) ────────────────────────────────────────────────────
from graphrag.topic_router import build_topic_router
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

# ── D3 imports ────────────────────────────────────────────────────────────────
from stores.neo4j_store     import Neo4jStore       # Abdullah's store
from graphrag.topic_parser  import TopicParser      # Abdullah's parser
from graphrag.seed_search   import seed_search      # our bridge function
from graphrag.expand        import expand_to_chunks # Abdullah's Step 2
from graphrag.rank          import (                # Abdullah's Step 3
    embed_query       as graphrag_embed_query,
    adaptive_alpha,
    pool_embeddings_from_state,
    rank_pool,
)
from agents.answer  import generate_answer        # Abdullah's Step 4
from graphrag.safety  import (                      # Abdullah's safety layer
    provenance_filter,
    source_pinning_check,
)
from agents.judge   import judge_answer           # Abdullah's judge
from agents.reflect import reflect_answer         # Abdullah's reflect

# ── D4 imports ────────────────────────────────────────────────────────────────
from graphrag.local_llm import (                    # our QLoRA-tuned Ollama client
    load_local_llm,
    generate_answer_local,
)


# ---------------------------------------------------------------------------
# Load run card — AutoML best hyperparameters  (D2, unchanged)
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
# App state  (D2 + D3 fields unchanged — D4 field at bottom)
# ---------------------------------------------------------------------------

class AppState:
    # ── D2 ────────────────────────────────────────────────────────────────────
    model            = None   # SentenceTransformer
    qdrant           = None   # QdrantClient
    mongo_db         = None   # MongoDB Database
    hybrid_adapter   = None   # HybridWeightAdapter (D1)
    run_card         = {}     # AutoML best hyperparameters
    all_chunks       = []     # list[dict] — all chunks in the store
    bm25_index       = None   # BM25Okapi index
    dense_embeddings = None   # np.ndarray (N, 384)
    default_k        = 10
    # ── D3 ────────────────────────────────────────────────────────────────────
    neo4j            = None   # Neo4jStore — for subgraph selection
    topic_parser     = None   # TopicParser — maps query text → Topic names
    # ── D4 ────────────────────────────────────────────────────────────────────
    local_llm        = None   # dict from load_local_llm() — Ollama state

state = AppState()


# ---------------------------------------------------------------------------
# Build retrieval indexes from MongoDB  (D2, unchanged)
# ---------------------------------------------------------------------------

def build_retrieval_indexes() -> None:
    from retrieval.bm25_retriever import build_bm25_index

    print("  Rebuilding retrieval indexes...")
    state.all_chunks = list(
        state.mongo_db["chunks"].find({}, {"_id": 0})
    )

    if not state.all_chunks:
        print("  No chunks in MongoDB yet.")
        state.bm25_index       = None
        state.dense_embeddings = None
        return

    state.bm25_index = build_bm25_index(state.all_chunks)
    _build_dense_from_qdrant()
    print(f"  Indexes rebuilt: {len(state.all_chunks)} chunks")


def _build_dense_from_qdrant() -> None:
    """Pull vectors from Qdrant — fast, no re-encoding.  (D2, unchanged)"""
    import uuid
    print("  Pulling dense vectors from Qdrant...")

    dense_embeddings = []
    ordered_chunks   = []

    for chunk in state.all_chunks:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        try:
            results = state.qdrant.retrieve(
                collection_name=COLLECTION_NAME,
                ids            =[point_id],
                with_vectors   =True,
            )
            if results:
                dense_embeddings.append(results[0].vector)
                ordered_chunks.append(chunk)
        except Exception:
            pass

    state.all_chunks       = ordered_chunks
    state.dense_embeddings = np.array(dense_embeddings) if dense_embeddings else None
    print(f"  Dense vectors loaded: {len(ordered_chunks)} chunks")


# ---------------------------------------------------------------------------
# Lifespan  (D2 + D3 startup unchanged — D4 additions at the bottom)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")

    # ── D2 startup (unchanged) ────────────────────────────────────────────────
    state.run_card  = load_run_card()
    state.default_k = state.run_card.get("best_k", 10)
    alpha           = state.run_card.get("best_alpha", 0.5)

    print(f"  Loading model: {EMBED_MODEL}")
    state.model = load_model()

    state.qdrant = get_qdrant_client()
    ensure_collection(state.qdrant)

    state.mongo_db = get_mongo_db()
    ensure_indexes(state.mongo_db)

    try:
        from adaptation.online_learner import HybridWeightAdapter
        state.hybrid_adapter = HybridWeightAdapter(alpha=alpha)
        print(f"  HybridWeightAdapter warm-started with alpha={alpha}")
    except ImportError:
        print("  HybridWeightAdapter not found — using static alpha")
        state.hybrid_adapter = None

    print("  Loading chunks and building BM25 index...")
    state.all_chunks = list(state.mongo_db["chunks"].find({}, {"_id": 0}))
    if state.all_chunks:
        from retrieval.bm25_retriever import build_bm25_index
        state.bm25_index = build_bm25_index(state.all_chunks)
    else:
        state.bm25_index = None

    _build_dense_from_qdrant()

    # ── D3 startup additions ──────────────────────────────────────────────────
    # Neo4jStore connects to bolt://localhost:7687 using .env credentials.
    # TopicParser reuses state.model (already loaded above) — no extra cost.
    print("  Connecting to Neo4j and initialising TopicParser...")
    state.neo4j        = Neo4jStore()
    from graphrag.topic_router import build_topic_router

    state.topic_parser = TopicParser(state.neo4j, model=state.model)
    state.topic_router = build_topic_router(state.topic_parser)
    print("  Neo4j + TopicParser ready.")

    # ── D4 startup additions ──────────────────────────────────────────────────
    # Connect to local Ollama running the QLoRA-tuned Qwen2.5-1.5B model.
    # Non-fatal: if Ollama isn't running, /ask with llm="openrouter" still works.
    print("  Connecting to Ollama (tuned local LLM)...")
    try:
        state.local_llm = load_local_llm()
        print(f"  Local LLM ready: {state.local_llm['model']}")
    except Exception as e:
        print(f"  Local LLM unavailable: {e}")
        print(f"  /ask will only work with llm=\"openrouter\".")
        state.local_llm = None
    # ── end D4 additions ──────────────────────────────────────────────────────

    print("Startup complete.\n")
    yield
    print("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       ="PDF-Papers AI Agent",
    description ="Hybrid Retrieval + GraphRAG + QLoRA-tuned local LLM",
    version     ="0.4.0",   # bumped to D4
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
    top_k : int | None = None

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

# ── D3 + D4 request model ────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query     : str
    condition : Literal["graph_guided", "hybrid", "bm25_only"] = "graph_guided"
    llm       : Literal["openrouter", "tuned_local"] = "openrouter"


# ---------------------------------------------------------------------------
# Helper — embed query for /search  (D2, unchanged)
# ---------------------------------------------------------------------------

def embed_query_local(query: str) -> np.ndarray:
    """D2 helper — used only by /search."""
    return state.model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
    )


# ---------------------------------------------------------------------------
# POST /ingest  (D2, unchanged)
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        import pandas as pd
        paper_id_hint = os.path.splitext(file.filename or "unknown")[0]
        csv_path      = "data/papers_enriched.csv"
        csv_row       = None

        if os.path.exists(csv_path):
            df    = pd.read_csv(csv_path)
            match = df[df["paper_id"].astype(str) == paper_id_hint]
            if not match.empty:
                csv_row = match.iloc[0]

        doc = parse_pdf(tmp_path, csv_row=csv_row)
        if csv_row is None:
            doc["paper_id"] = paper_id_hint

        paper_id = doc["paper_id"]

        if paper_exists(state.mongo_db, paper_id):
            return IngestResponse(
                paper_id        =paper_id,
                title           =doc.get("title"),
                chunks_embedded =0,
                pages_parsed    =doc.get("page_count", 0),
                elapsed_seconds =round(time.time() - start, 2),
                status          ="skipped — already ingested",
            )

        chunks                   = chunk_document(doc)
        clean_chunks, embeddings = embed_chunks(chunks, state.model)

        if clean_chunks:
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
            insert_chunks(state.mongo_db, clean_chunks, doc)

        insert_paper(state.mongo_db, doc)
        insert_run_card(state.mongo_db, {
            "paper_id":        paper_id,
            "chunks_embedded": len(clean_chunks),
            "pages_parsed":    doc.get("page_count", 0),
            "model":           EMBED_MODEL,
        })

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
# POST /search  (D2, unchanged)
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

    top_k = request.top_k or state.default_k

    if state.hybrid_adapter is not None:
        weights = state.hybrid_adapter.get_weights()
        alpha   = weights.get("dense_weight", 0.5)
    else:
        weights = {"dense_weight": 0.5, "bm25_weight": 0.5}
        alpha   = 0.5

    query_vector = embed_query_local(request.query)

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
# POST /ask  (D3 — endpoint; D4 — adds llm switching)
# ---------------------------------------------------------------------------

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Full GraphRAG pipeline with optional condition + llm switching for ablation.

    condition = "graph_guided" (default) — full 14-step pipeline
    condition = "hybrid"                 — skip graph, use all chunks, hybrid alpha
    condition = "vector_only"            — skip graph, use all chunks, alpha=1.0

    llm = "openrouter" (default)         — hosted Llama-3.3-70B (free tier)
    llm = "tuned_local"                  — our QLoRA-tuned Qwen2.5-1.5B via Ollama
    """
    start = time.time()
    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if state.bm25_index is None or state.dense_embeddings is None:
        raise HTTPException(status_code=503, detail="No documents ingested yet.")

    # ── Step 1 · Embed query (once, reused everywhere) ───────────────────────
    qv = graphrag_embed_query(state.model, query)

    # ── Step 2 · Get alpha from online learner ───────────────────────────────
    alpha, weights = adaptive_alpha(state.hybrid_adapter)

    # ── Condition switch — graph_guided vs hybrid vs vector_only ─────────────
    if req.condition == "graph_guided":
        seed_ids = seed_search(query, state, alpha, qv, n_seed_papers=10)
        topics = state.topic_router.predict(query)
        subgraph = state.neo4j.select_subgraph(
            seed_paper_ids=seed_ids, topics=topics, max_papers=20
        )
        pool = expand_to_chunks(state.mongo_db, subgraph)
        pool = provenance_filter(state.mongo_db, pool)

        papers_in_subgraph = len(subgraph)
        seed_ids_out       = seed_ids
        topics_out         = topics

    else:
        pool = list(state.all_chunks)
        if req.condition == "bm25_only":
            alpha = 1.0
        papers_in_subgraph = 0
        seed_ids_out       = []
        topics_out         = []

    # ── Step 8 · Align embeddings ────────────────────────────────────────────
    aligned_pool, pool_embeddings = pool_embeddings_from_state(
        pool, state.all_chunks, state.dense_embeddings
    )

    # ── Step 9 · Rank ────────────────────────────────────────────────────────
    ranked = rank_pool(
        query, aligned_pool, state.model, pool_embeddings,
        alpha=alpha, k=8, query_vector=qv,
    )

    # ── Step 10 · Generate answer — D4 LLM switch ────────────────────────────
    if req.llm == "tuned_local":
        if state.local_llm is None:
            raise HTTPException(
                status_code=503,
                detail="Local LLM not loaded. Is Ollama running with pdfpapers-tuned?",
            )
        result = generate_answer_local(query, ranked, state.local_llm)
    else:
        result = generate_answer(query, ranked)

    # ── Step 11 · Safety post-check ──────────────────────────────────────────
    pin = source_pinning_check(
        result["answer"],
        ranked[: result["chunks_used"]],
    )

    # ── Step 12 · Judge ──────────────────────────────────────────────────────
    j_before = judge_answer(query, result["answer"], ranked)

    # ── Step 13 · Reflect ────────────────────────────────────────────────────
    reflect = reflect_answer(query, ranked, result, j_before)

    # ── Step 14 · Return ─────────────────────────────────────────────────────
    return {
        "answer":              reflect["revised_answer"],
        "citations":           reflect["revised_citations"],
        "safety":              pin,
        "judge_before":        reflect["initial_judge"],
        "judge_after":         reflect["revised_judge"],
        "improvement":         reflect["improvement"],
        "reflected":           reflect["reflection_triggered"],
        "condition":           req.condition,
        "llm":                 req.llm,                  # D4 — ablation label
        "llm_model":           result.get("model"),      # D4 — actual model name
        "seed_papers":         seed_ids_out,
        "topics_used":         topics_out,
        "papers_in_subgraph":  papers_in_subgraph,
        "pool_size":           len(pool),
        "alpha":               alpha,
        "elapsed_seconds":     round(time.time() - start, 2),
    }


class FeedbackRequest(BaseModel):
    query           : str
    helpful         : bool
    chunk_ids       : list[str] | None = None
    correct_topic   : str | None = None    # for the topic router

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    # Update the topic router (learner) — D1 online learning closure
    if state.topic_router is not None and req.correct_topic:
        state.topic_router.update(
            req.query,
            true_topic=req.correct_topic,
            helpful=req.helpful,
        )
        state.topic_router.save()   # persist across restarts

    # Bonus: also update the hybrid weight adapter from D1
    if state.hybrid_adapter is not None and req.chunk_ids:
        # signal: helpful=True means current alpha was good
        state.hybrid_adapter.update(req.query, signal=1.0 if req.helpful else 0.0)

    return {
        "status":    "ok",
        "new_alpha": state.hybrid_adapter.get_weights() if state.hybrid_adapter else None,
    }

# ---------------------------------------------------------------------------
# POST /rebuild-index  (D2, unchanged)
# ---------------------------------------------------------------------------

@app.post("/rebuild-index")
async def rebuild_index():
    """Rebuild BM25 + dense indexes after batch ingest."""
    start = time.time()
    build_retrieval_indexes()
    return {
        "status":          "ok",
        "chunks_indexed":  len(state.all_chunks),
        "elapsed_seconds": round(time.time() - start, 2),
    }


@app.get("/stats")
async def stats():
    """
    Surfaces the live state of D1's online learning components.
    Used in the demo to show 'ADWIN fired N times during this session'.
    """
    return {
        "topic_router_drift_events": (
            state.topic_router.drift_events
            if state.topic_router is not None else None
        ),
        "topic_router_accuracy": (
            state.topic_router.accuracy
            if state.topic_router is not None else None
        ),
        "hybrid_alpha": (
            state.hybrid_adapter.get_weights()
            if state.hybrid_adapter is not None else None
        ),
        "chunks_indexed": len(state.all_chunks),
    }

# ---------------------------------------------------------------------------
# GET /health  (D2 + D3 + D4 additions)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status"           : "ok",
        "version"          : "0.4.0-d4",
        "model"            : EMBED_MODEL,
        "collection"       : COLLECTION_NAME,
        "vector_dim"       : VECTOR_DIM,
        "chunks"           : len(state.all_chunks),
        "alpha"            : state.hybrid_adapter.get_weights() if state.hybrid_adapter else 0.5,
        "top_k"            : state.default_k,
        "run_card"         : state.run_card,
        # D3
        "neo4j_ready"      : state.neo4j is not None,
        "topic_parser"     : state.topic_parser is not None,
        # D4
        "local_llm_ready"  : state.local_llm is not None,
        "local_llm_model"  : state.local_llm["model"] if state.local_llm else None,
        "topic_router": state.topic_router is not None,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Important: do NOT use --reload (it restarts when eval scripts touch files)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
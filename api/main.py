"""
api/main.py
------------
FastAPI application — D2 retrieval stack + D3 GraphRAG pipeline.

Endpoints:
    POST /ingest        — upload PDF → parse → chunk → embed → store
    POST /search        — query → hybrid_search() → cited results
    POST /ask           — query → full GraphRAG pipeline → grounded answer  [D3]
    POST /rebuild-index — rebuild BM25 + dense indexes after batch ingest
    GET  /health        — liveness check

Startup:
    - Reads run_card.yaml for best_alpha, best_k, best_svd_dim
    - Loads embedding model once
    - Connects to Qdrant, MongoDB, Neo4j                                    [D3]
    - Loads all chunks from MongoDB + builds BM25 + dense indexes
    - Warm-starts HybridWeightAdapter with AutoML best_alpha
    - Initialises TopicParser for query → Neo4j topic mapping               [D3]
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

sys.path.insert(0, ".")
load_dotenv()

# ── D2 imports (unchanged) ────────────────────────────────────────────────────
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
from graphrag.answer  import generate_answer        # Abdullah's Step 4
from graphrag.safety  import (                      # Abdullah's safety layer
    provenance_filter,
    source_pinning_check,
)
from graphrag.judge   import judge_answer           # Abdullah's judge
from graphrag.reflect import reflect_answer         # Abdullah's reflect


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
# App state  (D2 fields unchanged — D3 fields added at bottom)
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
# Lifespan  (D2 startup unchanged — D3 additions at the bottom of startup)
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
    state.topic_parser = TopicParser(state.neo4j, model=state.model)
    print("  Neo4j + TopicParser ready.")
    # ── end D3 additions ──────────────────────────────────────────────────────

    print("Startup complete.\n")
    yield
    print("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       ="PDF-Papers AI Agent",
    description ="Hybrid Retrieval + GraphRAG with Online Learning",
    version     ="0.3.0",   # bumped to D3
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

# ── D3 request model ─────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str   # plain-text question from the user


# ---------------------------------------------------------------------------
# Helper — embed query for /search  (D2, unchanged)
# Note: /ask uses graphrag_embed_query() from graphrag.rank instead,
#       because that function accepts (model, query) and returns np.ndarray.
#       This helper below only accepts (query) and reads state internally.
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
# POST /ask  (D3 — new endpoint)
# ---------------------------------------------------------------------------

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Full GraphRAG pipeline — 14 steps.

    Plain-English summary of each step:

    1  Embed the query into a vector (done ONCE, reused everywhere)
    2  Get the current hybrid weight (alpha) from the online learner
    3  Seed search — quick scan of all chunks → 5 anchor paper_ids
    4  Topic parser — map the question words to Neo4j Topic node names
    5  Subgraph — ask the knowledge graph for papers related to those topics
    6  Expand — fetch all chunks belonging to those papers from MongoDB
    7  Safety pre-filter — drop any chunks with missing/bad provenance
    8  Align embeddings — look up pre-built vectors for the pool (no re-encoding)
    9  Rank — score chunks with hybrid BM25+dense+graph and keep top-8
    10 Generate answer — LLM answers using ONLY the ranked chunks, with [1][2] citations
    11 Safety post-check — verify every sentence in the answer has a valid citation
    12 Judge — break the answer into claims, check each is grounded in a chunk
    13 Reflect — if hallucinations found, LLM revises; then re-judge
    14 Return — final answer, citations, safety report, judge scores, improvement delta
    """
    start = time.time()
    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if state.bm25_index is None or state.dense_embeddings is None:
        raise HTTPException(status_code=503, detail="No documents ingested yet.")

    # ── Step 1 · Embed query ─────────────────────────────────────────────────
    # We embed ONCE here and pass qv around — never embed again.
    # graphrag_embed_query() applies the BGE query prefix automatically.
    qv = graphrag_embed_query(state.model, query)

    # ── Step 2 · Get adaptive alpha from online learner ──────────────────────
    # adaptive_alpha() reads state.hybrid_adapter and returns a float.
    # If no adapter loaded it returns the default (0.5).
    alpha, weights = adaptive_alpha(state.hybrid_adapter)

    # ── Step 3 · Seed search ─────────────────────────────────────────────────
    # Scans ALL 7,979 chunks with hybrid search (k=30), extracts the unique
    # paper_ids from the top results, returns the best 5.
    # These are the "anchor" papers we'll expand from in the graph.
    seed_ids = seed_search(query, state, alpha, qv, n_seed_papers=5)

    # ── Step 4 · Topic parsing ───────────────────────────────────────────────
    # TopicParser looks at the query text and maps it to actual Neo4j
    # Topic node names (e.g. "knowledge_representation", "planning_and_scheduling").
    # This lets us filter the graph by topic, not just by paper similarity.
    topics = state.topic_parser.parse(query)

    # ── Step 5 · Subgraph selection (GraphRAG Step 1) ────────────────────────
    # select_subgraph() runs Cypher queries in Neo4j using:
    #   - seed_ids  → papers we know are relevant
    #   - topics    → topic nodes to expand through
    # Returns up to 20 papers with graph-relevance scores and reasons.
    subgraph = state.neo4j.select_subgraph(
        seed_paper_ids=seed_ids,
        topics=topics,
        max_papers=20,
    )

    # ── Step 6 · Expand to chunks (GraphRAG Step 2) ──────────────────────────
    # Takes the paper_ids from the subgraph and fetches ALL their chunks
    # from MongoDB. Each chunk gets a graph_score and graph_reasons attached.
    # This is our "candidate pool" — typically 200–800 chunks.
    pool = expand_to_chunks(state.mongo_db, subgraph)

    # ── Step 7 · Provenance filter (Safety ★ — pre-LLM) ─────────────────────
    # Drops any chunk whose paper has missing or unverifiable provenance
    # in MongoDB. We check BEFORE sending to the LLM — we never want the
    # model citing something we can't verify.
    pool = provenance_filter(state.mongo_db, pool)

    # ── Step 8 · Align pool embeddings ───────────────────────────────────────
    # At startup we loaded ALL embeddings from Qdrant into state.dense_embeddings.
    # Here we look up which rows correspond to our pool chunks.
    # No re-encoding — just an index lookup. Fast.
    aligned_pool, pool_embeddings = pool_embeddings_from_state(
        pool, state.all_chunks, state.dense_embeddings
    )

    # ── Step 9 · Rank (GraphRAG Step 3) ──────────────────────────────────────
    # rank_pool() scores every chunk in the pool using:
    #   BM25 score   (keyword match)
    #   dense score  (embedding similarity, using our pre-computed qv)
    #   graph score  (how central this chunk's paper was in the subgraph)
    # Blended with alpha weighting, keeps top-8.
    ranked = rank_pool(
        query,
        aligned_pool,
        state.model,
        pool_embeddings,
        alpha=alpha,
        k=8,
        query_vector=qv,
    )

    # ── Step 10 · Generate answer (GraphRAG Step 4) ──────────────────────────
    # generate_answer() sends the top-8 chunks to the LLM (via OpenRouter).
    # The LLM ONLY sees those chunks — not the whole corpus.
    # It must use [1][2] style inline citations tied to the chunk list.
    result = generate_answer(query, ranked)

    # ── Step 11 · Source pinning check (Safety ★ — post-LLM) ────────────────
    # Checks every sentence ≥60 chars in the answer has a [N] citation
    # that maps to a real chunk in ranked. Flags:
    #   out_of_range     — citation [N] where N > chunks_used
    #   uncited_sentences — long sentences with no citation at all
    pin = source_pinning_check(
        result["answer"],
        ranked[: result["chunks_used"]],
    )

    # ── Step 12 · Judge answer (Quality ✦) ───────────────────────────────────
    # judge_answer() extracts every factual claim from the answer and checks
    # each one against the ranked chunks. Returns:
    #   faithfulness_score   — float, grounded_claims / total_claims
    #   ungrounded_claims    — list of claim strings that aren't in ANY chunk
    j_before = judge_answer(query, result["answer"], ranked)

    # ── Step 13 · Reflect and revise (Quality ✦) ─────────────────────────────
    # reflect_answer() takes the ungrounded_claims and passes them BACK to
    # the LLM as a critique: "these claims are not supported, fix them."
    # The LLM rewrites the answer using only the context.
    # Then judge_answer runs again on the revised answer.
    # If the initial answer was already FULLY_GROUNDED → skipped entirely
    # (reflected=False, no extra API call made).
    reflect = reflect_answer(query, ranked, result, j_before)

    # ── Step 14 · Return ─────────────────────────────────────────────────────
    return {
        # Final answer (revised if reflection was triggered)
        "answer":       reflect["revised_answer"],
        "citations":    reflect["revised_citations"],

        # Safety: is_clean bool + any uncited/out-of-range sentences
        "safety":       pin,

        # Faithfulness + relevance scores before reflection
        "judge_before": reflect["initial_judge"],

        # Faithfulness + relevance scores after reflection (same if skipped)
        "judge_after":  reflect["revised_judge"],

        # Delta: faithfulness_delta, relevance_delta, claims_fixed
        "improvement":  reflect["improvement"],

        # Was reflection needed? False = answer was clean on first try
        "reflected":    reflect["reflection_triggered"],

        # Debug / transparency fields
        "seed_papers":      seed_ids,
        "topics_used":      topics,
        "papers_in_subgraph": len(subgraph),
        "pool_size":        len(pool),
        "alpha":            alpha,
        "elapsed_seconds":  round(time.time() - start, 2),
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


# ---------------------------------------------------------------------------
# GET /health  (D2 + D3 additions)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status"      : "ok",
        "version"     : "0.3.0-d3",
        "model"       : EMBED_MODEL,
        "collection"  : COLLECTION_NAME,
        "vector_dim"  : VECTOR_DIM,
        "chunks"      : len(state.all_chunks),
        "alpha"       : state.hybrid_adapter.get_weights() if state.hybrid_adapter else 0.5,
        "top_k"       : state.default_k,
        "run_card"    : state.run_card,
        # D3 additions
        "neo4j_ready" : state.neo4j is not None,
        "topic_parser": state.topic_parser is not None,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Important: do NOT use --reload (it restarts when eval scripts touch files)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
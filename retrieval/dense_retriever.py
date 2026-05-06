import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_path: str = "data/sample_chunks.json") -> list[dict]:
    """
    Loads chunks from JSON file into memory.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks


def build_dense_index(
    chunks: list[dict],
    model_name: str = "BAAI/bge-small-en",
    normalize: bool = True
) -> tuple[SentenceTransformer, np.ndarray]:
    """
    Converts all chunk texts into embedding vectors.
    Returns:
      model      → needed later to encode queries
      embeddings → (50 x 384) numpy matrix, one row per chunk
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

    print(f"✅ Dense index built — shape: {embeddings.shape}")
    return model, embeddings


def dense_search(
    query: str,
    chunks: list[dict],
    model: SentenceTransformer,
    embeddings: np.ndarray,
    k: int = 5,
    normalize: bool = True,
    query_vector: np.ndarray | None = None  # ← pre-projected vector from Optuna
) -> list[dict]:
    """
    Searches using cosine similarity between query and chunk embeddings.
    If query_vector is provided (e.g. already SVD-projected + normalized
    by the Optuna trial), it is used directly — skipping encode + normalize.
    """
    if query_vector is None:
        # standard path: encode fresh and normalize
        query_vector = model.encode(query, convert_to_numpy=True)
        if normalize:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

    # embeddings: (n, d)  query_vector: (d,)  → scores: (n,)
    scores = embeddings @ query_vector

    scored_chunks = [
        {**chunks[i], "score": float(scores[i])}
        for i in range(len(chunks))
    ]

    results = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[:k]
    return results
    

# retrieval/dense_retriever.py
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

    Steps:
      1. Load the embedding model
      2. Encode all chunk texts → matrix of shape (num_chunks, 384)
      3. Optionally normalize vectors

    Returns:
      model      → needed later to encode queries
      embeddings → (50 x 384) numpy matrix, one row per chunk
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # encode all texts at once → shape: (50, 384)
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,    # shows a progress bar
        convert_to_numpy=True      # returns numpy array
    )

    if normalize:
        # normalize each vector to length 1
        # after this, cosine similarity = simple dot product (faster)
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
    normalize: bool = True
) -> list[dict]:
    """
    Searches using cosine similarity between query and chunk embeddings.

    Steps:
      1. Encode query → query vector (384 numbers)
      2. Compute cosine similarity vs all chunk vectors
      3. Sort by similarity, return top-k
    """
    # encode the query → shape: (384,)
    query_vector = model.encode(
        query,
        convert_to_numpy=True
    )

    if normalize:
        # normalize query vector same way as chunks
        query_vector = query_vector / np.linalg.norm(query_vector)

    # cosine similarity = dot product (since both are normalized)
    # scores shape: (50,) — one score per chunk
    scores = embeddings @ query_vector

    # pair each chunk with its score
    scored_chunks = [
        {**chunks[i], "score": float(scores[i])}
        for i in range(len(chunks))
    ]

    # sort by score descending, return top-k
    results = sorted(
        scored_chunks,
        key=lambda x: x["score"],
        reverse=True
    )[:k]

    return results

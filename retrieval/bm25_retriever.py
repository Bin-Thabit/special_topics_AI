# retrieval/bm25_retriever.py
import json
from rank_bm25 import BM25Okapi


def load_chunks(chunks_path: str = "data/sample_chunks.json") -> list[dict]:
    """
    Loads chunks from JSON file into memory.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Builds BM25 index from all chunk texts.

    BM25 needs each text as a list of words:
    "BERT is a model" → ["bert", "is", "a", "model"]
    """
    tokenized_corpus = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]
    index = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25 index built over {len(chunks)} chunks")
    return index


def bm25_search(
    query: str,
    chunks: list[dict],
    index: BM25Okapi,
    k: int = 5
) -> list[dict]:
    """
    Searches BM25 index for the query.
    Returns top-k chunks with their scores.
    """
    # tokenize query same way as corpus
    tokenized_query = query.lower().split()

    # score every chunk
    scores = index.get_scores(tokenized_query)

    # attach score to each chunk
    scored_chunks = [
        {**chunks[i], "score": float(scores[i])}
        for i in range(len(chunks))
    ]

    # sort by score, return top-k
    results = sorted(
        scored_chunks,
        key=lambda x: x["score"],
        reverse=True
    )[:k]

    return results
    
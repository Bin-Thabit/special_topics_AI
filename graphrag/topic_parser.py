"""
graphrag/topic_parser.py
------------------------
Map a free-text question to the Topic names that ACTUALLY exist in Neo4j,
so select_subgraph()'s branch (d) ('topic_match') fires with real strings.

Why embeddings, not keyword matching?
  Topic names are snake_case labels ("automated_reasoning") that never appear
  verbatim in a user's question. We embed the question and every topic with the
  SAME bge model used during ingestion, then keep the closest topics by cosine
  similarity. The topic list is pulled LIVE from the graph, so it can never go
  stale relative to what was seeded.

Run:
    python -m graphrag.topic_parser
"""
from __future__ import annotations

import os
import logging

import numpy as np
from dotenv import load_dotenv

from stores.neo4j_store import Neo4jStore

load_dotenv()
log = logging.getLogger(__name__)

# BGE prefixes — must match the D2 ingestion convention.
Q_PREFIX = "Represent this sentence for searching relevant passages: "
T_PREFIX = "Represent this passage: "


class TopicParser:
    """
    Turns a question into a short list of real Topic names.

    Pass in an already-loaded SentenceTransformer (e.g. the one the API loads at
    startup) to avoid loading the model twice; otherwise it loads from .env.
    Topic embeddings are computed once and cached — call refresh_topics() if you
    re-seed the graph with new topics.
    """

    def __init__(self, store: Neo4jStore, model=None, model_name: str | None = None):
        self.store = store
        self.model = model or self._load_model(model_name)
        self._topics: list[str] = []
        self._topic_vecs: np.ndarray | None = None
        self.refresh_topics()

    @staticmethod
    def _load_model(model_name: str | None):
        from sentence_transformers import SentenceTransformer

        name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        log.info("TopicParser loading embedding model: %s", name)
        return SentenceTransformer(name)

    @staticmethod
    def _humanize(topic: str) -> str:
        # "automated_reasoning" -> "automated reasoning"
        return topic.replace("_", " ").strip()

    def refresh_topics(self) -> None:
        """Reload the topic list from Neo4j and recompute their embeddings."""
        rows = self.store.run_query("MATCH (t:Topic) RETURN t.name AS name ORDER BY name")
        self._topics = [r["name"] for r in rows if r.get("name")]
        if not self._topics:
            log.warning("No Topic nodes found in graph — parse() will return []")
            self._topic_vecs = None
            return
        texts = [T_PREFIX + self._humanize(t) for t in self._topics]
        # normalize so a plain dot product equals cosine similarity
        self._topic_vecs = self.model.encode(texts, normalize_embeddings=True)
        log.info("TopicParser cached %d topic embeddings", len(self._topics))

    def parse(self, question: str, top_k: int = 2, min_score: float = 0.62) -> list[str]:
        """
        Return up to `top_k` topic names whose similarity to the question is at
        least `min_score`. Returns [] if nothing clears the bar — that's the
        correct answer for out-of-taxonomy questions, and select_subgraph()
        still works off the seed papers alone.

        NOTE: bge cosine scores cluster in a narrow high band (~0.5-0.75 even
        for weak matches), so `min_score` is model-specific — calibrate it with
        parse_scored() on real questions. ~0.62 fits this corpus: it keeps clear
        hits (0.65+) and rejects no-match questions (which top out around 0.58).
        """
        scored = self.parse_scored(question, top_k=top_k)
        return [hit["topic"] for hit in scored if hit["score"] >= min_score]

    def parse_scored(self, question: str, top_k: int = 5) -> list[dict]:
        """Like parse() but returns {topic, score} pairs — handy for calibrating min_score."""
        if not question or self._topic_vecs is None:
            return []
        q = self.model.encode([Q_PREFIX + question], normalize_embeddings=True)[0]
        sims = self._topic_vecs @ q  # cosine, since both sides are normalized
        order = np.argsort(-sims)[:top_k]
        return [{"topic": self._topics[i], "score": round(float(sims[i]), 3)} for i in order]


if __name__ == "__main__":
    # Smoke test: prints scores so you can pick a sensible min_score.
    logging.basicConfig(level=logging.INFO)
    with Neo4jStore() as store:
        parser = TopicParser(store)
        questions = [
            "How does reinforcement learning handle long-horizon planning?",
            "What methods exist for reasoning over knowledge graphs?",
            "Tell me about video generation and motion control models",
        ]
        for q in questions:
            print(f"\nQ: {q}")
            for hit in parser.parse_scored(q, top_k=5):
                print(f"   {hit['score']:.3f}  {hit['topic']}")
            print("   -> parse():", parser.parse(q))
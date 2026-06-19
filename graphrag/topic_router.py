"""
graphrag/topic_router.py — D1 → D3 topic prediction bridge.

Routes a free-text query to the cs.AI subtopic names that exist in Neo4j,
so select_subgraph() branch (d) ('topic_match') fires with real strings.

Two prediction sources, used as a primary + fallback pair:

  1. PRIMARY  — QueryTopicLearner from D1.
                The trained online Naive Bayes classifier with ADWIN drift
                detection. Reflects learned patterns from feedback over time.

  2. FALLBACK — TopicParser (embedding-based).
                Cosine similarity between the query and each topic's
                centroid embedding. Independent of the learner's training
                history, so it works even when the learner is uncertain
                (cold start, post-drift, or low-confidence prediction).

When primary returns at least one topic above the confidence threshold, we
use it. Otherwise we fall back to the parser. This combines learned patterns
(D1 deliverable) with semantic similarity (D3) as a safety net.

Online updates: /feedback calls update() to keep the learner adapting at
runtime. save() persists the updated model so adaptation isn't lost on
process restart.

Usage in /ask
-------------
  router = TopicRouter(learner, parser)               # at startup
  topics = router.predict(query)                       # in /ask
  router.update(query, true_topic, predicted, helpful) # in /feedback
"""
from __future__ import annotations

import logging
from typing import Optional

from adaptation.online_learner import (
    QueryTopicLearner,
    FeedbackEvent,
    TOPICS,
)
from graphrag.topic_parser import TopicParser

log = logging.getLogger(__name__)

# Default confidence floor. Tuned around the learner's observed per-topic
# recall band — see scripts/retrain_query_classifier.py output. Topics with
# strong recall (>=0.8) cleanly exceed this; weak topics (<0.4) need the
# fallback. Override per call via min_confidence=.
DEFAULT_MIN_CONFIDENCE = 0.25


class TopicRouter:
    """
    Primary-with-fallback topic prediction for select_subgraph().

    Args
      learner             : a trained QueryTopicLearner (load_model from pkl).
      parser              : a TopicParser bound to the live Neo4j graph.
      min_confidence      : minimum learner probability for a topic to be kept.
      parser_min_score    : minimum cosine score for parser-returned topics
                            (bge scores cluster in a narrow band — 0.62 is
                            strict and rejects out-of-taxonomy questions
                            cleanly; lower to ~0.55 to accept weaker matches).
      corroborate_learner : when True, run the parser even when the learner
                            cleared its threshold. If the parser strongly
                            disagrees with the learner's top pick, prefer the
                            parser. Catches the over-firing failure mode on
                            high-recall but loose-fitting topics.
      max_topics          : hard cap on topics returned (1-3 is the sweet spot).
      always_use_fallback : if True, also run the parser and UNION its picks
                            with the learner's (use for ablation; off by default).
    """

    def __init__(
        self,
        learner: QueryTopicLearner,
        parser: TopicParser,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        parser_min_score: float = 0.55,
        corroborate_learner: bool = True,
        max_topics: int = 3,
        always_use_fallback: bool = False,
    ):
        self.learner = learner
        self.parser  = parser
        self.min_confidence = float(min_confidence)
        self.parser_min_score = float(parser_min_score)
        self.corroborate_learner = bool(corroborate_learner)
        self.max_topics = int(max_topics)
        self.always_use_fallback = bool(always_use_fallback)

        # Quick sanity check: the learner's TOPICS list and the parser's live
        # topic list should overlap, otherwise branch (d) of select_subgraph
        # won't match anything. Warn but don't fail — graph might use a
        # superset (you saw 20 nodes vs 12 in the D2 summary).
        graph_topics = set(self.parser._topics) if self.parser._topics else set()
        if graph_topics and not (set(TOPICS) & graph_topics):
            log.warning(
                "TopicRouter: learner TOPICS and Neo4j Topic nodes do not overlap. "
                "Branch (d) of select_subgraph may match zero papers."
            )

    # ------------------------------------------------------------------ predict
    def predict(
        self,
        query: str,
        min_confidence: float | None = None,
        max_topics: int | None = None,
        return_scored: bool = False,
    ) -> list[str] | dict:
        """
        Return topic name(s) for this query.

        Routing logic:
          1. Ask the learner. Keep topics with prob >= min_confidence,
             up to max_topics, sorted by confidence.
          2. If the learner gave none above threshold, fall back to the
             parser. (Or always union if always_use_fallback=True.)

        Args
          return_scored : if True, return a dict with source, topics, scores,
                          confidence, and parser_used — useful for the /ask
                          response and the D3 ablation.

        Returns
          list[str] of topic names by default, or the diagnostic dict.
        """
        min_conf = self.min_confidence if min_confidence is None else float(min_confidence)
        max_t    = self.max_topics if max_topics is None else int(max_topics)

        # --- 1. learner (primary) ----------------------------------------
        learner_topics: list[tuple[str, float]] = []
        confidence = 0.0
        try:
            result = self.learner.predict(query)
            # probabilities are already sorted desc by predict()
            probs = result.get("probabilities", {})
            if probs:
                confidence = max(probs.values())
                learner_topics = [
                    (t, float(p)) for t, p in probs.items()
                    if p >= min_conf
                ][:max_t]
        except Exception as e:
            log.warning("TopicRouter: learner.predict failed (%s) — falling back", e)

        # --- 2. fallback or corroboration --------------------------------
        # We call the parser if:
        #   - the learner returned nothing above threshold (classic fallback)
        #   - OR always_use_fallback is on (ablation/union mode)
        #   - OR corroborate_learner is on (sanity-check the learner's pick)
        parser_used = False
        parser_topics: list[tuple[str, float]] = []
        should_call_parser = (
            not learner_topics
            or self.always_use_fallback
            or self.corroborate_learner
        )
        if should_call_parser:
            parser_used = True
            try:
                scored = self.parser.parse_scored(query, top_k=max_t)
                parser_topics = [
                    (s["topic"], float(s["score"])) for s in scored
                    if s["score"] >= self.parser_min_score
                ]
            except Exception as e:
                log.warning("TopicRouter: parser call failed (%s)", e)
                parser_used = False

        # Corroboration override: when learner fired alone with a single
        # pick but the parser disagrees AND the parser's top score is at
        # least 0.60, trust the parser. Catches the high-recall topic
        # over-firing pattern (e.g. learner says ai_ethics_and_safety on a
        # LoRA question because that topic dominates its priors).
        overridden = False
        if (
            self.corroborate_learner
            and learner_topics
            and parser_topics
            and not self.always_use_fallback
        ):
            learner_top = learner_topics[0][0]
            parser_top, parser_score = parser_topics[0]
            if parser_top != learner_top and parser_score >= 0.60:
                log.info(
                    "TopicRouter: parser overrode learner — learner=%s (%.2f) "
                    "vs parser=%s (%.2f)",
                    learner_top, learner_topics[0][1], parser_top, parser_score,
                )
                learner_topics = []   # drop learner, parser wins
                overridden = True

        # --- 3. merge -----------------------------------------------------
        # Learner wins on duplicates (its confidence is what we keep).
        merged: dict[str, float] = {}
        for t, s in learner_topics:
            merged[t] = s
        for t, s in parser_topics:
            merged.setdefault(t, s)

        ordered = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:max_t]
        topic_names = [t for t, _ in ordered]

        if not return_scored:
            return topic_names

        if overridden:
            source = "parser (override)"
        elif learner_topics and parser_topics:
            source = "learner+parser"
        elif parser_used and parser_topics:
            source = "parser"
        elif learner_topics:
            source = "learner"
        else:
            source = "none"

        return {
            "topics":           topic_names,
            "scored":           ordered,
            "source":           source,
            "confidence":       round(confidence, 4),
            "parser_used":      parser_used,
            "overridden":       overridden,
            "min_confidence":   min_conf,
            "parser_min_score": self.parser_min_score,
        }

    # ------------------------------------------------------------------ update
    def update(
        self,
        query: str,
        true_topic: str,
        predicted_topic: str | None = None,
        helpful: bool = True,
    ) -> dict:
        """
        Online learning update from /feedback.

        The learner takes a FeedbackEvent which needs the user-confirmed true
        topic, what we originally predicted, and whether the answer was helpful.
        If predicted_topic is omitted, we re-run predict() to recover it (which
        is what /feedback usually has to do, since it only sees the query +
        helpful flag).

        Returns the learner's per-step report (accuracy, drift status, etc.).
        """
        if predicted_topic is None:
            predicted_topic = self.learner.predict(query).get("topic", "other")

        event = FeedbackEvent(
            query=query,
            predicted_topic=predicted_topic,
            true_topic=true_topic,
            helpful=bool(helpful),
        )
        return self.learner.learn_from_feedback(event)

    # -------------------------------------------------------------------- save
    def save(self, path: str = "data/query_classifier.pkl") -> None:
        """Persist the (possibly updated) learner so /feedback gains survive restarts."""
        self.learner.save_model(path)


# --------------------------------------------------------------------- factory
def build_topic_router(
    parser: TopicParser,
    learner_path: str = "data/query_classifier.pkl",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> TopicRouter:
    """
    Convenience builder for the API startup. Loads the pickled learner;
    falls back to a fresh QueryTopicLearner if the file is missing
    (the parser-only path will dominate until training data accumulates).
    """
    try:
        learner = QueryTopicLearner.load_model(learner_path)
        log.info("TopicRouter: loaded trained learner from %s", learner_path)
    except (FileNotFoundError, OSError) as e:
        log.warning(
            "TopicRouter: no trained learner at %s (%s) — using fresh instance. "
            "Parser fallback will carry most predictions until /feedback warms it up.",
            learner_path, e,
        )
        learner = QueryTopicLearner()
    return TopicRouter(learner, parser, min_confidence=min_confidence)


# ---------------------------------------------------------------------- smoke
if __name__ == "__main__":
    # Run from project root: python -m graphrag.topic_router
    logging.basicConfig(level=logging.INFO)

    from stores.neo4j_store import Neo4jStore

    with Neo4jStore() as store:
        parser = TopicParser(store)
        router = build_topic_router(parser)

        queries = [
            "How does reinforcement learning handle long-horizon planning?",
            "What methods exist for reasoning over knowledge graphs?",
            "Tell me about video generation and motion control models",
            "Explain non-monotonic logic and default reasoning",
        ]
        for q in queries:
            print(f"\nQ: {q}")
            info = router.predict(q, return_scored=True)
            print(f"   source     : {info['source']}")
            print(f"   confidence : {info['confidence']}")
            print(f"   topics     : {info['topics']}")
            for t, s in info["scored"]:
                print(f"     {s:>6.3f}  {t}")
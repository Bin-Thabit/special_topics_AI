"""
adaptation/online_learner.py
-----------------------------
Online query-to-topic classifier for the PDF-Papers AI Agent.

PURPOSE
-------
When a user submits a query to the agent (e.g. "how does attention work?"),
we classify it into one of 9 research topics BEFORE searching the corpus.
Knowing the topic helps the retrieval system search smarter in D2/D3.

CLASSES
-------
QueryTopicLearner   : classifies a query into a research topic
                      updates incrementally on every example
HybridWeightAdapter : adapts BM25 vs dense fusion weight
                      from user helpful/not-helpful feedback

Both use ADWIN to detect distribution shifts and reset automatically.

WHY ONLINE LEARNING?
--------------------
A normal sklearn classifier trains once on a fixed dataset.
An online learner updates itself after EVERY query it sees.
This means it adapts as users interests shift over time —
no retraining, no storing all data in memory.
River is the Python library built for this exact use case.

PIPELINE (QueryTopicLearner)
----------------------------
raw query text
    └─► BagOfWords              : word count dict
    └─► MultinomialNB           : incremental Naive Bayes classifier

WHY MultinomialNB OVER SoftmaxRegression?
-----------------------------------------
MultinomialNB is mathematically designed for sparse word count features.
It learns the probability of each word given each topic.
In testing it achieved 0.77 accuracy vs 0.38 for SoftmaxRegression
on the same data — making it a much better fit for short query text.

PREQUENTIAL EVALUATION
----------------------
Rule: always PREDICT first, then LEARN.
This ensures accuracy is always measured on unseen data —
identical to real deployment conditions.

ADWIN DRIFT DETECTION
---------------------
ADWIN watches the stream of prediction errors (0=correct, 1=wrong).
When the recent error rate is statistically higher than the older window,
it signals concept drift. We rebuild the pipeline from scratch on drift.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from river import drift, feature_extraction, metrics, naive_bayes, utils

logger = logging.getLogger(__name__)


# ── Topic labels ──────────────────────────────────────────────────────────────
# These are the cs.AI sub-areas your model classifies queries into.
# You can extend this list to match your actual corpus topics.
TOPICS = [
    "reinforcement_learning",
    "computer_vision",
    "natural_language_processing",
    "knowledge_representation",
    "planning_search",
    "other",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class QueryFeedback:
    """
    One labelled interaction from a user.

    query    : the raw text the user typed into the agent
    topic    : the correct topic label
               (comes from user feedback or manual annotation)
    helpful  : did the user mark the answer as helpful? (y/n)
               used by HybridWeightAdapter to adapt fusion weight
    timestamp: when this interaction happened
    """
    query: str
    topic: str
    helpful: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeedbackEvent:
    """
    One piece of user feedback for hybrid weight adaptation.

    query           : the raw query text
    predicted_topic : what the learner predicted
    true_topic      : the correct topic label
    helpful         : did the user mark the answer as helpful?
    timestamp       : when this happened
    """
    query: str
    predicted_topic: str
    true_topic: str
    helpful: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class LearnerState:
    """
    A snapshot of the learner metrics at one point in time.
    Saved every 10 samples and used to draw the prequential chart.

    step             : how many samples have been processed so far
    accuracy         : cumulative prequential accuracy at this step
    rolling_accuracy : accuracy over the last 50 queries (recent performance)
    drift_detected   : was drift detected at this exact step?
    resets           : total number of classifier resets so far
    """
    step: int
    accuracy: float
    rolling_accuracy: float
    drift_detected: bool
    resets: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ── QueryTopicLearner ─────────────────────────────────────────────────────────

class QueryTopicLearner:
    """
    Incremental query-to-topic classifier with ADWIN drift detection.

    Pipeline
    --------
    BagOfWords + MultinomialNB (no Pipeline wrapper — gives more control
    over when each step learns vs transforms, important for NB correctness)

    Prequential protocol (test-then-train)
    --------------------------------------
    1. transform query → word counts
    2. predict BEFORE learning (honest evaluation)
    3. update accuracy metrics (cumulative + rolling)
    4. feed error to ADWIN
    5. if drift → rebuild pipeline from scratch
    6. learn on the true label
    7. log to JSONL + append to history

    Usage
    -----
    learner = QueryTopicLearner()

    # inference (no learning)
    result = learner.predict("how does attention work?")

    # learning from a labelled query
    fb = QueryFeedback(query="how does attention work?",
                       topic="transformers", helpful=True)
    record = learner.learn_one(fb)

    # learning from a user feedback event
    event = FeedbackEvent(query="...", predicted_topic="...",
                          true_topic="transformers", helpful=True)
    record = learner.learn_from_feedback(event)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        adwin_delta: float = 0.002,
        log_path: str = "data/prequential_log.jsonl",
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        alpha       : MultinomialNB smoothing parameter
                      higher = smoother probability estimates
                      1.0 is the standard Laplace smoothing default
        adwin_delta : ADWIN sensitivity
                      lower = more sensitive = fires more easily
                      0.002 is the standard starting point
        log_path    : where to write the real-time JSONL log
        seed        : random seed for reproducibility
        """
        self.alpha = alpha
        self.adwin_delta = adwin_delta
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Build the model
        self._build_pipeline()

        # ADWIN drift detector — watches prediction error rate
        self.adwin = drift.ADWIN(delta=adwin_delta)

        # Cumulative prequential accuracy
        self.prequential_acc = metrics.Accuracy()

        # Rolling accuracy over last 50 queries
        # More honest than cumulative for an online learner —
        # shows what the model is doing RIGHT NOW, not historical average
        self.rolling_acc = utils.Rolling(metrics.Accuracy(), window_size=50)
        
        # Per-topic recall: when true label IS this topic, how often correct?
        # Answers: "which topics does the model struggle to recognize?"
        self.per_topic_recall = {
            topic: metrics.Recall() for topic in TOPICS
        }

        # Per-topic precision: when we PREDICT this topic, how often correct?
        # Answers: "which topics does the model over-predict or confuse?"
        self.per_topic_precision = {
            topic: metrics.Precision() for topic in TOPICS
        }

        # Per-topic helpfulness tracking
        # Tracks how often users mark answers as helpful per topic
        # Low helpfulness rate = retrieval is failing for that topic
        # Flows into D2 — topics with low helpfulness need more PDFs
        self.topic_helpfulness: dict[str, dict[str, int]] = {
            topic: {"helpful": 0, "not_helpful": 0}
            for topic in TOPICS
        }

        # Counters
        self.n_samples: int = 0
        self.n_resets: int = 0
        self.drift_detected: bool = False

        # Drift step log — steps where ADWIN fired
        self.drift_points: list[int] = []

        # History snapshots for the prequential plot (every 10 steps)
        self.history: list[LearnerState] = []

    def _build_pipeline(self) -> None:
        """
        Build (or rebuild) the BagOfWords + MultinomialNB pipeline.

        We keep BagOfWords and MultinomialNB as separate objects
        instead of using compose.Pipeline. This gives us explicit
        control over when each component learns vs transforms —
        important because NB needs to transform before learning
        but also needs to update its vocabulary separately.

        Called once at init and again on every drift reset.
        """
        self.bow = feature_extraction.BagOfWords()
        self.model = naive_bayes.MultinomialNB(alpha=self.alpha)

    def predict(self, query: str) -> dict:
        """
        Predict the topic of a query WITHOUT updating the model.
        Used at inference time when a user submits a question to the agent.

        Returns
        -------
        dict with predicted topic and probability scores sorted by confidence
        """
        x = self.bow.transform_one(query)

        try:
            topic = self.model.predict_one(x)
            probas = self.model.predict_proba_one(x)
        except Exception:
            topic = None
            probas = {}

        # Handle cold start — model returns None before any training
        if not topic:
            topic = "other"
        if not probas:
            probas = {t: round(1.0 / len(TOPICS), 4) for t in TOPICS}

        return {
            "query": query,
            "topic": topic,
            # Sort by probability descending so highest confidence is first
            "probabilities": {
                k: round(v, 4)
                for k, v in sorted(
                    probas.items(), key=lambda item: item[1], reverse=True
                )
            },
        }

    def learn_one(self, feedback: QueryFeedback) -> dict:
        """
        Core prequential update — called for every new labelled query.

        Order of operations (fixed — do NOT reorder):
        1. transform  → word counts via BagOfWords
        2. predict    → BEFORE learning (prequential rule)
        3. evaluate   → update cumulative + rolling accuracy
        4. drift      → feed error to ADWIN, rebuild if drift detected
        5. learn      → update MultinomialNB weights + BagOfWords vocab
        6. log        → append to JSONL + history snapshot

        Parameters
        ----------
        feedback : QueryFeedback with query text and correct topic label

        Returns
        -------
        dict with prediction result, both accuracy metrics, drift status
        """
        query = feedback.query
        y = feedback.topic

        # ── 1. Transform ──────────────────────────────────────────────────────
        x = self.bow.transform_one(query)

        # ── 2. Predict BEFORE learning ────────────────────────────────────────
        pred = self.model.predict_one(x)
        if pred is None:
            pred = "other"   # cold start fallback

        # ── 3. Evaluate ───────────────────────────────────────────────────────
        correct = int(pred == y)
        self.prequential_acc.update(y, pred)
        self.rolling_acc.update(y, pred)
        # Update per-topic recall and precision
        # River's Recall and Precision are binary metrics so we pass
        # True/False indicating whether this sample belongs to each topic
        for topic in TOPICS:
            self.per_topic_recall[topic].update(
                y_true=(y == topic),       # is the true label this topic?
                y_pred=(pred == topic),    # did we predict this topic?
            )
            self.per_topic_precision[topic].update(
                y_true=(y == topic),
                y_pred=(pred == topic),
            )

        # ── 4. Drift detection ────────────────────────────────────────────────
        # Feed error signal: 1 = wrong prediction, 0 = correct
        self.adwin.update(1 - correct)
        self.drift_detected = self.adwin.drift_detected

        if self.drift_detected:
            logger.warning(
                "ADWIN drift detected at step %d — rebuilding pipeline.",
                self.n_samples,
            )
            self.drift_points.append(self.n_samples)
            self._handle_drift()
            # Re-transform with fresh BagOfWords after rebuild
            x = self.bow.transform_one(query)

        # ── 5. Learn ──────────────────────────────────────────────────────────
        self.model.learn_one(x, y)
        self.bow.learn_one(query)
        self.n_samples += 1

        # Update helpfulness tracker from QueryFeedback.helpful flag
        if feedback.helpful:
            self.topic_helpfulness[y]["helpful"] += 1
        else:
            self.topic_helpfulness[y]["not_helpful"] += 1
            
        # ── 6. Log ────────────────────────────────────────────────────────────
        if self.n_samples % 10 == 0:
            self._record_state()

        self._log_jsonl()

        return {
            "predicted": pred,
            "actual": y,
            "correct": bool(correct),
            "accuracy": self.prequential_acc.get(),
            "rolling_accuracy": self.rolling_acc.get(),
            "drift_detected": self.drift_detected,
            "n_resets": self.n_resets,
            "step": self.n_samples,
        }

    def learn_from_feedback(self, event: FeedbackEvent) -> dict:
        """
        Learn from a user feedback event.
        Called by the /feedback API endpoint in D2.

        Also updates the helpfulness tracker for the predicted topic.
        This tells us which topics are failing in retrieval — not just
        which topics the classifier gets wrong.

        helpful=True  → user confirmed the answer was good
        helpful=False → user said the answer was bad
        """
        # Update helpfulness tracker for the predicted topic
        # We track the PREDICTED topic not the true topic because
        # the user experienced the answer for what we predicted —
        # they don't know what the true topic should have been
        if event.predicted_topic in self.topic_helpfulness:
            if event.helpful:
                self.topic_helpfulness[event.predicted_topic]["helpful"] += 1
            else:
                self.topic_helpfulness[event.predicted_topic]["not_helpful"] += 1

        fb = QueryFeedback(
            query=event.query,
            topic=event.true_topic,
            helpful=event.helpful,
        )
        return self.learn_one(fb)

    def _handle_drift(self) -> None:
        """
        Called when ADWIN detects concept drift.
        Rebuilds the entire pipeline from scratch.

        Unlike the selective reset (keep TFIDF, reset classifier),
        we do a full rebuild here because BagOfWords vocabulary
        may also be stale when the topic distribution shifts completely.
        """
        self.n_resets += 1
        self.drift_detected = False
        self._build_pipeline()
        # Reset ADWIN window to monitor the new distribution fresh
        # self.adwin = drift.ADWIN(delta=self.adwin_delta)
        logger.info("Pipeline rebuilt. Total resets: %d", self.n_resets)

    def topic_accuracy_report(self) -> dict:
        """
        Returns per-topic recall and precision sorted by recall ascending.

        Recall    : when true label IS this topic, how often correct?
                    low recall = model misses this topic = needs more data
        Precision : when we PREDICT this topic, how often correct?
                    low precision = model confuses other topics with this one

        Used by the D1 notebook to show which topics need more coverage.
        Also flows into D2 — topics with low recall need more PDFs in the corpus.
        """
        report = {}
        for topic in TOPICS:
            recall    = self.per_topic_recall[topic].get()
            precision = self.per_topic_precision[topic].get()
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            report[topic] = {
                "recall":    round(recall, 3),
                "precision": round(precision, 3),
                "f1":        round(f1, 3),
            }

        # Sort by recall ascending — weakest topics first
        return dict(
            sorted(report.items(), key=lambda x: x[1]["recall"])
        )

    def helpfulness_report(self) -> dict:
        """
        Returns helpfulness rate per topic sorted ascending (worst first).

        Helpfulness rate = helpful / (helpful + not_helpful)

        A low rate means users are consistently marking answers for that
        topic as not helpful — indicating the retrieval pipeline or corpus
        coverage needs improvement for that topic in D2.

        Topics with zero feedback are marked as None — not enough data yet.

        Returns
        -------
        dict mapping topic -> {helpful, not_helpful, rate, total}
        sorted by rate ascending (worst performing topics first)
        """
        report = {}
        for topic, counts in self.topic_helpfulness.items():
            total = counts["helpful"] + counts["not_helpful"]
            rate = (
                round(counts["helpful"] / total, 3)
                if total > 0
                else None
            )
            report[topic] = {
                "helpful":     counts["helpful"],
                "not_helpful": counts["not_helpful"],
                "total":       total,
                "rate":        rate,
            }

        # Sort by rate ascending — topics with no feedback go last
        return dict(
            sorted(
                report.items(),
                key=lambda x: (x[1]["rate"] is None, x[1]["rate"] or 0)
            )
        )
    def _record_state(self) -> None:
        """Appends a LearnerState snapshot to self.history every 10 steps."""
        self.history.append(LearnerState(
            step=self.n_samples,
            accuracy=round(self.prequential_acc.get(), 4),
            rolling_accuracy=round(self.rolling_acc.get(), 4),
            drift_detected=self.drift_detected,
            resets=self.n_resets,
        ))

    def _log_jsonl(self) -> None:
        """
        Append current state to the JSONL log file after every step.
        Real-time logging means no data is lost if the process crashes.
        The notebook loads this file to draw the prequential chart.
        """
        record = {
            "step": self.n_samples,
            "accuracy": round(self.prequential_acc.get(), 4),
            "rolling_accuracy": round(self.rolling_acc.get(), 4),
            "drift_detected": self.drift_detected,
            "n_resets": self.n_resets,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save(self, path: str | Path = "data/learner_state.json") -> None:
        """
        Save metrics summary and full history to JSON.
        The D1 notebook loads this for the prequential chart.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_samples": self.n_samples,
            "n_resets": self.n_resets,
            "final_accuracy": round(self.prequential_acc.get(), 4),
            "final_rolling_accuracy": round(self.rolling_acc.get(), 4),
            "drift_points": self.drift_points,
            "adwin_delta": self.adwin_delta,
            "alpha": self.alpha,
            "history": [
                {
                    "step": s.step,
                    "accuracy": s.accuracy,
                    "rolling_accuracy": s.rolling_accuracy,
                    "drift_detected": s.drift_detected,
                    "resets": s.resets,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in self.history
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Learner state saved to %s", path)

    def summary(self) -> dict:
        """Returns a quick summary of the current learner state."""
        return {
            "n_samples": self.n_samples,
            "n_resets": self.n_resets,
            "accuracy": round(self.prequential_acc.get(), 4),
            "rolling_accuracy": round(self.rolling_acc.get(), 4),
            "drift_points": self.drift_points,
            "topics": TOPICS,
        }


# # ── HybridWeightAdapter ───────────────────────────────────────────────────────

# class HybridWeightAdapter:
#     """
#     Adapts the BM25 vs dense fusion weight (alpha) from user feedback.
#     Used by the /feedback endpoint in D2 to tune retrieval in real time.

#     alpha = 0.0 → pure BM25  (keyword search)
#     alpha = 1.0 → pure dense (vector search)
#     alpha = 0.5 → equal blend (default)

#     Update logic
#     ------------
#     helpful=True  + dense   → increase alpha (dense worked well)
#     helpful=True  + bm25    → decrease alpha (bm25 worked well)
#     helpful=False + dense   → decrease alpha (dense failed)
#     helpful=False + bm25    → increase alpha (bm25 failed)
#     helpful=True  + hybrid  → no change (both contributed)
#     helpful=False + hybrid  → small decrease (blend needs tuning)

#     ADWIN also monitors the helpful/not-helpful signal stream.
#     If the helpfulness pattern shifts, it logs the drift event.

#     Usage
#     -----
#     adapter = HybridWeightAdapter()
#     new_alpha = adapter.update(helpful=True, retrieval_type="dense")
#     print(new_alpha)  # slightly higher than 0.5
#     """

#     def __init__(
#         self,
#         alpha: float = 0.5,
#         lr: float = 0.01,
#         adwin_delta: float = 0.002,
#     ) -> None:
#         """
#         Parameters
#         ----------
#         alpha       : initial fusion weight (0=BM25, 1=dense, 0.5=equal blend)
#         lr          : learning rate — how much to nudge alpha per feedback
#                       0.01 means each feedback moves alpha by 1%
#         adwin_delta : ADWIN sensitivity for helpfulness drift detection
#         """
#         self.alpha = alpha
#         self.lr = lr
#         self.adwin = drift.ADWIN(delta=adwin_delta)
#         self.step = 0
#         self.history: list[dict] = []

#     def update(
#         self,
#         helpful: bool,
#         retrieval_type: Literal["bm25", "dense", "hybrid"],
#     ) -> float:
#         """
#         Update alpha based on one feedback signal.

#         Parameters
#         ----------
#         helpful        : did the user mark the answer as helpful?
#         retrieval_type : which retrieval method was used for this query

#         Returns
#         -------
#         float : the new alpha value after this update
#         """
#         # Nudge alpha based on what worked and what didn't
#         if helpful:
#             if retrieval_type == "dense":
#                 # Dense worked well → lean more towards dense
#                 self.alpha = min(1.0, self.alpha + self.lr)
#             elif retrieval_type == "bm25":
#                 # BM25 worked well → lean more towards BM25
#                 self.alpha = max(0.0, self.alpha - self.lr)
#             # hybrid + helpful → no change, both contributed
#         else:
#             if retrieval_type == "dense":
#                 # Dense failed → lean away from dense
#                 self.alpha = max(0.0, self.alpha - self.lr)
#             elif retrieval_type == "bm25":
#                 # BM25 failed → lean away from BM25
#                 self.alpha = min(1.0, self.alpha + self.lr)
#             elif retrieval_type == "hybrid":
#                 # Hybrid failed → small nudge towards BM25 (more precise)
#                 self.alpha = max(0.0, self.alpha - self.lr * 0.5)

#         # Monitor helpfulness signal for drift
#         # 0 = helpful (good), 1 = not helpful (bad)
#         self.adwin.update(0 if helpful else 1)

#         if self.adwin.drift_detected:
#             logger.warning(
#                 "HybridWeightAdapter: helpfulness drift at step %d, alpha=%.3f",
#                 self.step, self.alpha,
#             )

#         self.step += 1
#         self.history.append({
#             "step": self.step,
#             "alpha": round(self.alpha, 4),
#             "helpful": helpful,
#             "retrieval_type": retrieval_type,
#         })

#         return round(self.alpha, 4)

#     def get_weights(self) -> dict:
#         """
#         Returns the current fusion weights as a dict.
#         Used by the retrieval pipeline in D2 to blend BM25 and dense scores.

#         Returns
#         -------
#         dict with bm25_weight and dense_weight that sum to 1.0
#         """
#         return {
#             "dense_weight": round(self.alpha, 4),
#             "bm25_weight": round(1.0 - self.alpha, 4),
#         }

#     def save(self, path: str | Path = "data/hybrid_adapter_state.json") -> None:
#         """Save adapter state and history to JSON."""
#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True)

#         data = {
#             "alpha": self.alpha,
#             "step": self.step,
#             "lr": self.lr,
#             "history": self.history,
#         }

#         with open(path, "w") as f:
#             json.dump(data, f, indent=2)

#         logger.info("HybridWeightAdapter state saved to %s", path)

#     def summary(self) -> dict:
#         """Returns a quick summary of the current adapter state."""
#         return {
#             "alpha": round(self.alpha, 4),
#             "step": self.step,
#             "weights": self.get_weights(),
#         }
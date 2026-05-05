"""
adaptation/online_learner.py
-----------------------------
Online query-to-topic classifier for the PDF-Papers AI Agent.

PURPOSE
-------
When a user submits a query to the agent (e.g. "how does attention work?"),
we want to classify it into one of 9 research topics BEFORE searching the
corpus. Knowing the topic helps the retrieval system search smarter in D2/D3.

WHY ONLINE LEARNING?
--------------------
A normal sklearn classifier trains once on a fixed dataset.
An online learner updates itself after EVERY query it sees.
This means it adapts as users' interests shift over time —
no retraining, no storing all data in memory.
River is the Python library built for this exact use case.

THE PIPELINE (3 steps)
----------------------
raw query text
    └─► TFIDF vectorizer   : converts text to weighted word scores
    └─► StandardScaler     : normalizes those scores
    └─► SoftmaxRegression  : multi-class classifier (9 topics)

WHY TFIDF NOT BAGOFWORDS?
-------------------------
Queries are short (one sentence). Words like "how", "what", "explain"
appear in every query regardless of topic — they carry no signal.
TFIDF penalizes those common words and rewards topic-specific words
like "attention", "backpropagation", "Q-learning".
River's TFIDF updates its statistics incrementally so it works
perfectly in an online setting.

PREQUENTIAL EVALUATION
----------------------
Rule: always PREDICT first, then LEARN.
This ensures our accuracy metric is always measured on unseen data —
exactly like real deployment conditions.

ADWIN DRIFT DETECTION
---------------------
ADWIN watches the stream of prediction errors (0=correct, 1=wrong).
When the recent error rate is statistically higher than before,
it signals that the query distribution has shifted (concept drift).
We then reset only the classifier weights — the TFIDF vocabulary
stays because the words users type don't change, only the topics do.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from river import compose, feature_extraction, linear_model, metrics, drift, preprocessing

logger = logging.getLogger(__name__)


# ── Topic labels ──────────────────────────────────────────────────────────────
# These 9 topics map to the Topic nodes in the Neo4j graph (built in D2).
# Every query will be classified into one of these.
TOPICS: list[str] = [
    "neural_networks",
    "transformers",
    "reinforcement_learning",
    "computer_vision",
    "natural_language_processing",
    "graph_neural_networks",
    "generative_models",
    "optimization",
    "other",
]


# ── Input / output data classes ───────────────────────────────────────────────

@dataclass
class QueryFeedback:
    """
    One labelled interaction from a user.

    query    : the raw text the user typed into the agent
    topic    : the correct topic label
               (comes from user feedback or manual annotation)
    helpful  : did the user mark the answer as helpful? (y/n)
               stored for future use in adaptive hybrid weight tuning
    timestamp: when this interaction happened
    """
    query: str
    topic: str
    helpful: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LearnerState:
    """
    A snapshot of the learner's metrics at one point in time.
    Saved every 10 samples and used to draw the prequential chart.

    step          : how many samples have been processed so far
    accuracy      : prequential accuracy at this step
    drift_detected: was drift detected at this exact step?
    resets        : total number of classifier resets so far
    """
    step: int
    accuracy: float
    drift_detected: bool
    resets: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ── Main learner ──────────────────────────────────────────────────────────────

class QueryTopicLearner:
    """
    Incremental query-to-topic classifier with ADWIN drift detection.

    Usage example
    -------------
    learner = QueryTopicLearner()

    # At inference time (no learning):
    result = learner.predict("how does attention work?")
    print(result["topic"])  # e.g. "transformers"

    # When feedback arrives (learn from it):
    fb = QueryFeedback(query="how does attention work?",
                       topic="transformers", helpful=True)
    result = learner.learn_one(fb)
    print(result["accuracy"])  # prequential accuracy so far
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._build_pipeline()

        # ADWIN drift detector
        # delta controls sensitivity — lower = fires more easily
        # 0.002 is a standard starting point for concept drift detection
        self.drift_detector = drift.ADWIN(delta=0.002)

        # Prequential accuracy metric (updated BEFORE learning on each sample)
        self.prequential_acc = metrics.Accuracy()

        # Counters
        self.n_samples: int = 0       # total samples processed
        self.n_resets: int = 0        # total drift-triggered resets
        self.drift_detected: bool = False

        # History snapshots — saved every 10 steps for the plot
        self.history: list[LearnerState] = []

    def _build_pipeline(self) -> None:
        """
        Builds the River 3-step pipeline.

        Called once at init.
        On drift reset, only the classifier step is replaced —
        this method is NOT called again on reset (that would wipe the TFIDF stats).
        """
        self.model = compose.Pipeline(
            # Extract the 'text' value from the input dict
            ("select", compose.Select("text")),
            # Step 1 — TFIDF vectorizer
            # Converts query text into a dict of {word: tfidf_score}
            # ngram_range=(1,2) means it captures single words AND word pairs
            # e.g. "attention work" as a unit, not just "attention" and "work" separately
            ("vectorizer", feature_extraction.TFIDF(
                lowercase=True,      # "Attention" and "attention" treated the same
                strip_accents=True,  # handles accented characters in paper titles
                ngram_range=(1, 2),  # unigrams + bigrams for better topic signal
            )),

            # Step 2 — StandardScaler
            # Normalizes each feature using running mean and variance.
            # Needed because TFIDF scores vary widely across vocabulary size.
            # River's version updates statistics incrementally — no full pass needed.
            ("scaler", preprocessing.StandardScaler()),

            # Step 3 — SoftmaxRegression (multi-class logistic regression)
            # Outputs a probability for each of the 9 topics.
            # Picks the highest probability as the predicted topic.
            # Updates weights via SGD on each sample.
            # l2=1e-4 adds slight regularization to prevent overfitting on small data.
            ("classifier", linear_model.SoftmaxRegression(
                optimizer=None,  # uses River's default SGD
                l2=1e-4,
            )),
        )

    def learn_one(self, feedback: QueryFeedback) -> dict:
        """
        Core prequential update — the heart of the online learner.

        The order of these 4 steps is fixed and must not change:

        1. PREDICT  — what topic do we think this query belongs to?
                      done BEFORE learning so the accuracy is unbiased
        2. EVALUATE — were we right? update the accuracy metric
        3. LEARN    — now update the model weights with the correct label
        4. DRIFT    — feed the error (0 or 1) to ADWIN
                      if ADWIN fires, reset the classifier weights

        Parameters
        ----------
        feedback : QueryFeedback with the query text and correct topic label

        Returns
        -------
        dict with prediction, accuracy, drift status, and step count
        """
        # River pipelines expect a dict as input
        x = {"text": feedback.query}
        y = feedback.topic

        # ── 1. PREDICT (before learning) ─────────────────────────────────────
        # On the very first sample the model has no weights yet,
        # so we catch the exception and fall back to "other"
        try:
            y_pred = self.model.predict_one(x)
        except Exception:
            y_pred = "other"

        # ── 2. EVALUATE ───────────────────────────────────────────────────────
        # correct = 1 if prediction matches label, else 0
        correct = int(y_pred == y)
        self.prequential_acc.update(y, y_pred)

        # ── 3. LEARN ──────────────────────────────────────────────────────────
        # Now we update the pipeline weights with the true label
        self.model.learn_one(x, y)
        self.n_samples += 1

        # ── 4. DRIFT DETECTION ────────────────────────────────────────────────
        # Feed the error signal into ADWIN
        # 1 = wrong prediction (error), 0 = correct prediction (no error)
        self.drift_detector.update(1 - correct)
        self.drift_detected = self.drift_detector.drift_detected

        if self.drift_detected:
            logger.warning(
                "ADWIN drift detected at step %d — resetting classifier.",
                self.n_samples,
            )
            self._handle_drift()

        # Save a snapshot every 10 samples for the prequential plot
        if self.n_samples % 10 == 0:
            self._record_state()

        return {
            "predicted": y_pred,
            "actual": y,
            "correct": bool(correct),
            "accuracy": self.prequential_acc.get(),
            "drift_detected": self.drift_detected,
            "n_resets": self.n_resets,
            "step": self.n_samples,
        }

    def _handle_drift(self) -> None:
        """
        Called when ADWIN detects concept drift.

        What we RESET:
            - SoftmaxRegression weights (the learned topic boundaries are stale)
            - ADWIN window (start monitoring the new distribution fresh)

        What we KEEP:
            - TFIDF vocabulary and IDF statistics
              (the words users type don't change, only topic distributions do)
            - StandardScaler running mean/variance
              (still useful context even after a topic shift)

        This selective reset means the model can recover faster after drift
        because it doesn't have to relearn the vocabulary from scratch.
        """
        self.n_resets += 1
        self.drift_detected = False

        # Replace only the classifier step — pipeline keys are accessible by name
        self.model["classifier"] = linear_model.SoftmaxRegression(
            optimizer=None,
            l2=1e-4,
        )

        # Reset ADWIN so it starts monitoring the new distribution
        self.drift_detector = drift.ADWIN(delta=0.002)

        logger.info("Classifier reset complete. Total resets: %d", self.n_resets)

    def predict(self, query: str) -> dict:
        """
        Predict the topic of a query WITHOUT updating the model.

        This is called at inference time when a user submits a question
        to the agent. It does NOT affect the model weights.

        Returns a dict with the predicted topic and probability scores
        for all 9 topics, sorted by confidence (highest first).
        """
        x = {"text": query}
        try:
            topic = self.model.predict_one(x)
            probas = self.model.predict_proba_one(x)
        except Exception:
            # Cold start — model not trained yet
            topic = "other"
            probas = {t: round(1.0 / len(TOPICS), 4) for t in TOPICS}

        return {
            "query": query,
            "topic": topic,
            # Sort probabilities highest first so the most confident topic is at the top
            "probabilities": {
                k: round(v, 4)
                for k, v in sorted(
                    probas.items(), key=lambda item: item[1], reverse=True
                )
            },
        }

    def _record_state(self) -> None:
        """Appends a LearnerState snapshot to self.history every 10 steps."""
        self.history.append(LearnerState(
            step=self.n_samples,
            accuracy=round(self.prequential_acc.get(), 4),
            drift_detected=self.drift_detected,
            resets=self.n_resets,
        ))

    def save(self, path: str | Path) -> None:
        """
        Saves the accuracy history to a JSON file.

        This JSON is what the D1 notebook loads to draw the prequential chart.

        Note: River model weights live in memory only.
        Persisting the full model state requires pickling the River pipeline,
        which we keep separate from this history export.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_samples": self.n_samples,
            "n_resets": self.n_resets,
            "final_accuracy": round(self.prequential_acc.get(), 4),
            "history": [
                {
                    "step": s.step,
                    "accuracy": s.accuracy,
                    "drift_detected": s.drift_detected,
                    "resets": s.resets,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in self.history
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("History saved to %s", path)

    def summary(self) -> dict:
        """Returns a quick summary of the current learner state."""
        return {
            "n_samples": self.n_samples,
            "n_resets": self.n_resets,
            "accuracy": round(self.prequential_acc.get(), 4),
            "topics": TOPICS,
        }
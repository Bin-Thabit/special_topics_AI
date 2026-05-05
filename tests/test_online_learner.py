"""
tests/test_online_learner.py
-----------------------------
Smoke tests for adaptation/online_learner.py

WHAT THESE TESTS DO
-------------------
They verify that the online learner:
  - returns the correct output structure
  - updates its internal counters correctly
  - handles the cold start (no training yet) without crashing
  - detects drift and increments the reset counter
  - saves history to a JSON file correctly

These are smoke tests — they check that nothing is broken,
not that the model is highly accurate. Accuracy depends on
data volume which is not the concern of unit tests.

HOW TO RUN
----------
From the project root:
    pytest tests/test_online_learner.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Allow running from project root on Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptation.online_learner import QueryTopicLearner, QueryFeedback, TOPICS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def learner():
    """
    Creates a fresh QueryTopicLearner before each test.
    seed=42 makes results reproducible across machines.
    Each test gets its own independent instance — no shared state.
    """
    return QueryTopicLearner(seed=42)


@pytest.fixture
def trained_learner():
    """
    A learner that has already seen 50 samples.
    Used for tests that need a warm model (not cold start).
    We feed it clean, correctly labelled samples so the
    model has meaningful weights before the test runs.
    """
    learner = QueryTopicLearner(seed=42)

    # Simple query templates per topic for warm-up
    samples = [
        ("how does attention work in transformers", "transformers"),
        ("explain backpropagation in neural networks", "neural_networks"),
        ("what is Q-learning in reinforcement learning", "reinforcement_learning"),
        ("how do convolutional networks process images", "computer_vision"),
        ("explain TF-IDF for text retrieval", "natural_language_processing"),
    ]

    # Cycle through samples 10 times = 50 total training steps
    for i in range(50):
        query, topic = samples[i % len(samples)]
        fb = QueryFeedback(query=query, topic=topic, helpful=True)
        learner.learn_one(fb)

    return learner


# ── Tests: predict() ──────────────────────────────────────────────────────────

def test_predict_cold_start_does_not_crash(learner):
    """
    predict() should not raise an exception on a fresh untrained model.
    The cold start fallback should return 'other' as the default topic.
    """
    result = learner.predict("how does attention work")
    assert result["topic"] == "other"


def test_predict_returns_valid_topic(trained_learner):
    """
    After training, predict() must return one of the 9 known topics.
    Returning anything outside TOPICS would break the Neo4j graph in D2.
    """
    result = trained_learner.predict("explain backpropagation in deep learning")
    assert result["topic"] in TOPICS


def test_predict_returns_all_probabilities(trained_learner):
    """
    predict() must return a probability score for every topic.
    The agent uses these scores to decide how confident the classification is.
    """
    result = trained_learner.predict("what is Q-learning")

    # All 9 topics must be present
    assert set(result["probabilities"].keys()) == set(TOPICS)

    # Every probability must be between 0 and 1
    for topic, prob in result["probabilities"].items():
        assert 0.0 <= prob <= 1.0, f"Probability out of range for topic: {topic}"


def test_predict_probabilities_sorted_descending(trained_learner):
    """
    Probabilities must be sorted highest first.
    This makes it easy for the agent to read the top prediction.
    """
    result = trained_learner.predict("explain self attention mechanism")
    probs = list(result["probabilities"].values())

    # Each value should be >= the next one
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1], "Probabilities are not sorted descending"


# ── Tests: learn_one() ────────────────────────────────────────────────────────

def test_learn_one_returns_required_keys(learner):
    """
    learn_one() must return all expected keys.
    The API /feedback endpoint and the agent both rely on this structure.
    """
    fb = QueryFeedback(
        query="explain backpropagation",
        topic="neural_networks",
        helpful=True,
    )
    result = learner.learn_one(fb)

    required_keys = {"predicted", "actual", "correct", "accuracy",
                     "drift_detected", "n_resets", "step"}

    for key in required_keys:
        assert key in result, f"Missing key in learn_one output: {key}"


def test_learn_one_increments_n_samples(learner):
    """
    n_samples must go up by 1 after each learn_one call.
    This counter is used for the prequential chart x-axis.
    """
    assert learner.n_samples == 0

    fb = QueryFeedback(query="test query", topic="other", helpful=True)
    learner.learn_one(fb)
    assert learner.n_samples == 1

    learner.learn_one(fb)
    assert learner.n_samples == 2


def test_learn_one_accuracy_stays_between_0_and_1(learner):
    """
    Prequential accuracy must always be a valid probability.
    Anything outside [0, 1] would mean a bug in the metric update.
    """
    samples = [
        ("how does attention work", "transformers"),
        ("explain backpropagation", "neural_networks"),
        ("what is Q-learning", "reinforcement_learning"),
        ("image segmentation methods", "computer_vision"),
        ("named entity recognition", "natural_language_processing"),
    ]

    for query, topic in samples * 4:  # 20 samples total
        fb = QueryFeedback(query=query, topic=topic, helpful=True)
        result = learner.learn_one(fb)
        assert 0.0 <= result["accuracy"] <= 1.0


def test_learn_one_actual_matches_input_label(learner):
    """
    The 'actual' field in the output must always match the label we passed in.
    This confirms the learner is not silently modifying the label.
    """
    fb = QueryFeedback(
        query="explain diffusion models",
        topic="generative_models",
        helpful=True,
    )
    result = learner.learn_one(fb)
    assert result["actual"] == "generative_models"


# ── Tests: history and snapshots ─────────────────────────────────────────────

def test_history_recorded_every_10_steps(learner):
    """
    A snapshot must be added to history every 10 samples.
    After 30 samples we expect exactly 3 snapshots.
    This history is what plot_metrics.py uses to draw the chart.
    """
    fb = QueryFeedback(query="test", topic="other", helpful=True)

    for _ in range(30):
        learner.learn_one(fb)

    assert len(learner.history) == 3


def test_history_snapshot_has_correct_keys(learner):
    """
    Each history snapshot must have the keys that plot_metrics.py expects.
    If a key is missing the chart will crash at plotting time.
    """
    fb = QueryFeedback(query="test", topic="other", helpful=True)

    for _ in range(10):
        learner.learn_one(fb)

    snapshot = learner.history[0]
    assert hasattr(snapshot, "step")
    assert hasattr(snapshot, "accuracy")
    assert hasattr(snapshot, "drift_detected")
    assert hasattr(snapshot, "resets")


# ── Tests: drift and resets ───────────────────────────────────────────────────

def test_n_resets_starts_at_zero(learner):
    """Sanity check — no resets should have happened on a fresh learner."""
    assert learner.n_resets == 0


def test_drift_reset_increments_counter():
    """
    When we inject maximum noise (random labels on fixed query),
    ADWIN should eventually fire and n_resets should go up.

    We use a fixed query with random labels to maximize the error rate
    and force ADWIN to detect drift as quickly as possible.
    """
    import random
    random.seed(99)

    learner = QueryTopicLearner(seed=99)
    initial_resets = learner.n_resets

    for _ in range(600):
        # Same query every time but random wrong label
        # This creates maximum prediction error -> ADWIN fires quickly
        fb = QueryFeedback(
            query="attention mechanism transformer",
            topic=random.choice(TOPICS),
            helpful=False,
        )
        learner.learn_one(fb)

    # After 600 samples of pure noise, at least one reset must have happened
    assert learner.n_resets > initial_resets, (
        "ADWIN should have detected drift under 600 samples of pure noise"
    )


# ── Tests: save() ─────────────────────────────────────────────────────────────

def test_save_creates_json_file(tmp_path, learner):
    """
    save() must write a JSON file to the given path.
    tmp_path is a pytest built-in fixture that gives us a temporary directory.
    The file is cleaned up automatically after the test.
    """
    fb = QueryFeedback(query="test query", topic="other", helpful=True)

    # Need at least 10 samples to have one history snapshot
    for _ in range(10):
        learner.learn_one(fb)

    output_file = tmp_path / "history.json"
    learner.save(output_file)

    assert output_file.exists(), "save() did not create the JSON file"


def test_save_json_has_correct_structure(tmp_path, learner):
    """
    The saved JSON must have the keys that the D1 notebook expects
    when loading history for the prequential chart.
    """
    fb = QueryFeedback(query="test query", topic="other", helpful=True)
    for _ in range(10):
        learner.learn_one(fb)

    output_file = tmp_path / "history.json"
    learner.save(output_file)

    with open(output_file) as f:
        data = json.load(f)

    # Top-level keys
    assert "n_samples" in data
    assert "n_resets" in data
    assert "final_accuracy" in data
    assert "history" in data

    # Each history entry must have the keys plot_metrics.py needs
    for entry in data["history"]:
        assert "step" in entry
        assert "accuracy" in entry
        assert "drift_detected" in entry
        assert "resets" in entry


# ── Tests: summary() ─────────────────────────────────────────────────────────

def test_summary_returns_correct_keys(learner):
    """summary() is used by the /stats API endpoint in D2."""
    s = learner.summary()
    for key in ["n_samples", "n_resets", "accuracy", "topics"]:
        assert key in s


def test_summary_topics_matches_global_topics(learner):
    """
    The topics list in summary() must match the global TOPICS constant.
    Neo4j seed script and the agent both rely on this being consistent.
    """
    assert learner.summary()["topics"] == TOPICS
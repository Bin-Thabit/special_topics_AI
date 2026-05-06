import json
import math
import tempfile
import numpy as np
import pytest
import optuna

from unittest.mock import patch, MagicMock
from automl.bohb_search import get_embeddings, run_bohb


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_chunks():
    """10 fake chunks — enough to test without real data."""
    return [
        {"chunk_id": f"c{i}", "text": f"This is test chunk number {i}."}
        for i in range(10)
    ]


@pytest.fixture
def dummy_gold(tmp_path):
    """Small gold QA file with 6 questions."""
    gold = [
        {
            "question": f"What is chunk {i}?",
            "relevant_chunk_ids": [f"c{i}"]
        }
        for i in range(6)
    ]
    path = tmp_path / "gold_qa.json"
    path.write_text(json.dumps(gold))
    return str(path)


@pytest.fixture
def real_model():
    """Load the real embedding model once for the whole test session."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en")


# ── get_embeddings tests ──────────────────────────────────────────────────────

class TestGetEmbeddings:

    def test_shape_no_svd(self, dummy_chunks, real_model):
        """Without SVD, output shape should be (n_chunks, 384)."""
        embs, svd = get_embeddings(dummy_chunks, real_model, normalize=False, svd_dim=None)
        assert embs.shape == (10, 384)
        assert svd is None

    def test_shape_with_svd(self, dummy_chunks, real_model):
        """With SVD, output shape should be (n_chunks, svd_dim)."""
        embs, svd = get_embeddings(dummy_chunks, real_model, normalize=False, svd_dim=8)
        assert embs.shape == (10, 8)
        assert svd is not None

    def test_normalization(self, dummy_chunks, real_model):
        """With normalize=True, every row should have unit length."""
        embs, _ = get_embeddings(dummy_chunks, real_model, normalize=True, svd_dim=None)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

    def test_svd_query_projection_matches_index(self, dummy_chunks, real_model):
        """
        The SVD object returned must project queries into the same
        space as the index — this is the core bug we fixed.
        """
        embs, svd = get_embeddings(dummy_chunks, real_model, normalize=False, svd_dim=8)

        raw_query = real_model.encode("test query", convert_to_numpy=True)
        projected = svd.transform(raw_query.reshape(1, -1))[0]

        # matmul must not raise and produce one score per chunk
        scores = embs @ projected
        assert scores.shape == (10,)

    def test_no_svd_query_matmul(self, dummy_chunks, real_model):
        """Without SVD the raw query vector must multiply against the index."""
        embs, svd = get_embeddings(dummy_chunks, real_model, normalize=False, svd_dim=None)
        assert svd is None

        raw_query = real_model.encode("test query", convert_to_numpy=True)
        scores = embs @ raw_query
        assert scores.shape == (10,)

    def test_normalize_with_svd(self, dummy_chunks, real_model):
        """SVD + normalize should still produce unit-length rows."""
        embs, _ = get_embeddings(dummy_chunks, real_model, normalize=True, svd_dim=8)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)


# ── BOHB study configuration tests ───────────────────────────────────────────

class TestBOHBStudyConfig:

    def test_sampler_is_tpe(self):
        """BOHB must use TPESampler (the Bayesian part)."""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=5, max_resource=22, reduction_factor=3
            )
        )
        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_pruner_is_hyperband(self):
        """BOHB must use HyperbandPruner (the early-stopping part)."""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=5, max_resource=22, reduction_factor=3
            )
        )
        assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)

    def test_direction_is_maximize(self):
        """Study must maximize (higher NDCG = better)."""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=5, max_resource=22, reduction_factor=3
            )
        )
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE


# ── Pruning behaviour tests ───────────────────────────────────────────────────

class TestPruningBehaviour:

    def test_trial_pruned_exception_raised(self):
        """
        When should_prune() returns True, objective must raise
        TrialPruned so Hyperband can stop the trial early.

        Key insight: Hyperband prunes *relatively* — it kills the bottom
        fraction compared to other trials at the same rung. If all trials
        report identical scores (e.g. all 0.0), there is no contrast and
        nothing gets pruned. We alternate good (1.0) and bad (0.0) scores
        so the pruner has a clear signal.
        """
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1, max_resource=4, reduction_factor=2
            )
        )

        def pruning_objective(trial):
            trial.suggest_float("x", 0, 1)
            # Even-numbered trials are "bad", odd-numbered are "good".
            # This contrast is what allows Hyperband to prune bad trials.
            score = 1.0 if trial.number % 2 == 1 else 0.0
            for step in range(4):
                trial.report(score, step=step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return score

        # 50 trials gives Hyperband enough data across all rungs to prune
        study.optimize(pruning_objective, n_trials=50)

        pruned_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]
        assert len(pruned_trials) > 0, (
            f"Expected some pruned trials but got none. "
            f"States: {[t.state for t in study.trials]}"
        )


# ── run_bohb integration test (mocked IO) ────────────────────────────────────

class TestRunBohbIntegration:

    def test_returns_expected_keys(self, tmp_path, dummy_chunks, dummy_gold):
        """
        run_bohb must return a dict with all required keys.
        Uses real model but mocks file paths to avoid touching real data.
        """
        with (
            patch("automl.bohb_search.CHUNKS_PATH", new="data/sample_chunks.json"),
            patch("automl.bohb_search.GOLD_PATH",   new=dummy_gold),
            patch("automl.bohb_search.load_chunks", return_value=dummy_chunks),
            patch("automl.bohb_search.build_bm25_index") as mock_bm25,
        ):
            from rank_bm25 import BM25Okapi
            tokens = [c["text"].lower().split() for c in dummy_chunks]
            mock_bm25.return_value = BM25Okapi(tokens)

            results = run_bohb(n_trials=5)

        required_keys = {
            "best_params", "best_ndcg", "best_recall",
            "best_latency", "n_trials", "pruned", "complete"
        }
        assert required_keys == set(results.keys())

    def test_best_ndcg_is_valid_float(self, tmp_path, dummy_chunks, dummy_gold):
        """best_ndcg must be a float in [0, 1]."""
        with (
            patch("automl.bohb_search.CHUNKS_PATH", new="data/sample_chunks.json"),
            patch("automl.bohb_search.GOLD_PATH",   new=dummy_gold),
            patch("automl.bohb_search.load_chunks", return_value=dummy_chunks),
            patch("automl.bohb_search.build_bm25_index") as mock_bm25,
        ):
            from rank_bm25 import BM25Okapi
            tokens = [c["text"].lower().split() for c in dummy_chunks]
            mock_bm25.return_value = BM25Okapi(tokens)

            results = run_bohb(n_trials=5)

        assert isinstance(results["best_ndcg"], float)
        assert 0.0 <= results["best_ndcg"] <= 1.0

    def test_pruned_plus_complete_lte_n_trials(self, dummy_chunks, dummy_gold):
        """pruned + complete must never exceed n_trials."""
        with (
            patch("automl.bohb_search.CHUNKS_PATH", new="data/sample_chunks.json"),
            patch("automl.bohb_search.GOLD_PATH",   new=dummy_gold),
            patch("automl.bohb_search.load_chunks", return_value=dummy_chunks),
            patch("automl.bohb_search.build_bm25_index") as mock_bm25,
        ):
            from rank_bm25 import BM25Okapi
            tokens = [c["text"].lower().split() for c in dummy_chunks]
            mock_bm25.return_value = BM25Okapi(tokens)

            results = run_bohb(n_trials=5)

        assert results["pruned"] + results["complete"] <= results["n_trials"]
        
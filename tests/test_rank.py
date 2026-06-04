"""
tests/test_rank.py — Unit tests for graphrag.rank using pytest
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Import functions under test
from graphrag.rank import (
    adaptive_alpha,
    embed_query,
    pool_embeddings_from_state,
    pool_embeddings_from_qdrant,
    rank_pool,
    rerank_cross_encoder,
)


class TestGraphRagRank(unittest.TestCase):

    def setUp(self):
        # 1. Dummy mock chunks
        self.mock_pool_chunks = [
            {"chunk_id": "c1", "paper_id": "p1", "text": "Sentence 1 from paper 1.", "graph_score": 0.9},
            {"chunk_id": "c2", "paper_id": "p1", "text": "Sentence 2 from paper 1.", "graph_score": 0.3},
            {"chunk_id": "c3", "paper_id": "p2", "text": "Sentence 1 from paper 2.", "graph_score": 0.6},
        ]

        self.mock_all_chunks = [
            {"chunk_id": "c0", "paper_id": "p0", "text": "Unrelated prepended chunk"},
            {"chunk_id": "c1", "paper_id": "p1", "text": "Sentence 1 from paper 1."},
            {"chunk_id": "c2", "paper_id": "p1", "text": "Sentence 2 from paper 1."},
            {"chunk_id": "c3", "paper_id": "p2", "text": "Sentence 1 from paper 2."},
            {"chunk_id": "c4", "paper_id": "p3", "text": "Postpended chunk"},
        ]

        # 2. Mock 2D embeddings array aligned to mock_all_chunks (length 5)
        # Each row is a basic mock vector [index, index, index]
        self.mock_dense_embeddings = np.array([
            [0.0, 0.0, 0.0],  # c0
            [1.0, 1.0, 1.0],  # c1
            [2.0, 2.0, 2.0],  # c2
            [3.0, 3.0, 3.0],  # c3
            [4.0, 4.0, 4.0],  # c4
        ])

        # Mock Model Node
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

    # ---------------------------------------------------------------------------
    # Helper & Alignment Extraction Level Tests
    # ---------------------------------------------------------------------------

    def test_adaptive_alpha_resolving(self):
        # Case A: adapter is completely missing (fall back to default)
        alpha, weights = adaptive_alpha(None, default=0.7)
        self.assertEqual(alpha, 0.7)
        self.assertEqual(weights["dense_weight"], 0.7)

        # Case B: adapter is active and provides run_card parameters
        mock_adapter = MagicMock()
        mock_adapter.get_weights.return_value = {"dense_weight": 0.21, "bm25_weight": 0.79}
        alpha, weights = adaptive_alpha(mock_adapter, default=0.5)
        self.assertEqual(alpha, 0.21)
        self.assertEqual(weights["bm25_weight"], 0.79)

    def test_embed_query_formatting(self):
        # Verify query prefix matches the actual BGE schema instruction contract
        embed_query(self.mock_model, "Which framework is faster?")
        self.mock_model.encode.assert_called_once_with(
            "Represent this sentence for searching relevant passages: Which framework is faster?",
            normalize_embeddings=True
        )

    def test_pool_embeddings_from_state_mapping(self):
        # Extracts only c1, c2, c3 out of state.dense_embeddings using aligned indices
        kept, pooled_vectors = pool_embeddings_from_state(
            self.mock_pool_chunks,
            self.mock_all_chunks,
            self.mock_dense_embeddings
        )

        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0]["chunk_id"], "c1")
        self.assertEqual(kept[1]["chunk_id"], "c2")
        self.assertEqual(kept[2]["chunk_id"], "c3")

        # Verify correct row alignments were sliced
        np.testing.assert_array_equal(pooled_vectors[0], [1.0, 1.0, 1.0])  # c1 row
        np.testing.assert_array_equal(pooled_vectors[1], [2.0, 2.0, 2.0])  # c2 row
        np.testing.assert_array_equal(pooled_vectors[2], [3.0, 3.0, 3.0])  # c3 row

    @patch("graphrag.rank.uuid.uuid5")
    def test_pool_embeddings_from_qdrant_retrieve_mapping(self, mock_uuid):
        # Force repeatable UUID DNS mappings
        mock_uuid.side_effect = lambda ns, name: f"uuid-for-{name}"

        # Mock Qdrant records
        mock_rec1 = MagicMock(id="uuid-for-c1", vector=[1.1, 1.1])
        mock_rec3 = MagicMock(id="uuid-for-c3", vector=[3.3, 3.3])
        # Note: c2's vector is purposefully missing from Qdrant search indices to test graceful drop

        mock_qdrant = MagicMock()
        mock_qdrant.retrieve.return_value = [mock_rec1, mock_rec3]

        kept, pooled_vectors = pool_embeddings_from_qdrant(self.mock_pool_chunks, mock_qdrant)

        # Assertions: Should drop c2 and preserve aligned indexes for c1 and c3
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["chunk_id"], "c1")
        self.assertEqual(kept[1]["chunk_id"], "c3")
        np.testing.assert_array_equal(pooled_vectors[0], [1.1, 1.1])
        np.testing.assert_array_equal(pooled_vectors[1], [3.3, 3.3])

    # ---------------------------------------------------------------------------
    # Step 3 Execution Tests (Ranking & Blend Operations)
    # ---------------------------------------------------------------------------

    @patch("graphrag.rank.build_bm25_index")
    @patch("graphrag.rank.hybrid_search")
    def test_rank_pool_pure_hybrid(self, mock_hybrid_search, mock_build_bm25):
        mock_build_bm25.return_value = MagicMock()  # Mock BM25 index object
        
        # Mock Hybrid search return (which outputs chunks annotated with a search score)
        mock_hybrid_search.return_value = [
            {"chunk_id": "c1", "score": 0.8},
            {"chunk_id": "c3", "score": 0.6},
            {"chunk_id": "c2", "score": 0.4},
        ]

        result = rank_pool(
            query="Test Query",
            pool_chunks=self.mock_pool_chunks,
            model=self.mock_model,
            pool_embeddings=self.mock_dense_embeddings[:3],
            alpha=0.5,
            k=2,
            graph_weight=0.0  # pure hybrid path
        )

        # Assert top-k slice limit parameter of 2
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["chunk_id"], "c1")
        self.assertEqual(result[0]["score"], 0.8)

    @patch("graphrag.rank.build_bm25_index")
    @patch("graphrag.rank.hybrid_search")
    def test_rank_pool_blended_with_graph_weight(self, mock_hybrid_search, mock_build_bm25):
        mock_build_bm25.return_value = MagicMock()
        
        # We simulate hybrid search returns
        # c1 has graph_score 0.9 (highest), c2 has 0.3 (lowest), c3 has 0.6 (midpoint)
        mock_hybrid_search.return_value = [
            {"chunk_id": "c1", "score": 0.5, "graph_score": 0.9},
            {"chunk_id": "c2", "score": 0.6, "graph_score": 0.3},
            {"chunk_id": "c3", "score": 0.7, "graph_score": 0.6},
        ]

        # Blend equation calculation details check:
        # Min graph_score is 0.3, Max is 0.9. Span is 0.6.
        # c1 (g_norm = (0.9 - 0.3) / 0.6 = 1.0) -> Score = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        # c2 (g_norm = (0.3 - 0.3) / 0.6 = 0.0) -> Score = 0.5 * 0.6 + 0.5 * 0.0 = 0.30
        # c3 (g_norm = (0.6 - 0.3) / 0.6 = 0.5) -> Score = 0.5 * 0.7 + 0.5 * 0.5 = 0.60
        # Order should be c1 (0.75), c3 (0.60), c2 (0.30)
        
        result = rank_pool(
            query="Test Query",
            pool_chunks=self.mock_pool_chunks,
            model=self.mock_model,
            pool_embeddings=self.mock_dense_embeddings[:3],
            alpha=0.5,
            k=3,
            graph_weight=0.5  # 50/50 blend ratio
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["chunk_id"], "c1")
        self.assertAlmostEqual(result[0]["score"], 0.75)
        self.assertEqual(result[1]["chunk_id"], "c3")
        self.assertAlmostEqual(result[1]["score"], 0.60)
        self.assertEqual(result[2]["chunk_id"], "c2")
        self.assertAlmostEqual(result[2]["score"], 0.30)

    # ---------------------------------------------------------------------------
    # Optional Cross-Encoder Reranker Tests
    # ---------------------------------------------------------------------------

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_cross_encoder_ordering(self, mock_cross_encoder_class):
        # Mock actual predict invocation scoring returns
        mock_ce_instance = MagicMock()
        mock_ce_instance.predict.return_value = [1.2, -0.4, 3.5]  # Scores mapped respectively to index order

        input_results = [
            {"chunk_id": "c1", "text": "Text 1"},
            {"chunk_id": "c2", "text": "Text 2"},
            {"chunk_id": "c3", "text": "Text 3"},
        ]

        reranked = rerank_cross_encoder(
            query="Dummy",
            results=input_results,
            top_n=2,
            cross_encoder=mock_ce_instance
        )

        # Assertions: Should sort by CrossEncoder output descending and slice to 2
        # c3 (3.5) should be 1st, c1 (1.2) should be 2nd, c2 (-0.4) is dropped
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["chunk_id"], "c3")
        self.assertEqual(reranked[0]["rerank_score"], 3.5)
        self.assertEqual(reranked[1]["chunk_id"], "c1")
        self.assertEqual(reranked[1]["rerank_score"], 1.2)



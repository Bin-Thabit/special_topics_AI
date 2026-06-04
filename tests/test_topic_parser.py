"""
tests/test_topic_parser.py — Unit tests for graphrag.topic_parser using pytest
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Import target class under test
from graphrag.topic_parser import TopicParser


class TestTopicParser(unittest.TestCase):

    def setUp(self):
        # 1. Mock Neo4j store
        self.mock_store = MagicMock()
        # Simulate returning 2 topic names from the graph schema
        self.mock_store.run_query.return_value = [
            {"name": "reinforcement_learning"},
            {"name": "knowledge_graphs"}
        ]

        # 2. Mock SentenceTransformer Model
        self.mock_model = MagicMock()
        
        # Configure the encoder to return deterministic normalized vectors
        # for both the background corpus (topics) and search query runs
        def mock_encode(texts, normalize_embeddings=True):
            # Check if we are embedding topics (length-2 list) or query (length-1 list)
            if any("Represent this passage:" in t for t in texts):
                # 2 Topics: reinforcement_learning (1.0, 0.0), knowledge_graphs (0.0, 1.0)
                return np.array([[1.0, 0.0], [0.0, 1.0]])
            elif any("Represent this sentence for searching relevant passages:" in t for t in texts):
                # Query: Return an embedding that is slightly closer to reinforcement_learning
                return np.array([[0.8, 0.6]])
            return np.array([[0.0, 0.0]])

        self.mock_model.encode.side_effect = mock_encode

    # ---------------------------------------------------------------------------
    # Helper level tests
    # ---------------------------------------------------------------------------

    def test_humanize_formatting(self):
        # Humanizes snake_case labels correctly
        self.assertEqual(TopicParser._humanize("automated_reasoning"), "automated reasoning")
        self.assertEqual(TopicParser._humanize("machine_learning_"), "machine learning")
        self.assertEqual(TopicParser._humanize(" single_token "), "single token")

    # ---------------------------------------------------------------------------
    # Cache & Extraction Engine Level Tests
    # ---------------------------------------------------------------------------

    def test_initialization_and_caching(self):
        # Instantiate with mocks
        parser = TopicParser(self.mock_store, model=self.mock_model)

        # Assertions
        self.mock_store.run_query.assert_called_once_with("MATCH (t:Topic) RETURN t.name AS name ORDER BY name")
        self.assertEqual(parser._topics, ["reinforcement_learning", "knowledge_graphs"])
        self.assertIsNotNone(parser._topic_vecs)
        self.assertEqual(parser._topic_vecs.shape, (2, 2))

    def test_refresh_topics_empty_db(self):
        # Force DB to return an empty array of topics
        self.mock_store.run_query.return_value = []
        
        parser = TopicParser(self.mock_store, model=self.mock_model)
        
        # Verify safe degradation state
        self.assertEqual(parser._topics, [])
        self.assertIsNone(parser._topic_vecs)

    # ---------------------------------------------------------------------------
    # Topic Matching & Similarity Logic Tests
    # ---------------------------------------------------------------------------

    def test_parse_scored_metrics(self):
        parser = TopicParser(self.mock_store, model=self.mock_model)
        
        # Parse query close to reinforcement_learning
        results = parser.parse_scored("How does reinforcement learning handle planning?")

        # Assertions
        # Expecting cosine similarity order descending:
        # reinforcement_learning (dot product [1.0, 0.0] * [0.8, 0.6] = 0.8)
        # knowledge_graphs       (dot product [0.0, 1.0] * [0.8, 0.6] = 0.6)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["topic"], "reinforcement_learning")
        self.assertEqual(results[0]["score"], 0.8)
        self.assertEqual(results[1]["topic"], "knowledge_graphs")
        self.assertEqual(results[1]["score"], 0.6)

    def test_parse_filtering_by_min_score(self):
        parser = TopicParser(self.mock_store, model=self.mock_model)
        
        # Case A: Min score 0.70 (Only reinforcement_learning (0.8) qualifies)
        hits_high = parser.parse("Any query", top_k=2, min_score=0.7)
        self.assertEqual(hits_high, ["reinforcement_learning"])

        # Case B: Min score 0.50 (Both 0.80 and 0.60 qualify)
        hits_low = parser.parse("Any query", top_k=2, min_score=0.5)
        self.assertEqual(hits_low, ["reinforcement_learning", "knowledge_graphs"])

        # Case C: Min score 0.90 (Neither qualifies)
        hits_none = parser.parse("Any query", top_k=2, min_score=0.9)
        self.assertEqual(hits_none, [])

    def test_parse_scored_edge_cases(self):
        parser = TopicParser(self.mock_store, model=self.mock_model)

        # Empty question handles gracefully
        self.assertEqual(parser.parse_scored(""), [])
        self.assertEqual(parser.parse_scored(None), [])

        # Uncached topics scenario
        parser._topic_vecs = None
        self.assertEqual(parser.parse_scored("Valid question"), [])



"""
tests/test_expand.py — Unit tests for graphrag.expand using pytest
"""
import unittest
from unittest.mock import MagicMock, patch

# Import the function under test
from graphrag.expand import expand_to_chunks


class TestGraphRagExpand(unittest.TestCase):

    def setUp(self):
        # Mock database connection object
        self.mock_db = MagicMock()
        
        # Simulated subgraphs coming out of Step 1 (Neo4j)
        self.mock_dict_subgraph = [
            {"paper_id": "paper_A", "score": 0.9, "reasons": ["high citation", "core entity"]},
            {"paper_id": "paper_B", "score": 0.5, "reasons": ["peripheral connection"]},
        ]
        self.mock_string_subgraph = ["paper_A", "paper_B"]

        # Simulated raw chunk layout coming out of MongoDB for those papers
        # Chunks are expected to be ordered by default (paper -> page -> index)
        self.mock_db_chunks = [
            {"chunk_id": "c1", "paper_id": "paper_A", "page_num": 1, "text": "Paper A chunk 1 text"},
            {"chunk_id": "c2", "paper_id": "paper_A", "page_num": 2, "text": "Paper A chunk 2 text"},
            {"chunk_id": "c3", "paper_id": "paper_A", "page_num": 3, "text": "Paper A chunk 3 text"},
            {"chunk_id": "c4", "paper_id": "paper_B", "page_num": 1, "text": "Paper B chunk 1 text"},
            {"chunk_id": "c5", "paper_id": "paper_B", "page_num": 2, "text": "Paper B chunk 2 text"},
        ]

    # ---------------------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------------------

    @patch("graphrag.expand.get_chunks_by_paper_ids")
    def test_expand_with_dictionary_subgraph(self, mock_get_chunks):
        # Setup mock return value
        mock_get_chunks.return_value = list(self.mock_db_chunks)

        # Execute
        result = expand_to_chunks(self.mock_db, self.mock_dict_subgraph)

        # Assertions
        mock_get_chunks.assert_called_once_with(self.mock_db, ["paper_A", "paper_B"])
        self.assertEqual(len(result), 5)
        
        # Verify that graph signal scores & reasons were correctly attached to the chunks
        self.assertEqual(result[0]["graph_score"], 0.9)
        self.assertEqual(result[0]["graph_reasons"], ["high citation", "core entity"])
        self.assertEqual(result[4]["graph_score"], 0.5)
        self.assertEqual(result[4]["graph_reasons"], ["peripheral connection"])

    @patch("graphrag.expand.get_chunks_by_paper_ids")
    def test_expand_with_string_list_subgraph(self, mock_get_chunks):
        mock_get_chunks.return_value = list(self.mock_db_chunks)

        # Execute with standard string identifiers instead of objects
        result = expand_to_chunks(self.mock_db, self.mock_string_subgraph)

        # Assertions
        mock_get_chunks.assert_called_once_with(self.mock_db, ["paper_A", "paper_B"])
        self.assertEqual(len(result), 5)
        # Check that empty default fallbacks applied cleanly
        self.assertEqual(result[0]["graph_score"], 0)
        self.assertEqual(result[0]["graph_reasons"], [])

    @patch("graphrag.expand.get_chunks_by_paper_ids")
    def test_max_chunks_per_paper_cap(self, mock_get_chunks):
        mock_get_chunks.return_value = list(self.mock_db_chunks)

        # Cap the selection to max 2 chunks per paper
        result = expand_to_chunks(self.mock_db, self.mock_dict_subgraph, max_chunks_per_paper=2)

        # Assertions
        # paper_A has 3 chunks -> should drop 'c3' and keep 'c1', 'c2'
        # paper_B has 2 chunks -> should keep both 'c4', 'c5'
        kept_chunk_ids = [c["chunk_id"] for c in result]
        self.assertEqual(kept_chunk_ids, ["c1", "c2", "c4", "c5"])
        self.assertEqual(len(result), 4)

    @patch("graphrag.expand.get_chunks_by_paper_ids")
    def test_max_total_chunks_global_cap(self, mock_get_chunks):
        # We need to create a unique chunk list copy because sort mutate operations occur inline
        mock_get_chunks.return_value = [
            {"chunk_id": "c4", "paper_id": "paper_B"},
            {"chunk_id": "c5", "paper_id": "paper_B"},
            {"chunk_id": "c1", "paper_id": "paper_A"},
            {"chunk_id": "c2", "paper_id": "paper_A"},
            {"chunk_id": "c3", "paper_id": "paper_A"},
        ]

        # Request a hard global pool limit of 3 chunks.
        # Since paper_A has a higher graph score (0.9 vs 0.5), chunks from paper_A must be prioritized.
        result = expand_to_chunks(self.mock_db, self.mock_dict_subgraph, max_total_chunks=3)

        # Assertions
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(chunk["paper_id"], "paper_A", "Global cap failed to prioritize higher scored paper pools.")

    def test_empty_subgraph_returns_early(self):
        # Edge case check: empty inputs shouldn't invoke store connections
        result_empty_list = expand_to_chunks(self.mock_db, [])
        result_none = expand_to_chunks(self.mock_db, None)

        self.assertEqual(result_empty_list, [])
        self.assertEqual(result_none, [])



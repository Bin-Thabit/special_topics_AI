"""
tests/test_answer.py — Unit tests for graphrag.answer
"""
import unittest
from unittest.mock import MagicMock, patch
from openai import RateLimitError

# Import the functions under test
# Adjust the import path if your folder structure dictates otherwise
from graphrag.answer import _fmt_authors, build_context, _cited_numbers, generate_answer


class TestGraphRagAnswer(unittest.TestCase):

    def setUp(self):
        # Sample mock chunks to simulate output from Step 3 (Ranking)
        self.mock_chunks = [
            {
                "chunk_id": "chk_001",
                "paper_id": "arxiv_1234",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani", "Shazeer"],
                "year": 2017,
                "page_num": 4,
                "text": "Transformers process sequence tokens in parallel using self-attention mechanisms.",
            },
            {
                "chunk_id": "chk_002",
                "paper_id": "arxiv_5678",
                "title": "Deep Residual Learning",
                "authors": ["He"],
                "year": 2016,
                "page_num": 2,
                "text": "Residual mappings allow deep neural networks to train effectively without exploding gradients.",
            }
        ]

    # ---------------------------------------------------------------------------
    # 1. Helper Function Tests
    # ---------------------------------------------------------------------------

    def test_fmt_authors(self):
        self.assertEqual(_fmt_authors([]), "Unknown")
        self.assertEqual(_fmt_authors(["Alice"]), "Alice")
        self.assertEqual(_fmt_authors(["Alice", "Bob"]), "Alice & Bob")
        self.assertEqual(_fmt_authors(["Alice", "Bob", "Charlie"]), "Alice et al.")

    def test_build_context(self):
        context_str, numbered = build_context(self.mock_chunks)
        
        # Verify that indexing works [1..N]
        self.assertEqual(numbered[0]["number"], 1)
        self.assertEqual(numbered[1]["number"], 2)
        
        # Verify specific strings exist in the context block
        self.assertIn("[1] \"Attention Is All You Need\"", context_str)
        self.assertIn("Vaswani & Shazeer | 2017 | page 4", context_str)
        self.assertIn("Transformers process sequence tokens", context_str)

    def test_cited_numbers(self):
        # Tests normal brackets, multiple brackets, and custom bracket normalization styles
        self.assertEqual(_cited_numbers("According to [1] and [2], things happen."), [1, 2])
        self.assertEqual(_cited_numbers("As shown in [1, 2]."), [1, 2])
        self.assertEqual(_cited_numbers("Look here 【1】 and here [3]."), [1, 3])
        self.assertEqual(_cited_numbers("No citations present."), [])

    # ---------------------------------------------------------------------------
    # 2. Main API Pipeline Mock Tests
    # ---------------------------------------------------------------------------

    @patch("graphrag.answer.os.getenv")
    @patch("openai.OpenAI")
    def test_generate_answer_success(self, mock_openai_class, mock_getenv):
        # Setup environment and OpenAI client mocks
        mock_getenv.return_value = "sk-or-dummy-key"
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Simulate a successful LLM chat completion payload
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Transformers process data using parallel attention systems [1]."))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Execute
        result = generate_answer(
            query="How do transformers work?",
            ranked_chunks=self.mock_chunks,
            model="test-model"
        )

        # Assertions
        self.assertEqual(result["query"], "How do transformers work?")
        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["sources_used"], [1])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["paper_id"], "arxiv_1234")
        self.assertEqual(result["citations"][0]["title"], "Attention Is All You Need")

    @patch("graphrag.answer.os.getenv")
    @patch("openai.OpenAI")
    def test_generate_answer_fallback_on_rate_limit(self, mock_openai_class, mock_getenv):
        mock_getenv.return_value = "sk-or-dummy-key"
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Simulate: 1st model throws a RateLimitError, 2nd model returns successfully
        # Note: RateLimitError requires an HTTP response/message setup, using dummy values
        dummy_response = MagicMock(status_code=429)
        rate_limit_err = RateLimitError("Rate limit hit", response=dummy_response, body=None)
        
        mock_success_response = MagicMock()
        mock_success_response.choices = [
            MagicMock(message=MagicMock(content="Residual networks mitigate extreme gradient issues 【2】."))
        ]
        
        # Side effect sequence: Error on 1st execution, success on 2nd
        mock_client.chat.completions.create.side_effect = [rate_limit_err, mock_success_response]

        # Execute
        result = generate_answer(
            query="Why do we use resnets?",
            ranked_chunks=self.mock_chunks,
            model="primary-failing-model",
            model_fallbacks=["fallback-working-model"]
        )

        # Assertions
        # Check that normalization logic cleaned up the 【2】 citation into [2]
        self.assertEqual(result["answer"], "Residual networks mitigate extreme gradient issues [2].")
        self.assertEqual(result["model"], "fallback-working-model")  # Verified it successfully switched models
        self.assertEqual(result["sources_used"], [2])
        self.assertEqual(result["citations"][0]["paper_id"], "arxiv_5678")

    def test_generate_answer_missing_api_key(self):
        # Force API key context execution to be missing completely
        with patch("graphrag.answer.os.getenv", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                generate_answer("Test query?", self.mock_chunks, openrouter_api_key=None)
            self.assertIn("OPENROUTER_API_KEY is not set", str(ctx.exception))

    def test_generate_answer_empty_chunks(self):
        # Edge case: handling situation where no text contexts are delivered from Step 3
        with patch("graphrag.answer.os.getenv", return_value="sk-or-dummy"):
            result = generate_answer("Any query?", [])
            self.assertEqual(result["answer"], "No context chunks were provided.")
            self.assertEqual(result["citations"], [])
            self.assertEqual(result["chunks_used"], 0)



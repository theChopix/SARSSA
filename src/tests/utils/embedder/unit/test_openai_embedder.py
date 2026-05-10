"""Unit tests for utils.embedder.openai_embedder.

The langchain ``OpenAIEmbeddings`` dependency is patched so no
network calls are made.
"""

from unittest.mock import MagicMock, patch


class TestOpenAIEmbeddingLLMGenerateEmbeddings:
    """Tests for ``OpenAIEmbeddingLLM.generate_embeddings``."""

    @patch("utils.embedder.openai_embedder.OpenAIEmbeddings")
    def test_forwards_to_embed_documents_once(
        self,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """Verify a batch call delegates to ``embed_documents`` exactly once.

        Args:
            mock_embeddings_cls: Patched ``OpenAIEmbeddings`` class.
        """
        from utils.embedder.openai_embedder import OpenAIEmbeddingLLM

        inner = MagicMock()
        expected = [[0.1, 0.2], [0.3, 0.4]]
        inner.embed_documents.return_value = expected
        mock_embeddings_cls.return_value = inner

        embedder = OpenAIEmbeddingLLM(model="text-embedding-3-small")
        result = embedder.generate_embeddings(["hello", "world"])

        assert result == expected
        inner.embed_documents.assert_called_once_with(["hello", "world"])
        inner.embed_query.assert_not_called()

    @patch("utils.embedder.openai_embedder.OpenAIEmbeddings")
    def test_empty_input_returns_empty_list_without_call(
        self,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """Verify an empty input list short-circuits and avoids the network.

        Args:
            mock_embeddings_cls: Patched ``OpenAIEmbeddings`` class.
        """
        from utils.embedder.openai_embedder import OpenAIEmbeddingLLM

        inner = MagicMock()
        mock_embeddings_cls.return_value = inner

        embedder = OpenAIEmbeddingLLM(model="text-embedding-3-small")
        result = embedder.generate_embeddings([])

        assert result == []
        inner.embed_documents.assert_not_called()


class TestOpenAIEmbeddingLLMGenerateEmbedding:
    """Tests for the existing single-text ``generate_embedding`` path."""

    @patch("utils.embedder.openai_embedder.OpenAIEmbeddings")
    def test_forwards_to_embed_query(
        self,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """Verify the single-text call still delegates to ``embed_query``.

        Args:
            mock_embeddings_cls: Patched ``OpenAIEmbeddings`` class.
        """
        from utils.embedder.openai_embedder import OpenAIEmbeddingLLM

        inner = MagicMock()
        inner.embed_query.return_value = [0.5, 0.6]
        mock_embeddings_cls.return_value = inner

        embedder = OpenAIEmbeddingLLM(model="text-embedding-3-small")
        result = embedder.generate_embedding("hello")

        assert result == [0.5, 0.6]
        inner.embed_query.assert_called_once_with("hello")

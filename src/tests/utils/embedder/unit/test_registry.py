"""Unit tests for utils.embedder.registry.

The langchain ``OpenAIEmbeddings`` dependency is patched so the
factory does not try to contact the OpenAI API while constructing
real embedder instances.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestCreateEmbedder:
    """Tests for the ``create_embedder`` factory."""

    @patch("utils.embedder.openai_embedder.OpenAIEmbeddings")
    def test_openai_returns_openai_embedder(
        self,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """Verify ``"openai"`` resolves to ``OpenAIEmbeddingLLM``.

        Args:
            mock_embeddings_cls: Patched ``OpenAIEmbeddings`` class.
        """
        mock_embeddings_cls.return_value = MagicMock()

        from utils.embedder.openai_embedder import OpenAIEmbeddingLLM
        from utils.embedder.registry import create_embedder

        embedder = create_embedder("openai", "text-embedding-3-small")

        assert isinstance(embedder, OpenAIEmbeddingLLM)
        assert embedder.model == "text-embedding-3-small"

    def test_unknown_provider_raises_value_error(self) -> None:
        """Verify an unknown provider raises ``ValueError`` listing the known ones."""
        from utils.embedder.registry import create_embedder

        with pytest.raises(ValueError, match="Unknown embedding provider 'nope'"):
            create_embedder("nope", "any-model")

    def test_unknown_provider_error_lists_known_providers(self) -> None:
        """Verify the error message includes the known provider names."""
        from utils.embedder.registry import create_embedder

        with pytest.raises(ValueError) as exc_info:
            create_embedder("nope", "any-model")

        assert "openai" in str(exc_info.value)


class TestKnownProviders:
    """Tests for the ``known_providers`` helper."""

    def test_includes_openai(self) -> None:
        """Verify the registered OpenAI provider shows up in the listing."""
        from utils.embedder.registry import known_providers

        providers = known_providers()

        assert "openai" in providers

    def test_returns_sorted_list(self) -> None:
        """Verify the returned list is sorted (stable order for UI use)."""
        from utils.embedder.registry import known_providers

        providers = known_providers()

        assert providers == sorted(providers)

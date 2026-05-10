"""Unit tests for plugins.labeling_evaluation._embedding_cache.

Each test clears the underlying ``lru_cache`` to keep assertions
about call counts independent of test execution order. The
embedder factory is patched per test so no network call is made
and so we can count instantiations directly.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    """Clear the module-private LRU cache before each test.

    Yields:
        None: Pytest fixture marker.
    """
    from plugins.labeling_evaluation._embedding_cache import _embed_label_tuple

    _embed_label_tuple.cache_clear()
    yield
    _embed_label_tuple.cache_clear()


def _make_embedder(matrix: list[list[float]]) -> MagicMock:
    """Build a fake embedder whose ``generate_embeddings`` returns *matrix*.

    Args:
        matrix: Embedding rows the fake will yield on a batch call.

    Returns:
        MagicMock: Stand-in for an :class:`EmbeddingLLM` instance.
    """
    embedder = MagicMock()
    embedder.generate_embeddings.return_value = matrix
    return embedder


class TestEmbedLabelsCacheBehavior:
    """Tests covering hit/miss semantics of the LRU cache."""

    @patch("plugins.labeling_evaluation._embedding_cache.create_embedder")
    def test_repeated_call_hits_cache(
        self,
        mock_create_embedder: MagicMock,
    ) -> None:
        """Verify a second call with identical args does not reinvoke the embedder.

        Args:
            mock_create_embedder: Patched factory.
        """
        from plugins.labeling_evaluation._embedding_cache import embed_labels

        embedder = _make_embedder([[0.1, 0.2], [0.3, 0.4]])
        mock_create_embedder.return_value = embedder

        first = embed_labels(["a", "b"], "openai", "text-embedding-3-small")
        second = embed_labels(["a", "b"], "openai", "text-embedding-3-small")

        np.testing.assert_array_equal(first, second)
        mock_create_embedder.assert_called_once_with("openai", "text-embedding-3-small")
        embedder.generate_embeddings.assert_called_once()

    @patch("plugins.labeling_evaluation._embedding_cache.create_embedder")
    def test_reordered_labels_force_a_miss(
        self,
        mock_create_embedder: MagicMock,
    ) -> None:
        """Verify a different label order is treated as a different cache key.

        Args:
            mock_create_embedder: Patched factory.
        """
        from plugins.labeling_evaluation._embedding_cache import embed_labels

        embedder = _make_embedder([[0.0]])
        mock_create_embedder.return_value = embedder

        embed_labels(["a", "b"], "openai", "m")
        embed_labels(["b", "a"], "openai", "m")

        assert mock_create_embedder.call_count == 2
        assert embedder.generate_embeddings.call_count == 2

    @patch("plugins.labeling_evaluation._embedding_cache.create_embedder")
    def test_different_model_forces_a_miss(
        self,
        mock_create_embedder: MagicMock,
    ) -> None:
        """Verify swapping the model produces a separate cache entry.

        Args:
            mock_create_embedder: Patched factory.
        """
        from plugins.labeling_evaluation._embedding_cache import embed_labels

        embedder = _make_embedder([[0.0]])
        mock_create_embedder.return_value = embedder

        embed_labels(["a"], "openai", "m1")
        embed_labels(["a"], "openai", "m2")

        assert mock_create_embedder.call_count == 2

    @patch("plugins.labeling_evaluation._embedding_cache.create_embedder")
    def test_different_provider_forces_a_miss(
        self,
        mock_create_embedder: MagicMock,
    ) -> None:
        """Verify swapping the provider produces a separate cache entry.

        Protects against returning an OpenAI-computed matrix for a
        Gemini request (or vice-versa) when labels and model match.

        Args:
            mock_create_embedder: Patched factory.
        """
        from plugins.labeling_evaluation._embedding_cache import embed_labels

        embedder = _make_embedder([[0.0]])
        mock_create_embedder.return_value = embedder

        embed_labels(["a"], "openai", "m")
        embed_labels(["a"], "gemini", "m")

        assert mock_create_embedder.call_count == 2


class TestEmbedLabelsReturnShape:
    """Tests covering the shape and contents of the returned ndarray."""

    @patch("plugins.labeling_evaluation._embedding_cache.create_embedder")
    def test_returns_ndarray_with_expected_shape(
        self,
        mock_create_embedder: MagicMock,
    ) -> None:
        """Verify the result is an ``np.ndarray`` of shape ``(N, D)``.

        Args:
            mock_create_embedder: Patched factory.
        """
        from plugins.labeling_evaluation._embedding_cache import embed_labels

        rows = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_create_embedder.return_value = _make_embedder(rows)

        result = embed_labels(["a", "b"], "openai", "m")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, np.array(rows))


class TestEmbedLabelsErrors:
    """Tests covering input validation and error propagation."""

    @patch("plugins.labeling_evaluation._embedding_cache.create_embedder")
    def test_empty_input_raises_value_error(
        self,
        mock_create_embedder: MagicMock,
    ) -> None:
        """Verify an empty label list is rejected before any factory call.

        Args:
            mock_create_embedder: Patched factory; must not be called.
        """
        from plugins.labeling_evaluation._embedding_cache import embed_labels

        with pytest.raises(ValueError, match="label_texts"):
            embed_labels([], "openai", "m")

        mock_create_embedder.assert_not_called()

    def test_unknown_provider_propagates(self) -> None:
        """Verify a bogus provider surfaces the registry's ``ValueError``."""
        from plugins.labeling_evaluation._embedding_cache import embed_labels

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            embed_labels(["a"], "nonexistent-provider", "m")

"""Integration tests for the labeling-evaluation embedding cache.

These tests exercise the wiring between the two consumer paths
(``_embedding_map.compute_label_embedding_coords`` for the
embedding-map plugins and ``embed_labels`` directly for the
dendrogram plugin) and the shared
``plugins.labeling_evaluation._embedding_cache`` module.

The unit tests in ``test_embedding_cache.py`` already cover the
cache helper in isolation. The point of this module is to confirm
the cross-plugin claim that motivated the cache in the first
place: when both consumers run with identical
``(labels, provider, model)`` in the same worker process, they
collapse to a single embedder construction and a single API call.

The provider-switch test additionally protects against the most
dangerous failure mode the cache could have — silently returning
an OpenAI-computed matrix for a Gemini request (or vice-versa)
when labels and model match.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

LABELS: list[str] = ["alpha", "beta", "gamma"]
EMBEDDINGS: list[list[float]] = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
MODEL = "text-embedding-3-small"


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    """Clear the module-private LRU cache before and after each test.

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


class TestCrossPluginCacheHit:
    """The cache must coalesce calls across distinct consumer paths."""

    @patch("plugins.labeling_evaluation._embedding_map.umap.UMAP")
    @patch("plugins.labeling_evaluation._embedding_cache.create_embedder")
    def test_embedding_map_then_dendrogram_hit_same_cache_entry(
        self,
        mock_create_embedder: MagicMock,
        mock_umap_cls: MagicMock,
    ) -> None:
        """Verify the second consumer reuses the first consumer's embedding matrix.

        Args:
            mock_create_embedder: Patched factory; we count its
                invocations.
            mock_umap_cls: Patched UMAP class so the embedding-map
                helper does not run a real reduction.
        """
        from plugins.labeling_evaluation._embedding_cache import embed_labels
        from plugins.labeling_evaluation._embedding_map import (
            compute_label_embedding_coords,
        )

        embedder = _make_embedder(EMBEDDINGS)
        mock_create_embedder.return_value = embedder

        reducer = MagicMock()
        reducer.fit_transform.return_value = np.zeros((len(LABELS), 2))
        mock_umap_cls.return_value = reducer

        compute_label_embedding_coords(
            label_texts=LABELS,
            embedding_provider="openai",
            embedding_model=MODEL,
            umap_n_neighbors=5,
            umap_min_dist=0.1,
            umap_metric="cosine",
            umap_random_state=42,
        )

        second_result = embed_labels(LABELS, "openai", MODEL)

        np.testing.assert_array_equal(second_result, np.array(EMBEDDINGS))
        mock_create_embedder.assert_called_once_with("openai", MODEL)
        embedder.generate_embeddings.assert_called_once()


class TestProviderParticipatesInCacheKey:
    """Swapping the provider while keeping labels + model must force a miss."""

    @patch("plugins.labeling_evaluation._embedding_map.umap.UMAP")
    @patch("plugins.labeling_evaluation._embedding_cache.create_embedder")
    def test_provider_switch_triggers_fresh_embedder(
        self,
        mock_create_embedder: MagicMock,
        mock_umap_cls: MagicMock,
    ) -> None:
        """Verify ``"gemini"`` does not pick up an ``"openai"``-cached matrix.

        Args:
            mock_create_embedder: Patched factory; we set a
                ``side_effect`` that returns distinct mocks per
                provider so we can count per-provider calls.
            mock_umap_cls: Patched UMAP class.
        """
        from plugins.labeling_evaluation._embedding_map import (
            compute_label_embedding_coords,
        )

        openai_embedder = _make_embedder(EMBEDDINGS)
        gemini_embedder = _make_embedder([[0.9, 0.1], [0.1, 0.9], [0.4, 0.6]])

        def _fake_factory(provider: str, _model: str) -> MagicMock:
            if provider == "openai":
                return openai_embedder
            if provider == "gemini":
                return gemini_embedder
            raise AssertionError(f"unexpected provider: {provider}")

        mock_create_embedder.side_effect = _fake_factory

        reducer = MagicMock()
        reducer.fit_transform.return_value = np.zeros((len(LABELS), 2))
        mock_umap_cls.return_value = reducer

        compute_label_embedding_coords(
            label_texts=LABELS,
            embedding_provider="openai",
            embedding_model=MODEL,
            umap_n_neighbors=5,
            umap_min_dist=0.1,
            umap_metric="cosine",
            umap_random_state=42,
        )
        compute_label_embedding_coords(
            label_texts=LABELS,
            embedding_provider="gemini",
            embedding_model=MODEL,
            umap_n_neighbors=5,
            umap_min_dist=0.1,
            umap_metric="cosine",
            umap_random_state=42,
        )

        assert mock_create_embedder.call_count == 2
        openai_embedder.generate_embeddings.assert_called_once()
        gemini_embedder.generate_embeddings.assert_called_once()

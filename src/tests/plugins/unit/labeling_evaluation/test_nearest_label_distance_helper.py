"""Unit tests for plugins.labeling_evaluation._nearest_label_distance.

Patches ``embed_labels`` so no OpenAI call is made; validates that
``compute_nearest_distances`` returns the cosine distance to the
nearest past-label embedding for each current label, indexed in
input order.
"""

from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clear_embedding_cache() -> Iterator[None]:
    """Clear the shared LRU cache before and after each test.

    Yields:
        None: Pytest fixture marker.
    """
    from plugins.labeling_evaluation._embedding_cache import _embed_label_tuple

    _embed_label_tuple.cache_clear()
    yield
    _embed_label_tuple.cache_clear()


class TestComputeNearestDistances:
    """Tests for compute_nearest_distances."""

    @patch("plugins.labeling_evaluation._nearest_label_distance.embed_labels")
    def test_each_current_label_picks_minimum_distance_past_label(
        self,
        mock_embed_labels: object,
    ) -> None:
        """Verify every current row maps to its argmin past row by cosine distance.

        Args:
            mock_embed_labels: Patched embedder helper; alternates
                its return value between current / past calls based
                on the input list.
        """
        from plugins.labeling_evaluation._nearest_label_distance import (
            compute_nearest_distances,
        )

        # current[0] ≈ past[1] (both axis-aligned to y), current[1] ≈ past[0]
        # (both axis-aligned to x). cosine_distance is 0 between identical
        # unit vectors and 1 between orthogonal ones.
        current_embeddings = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        past_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        def fake_embed(texts: list[str], _provider: str, _model: str) -> np.ndarray:
            if texts == ["c0", "c1", "c2"]:
                return current_embeddings
            if texts == ["p0", "p1"]:
                return past_embeddings
            raise AssertionError(f"unexpected embed_labels call: {texts}")

        mock_embed_labels.side_effect = fake_embed

        result = compute_nearest_distances(
            current_label_texts=["c0", "c1", "c2"],
            past_label_texts=["p0", "p1"],
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        np.testing.assert_array_equal(result.nearest_past_indices, [1, 0, 0])
        np.testing.assert_array_almost_equal(result.distances, [0.0, 0.0, 1 - np.sqrt(0.5)])

    @patch("plugins.labeling_evaluation._nearest_label_distance.embed_labels")
    def test_distances_indexed_in_current_input_order(
        self,
        mock_embed_labels: object,
    ) -> None:
        """Verify the returned distances align row-wise with the current input.

        Args:
            mock_embed_labels: Patched embedder helper.
        """
        from plugins.labeling_evaluation._nearest_label_distance import (
            compute_nearest_distances,
        )

        # Sized so each current row has exactly one identical past row,
        # at a *different* past index per current row — a swap that
        # would be invisible if the implementation accidentally sorted.
        current_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        past_embeddings = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        def fake_embed(texts: list[str], _provider: str, _model: str) -> np.ndarray:
            return current_embeddings if "c" in texts[0] else past_embeddings

        mock_embed_labels.side_effect = fake_embed

        result = compute_nearest_distances(
            current_label_texts=["c0", "c1", "c2"],
            past_label_texts=["p0", "p1", "p2"],
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        # Current row i maps to past row (i + 1) % 3 (the swap above).
        np.testing.assert_array_equal(result.nearest_past_indices, [1, 2, 0])
        np.testing.assert_array_almost_equal(result.distances, [0.0, 0.0, 0.0])

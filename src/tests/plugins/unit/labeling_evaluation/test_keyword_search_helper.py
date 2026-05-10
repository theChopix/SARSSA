"""Unit tests for plugins.labeling_evaluation._keyword_search.

The helper does cosine ranking on raw NumPy arrays, no IO. Tests
verify the math (top-k by cosine similarity desc), edge cases (k
larger than corpus), and input guards.
"""

import numpy as np
import pytest


class TestFindTopKNearest:
    """Tests for find_top_k_nearest."""

    def test_returns_indices_in_descending_similarity_order(self) -> None:
        """Verify the topmost index is the most-similar label."""
        from plugins.labeling_evaluation._keyword_search import find_top_k_nearest

        keyword = np.array([1.0, 0.0])
        labels = np.array(
            [
                [0.0, 1.0],  # orthogonal, sim 0
                [1.0, 0.0],  # identical, sim 1
                [0.5, 0.866],  # 60°, sim 0.5
            ]
        )

        indices, similarities = find_top_k_nearest(keyword, labels, k=3)

        np.testing.assert_array_equal(indices, [1, 2, 0])
        np.testing.assert_array_almost_equal(similarities, [1.0, 0.5, 0.0], decimal=3)

    def test_k_larger_than_corpus_clamps_to_corpus_size(self) -> None:
        """Verify k > N returns N items, not more, not an error."""
        from plugins.labeling_evaluation._keyword_search import find_top_k_nearest

        keyword = np.array([1.0, 0.0])
        labels = np.array([[0.0, 1.0], [1.0, 0.0]])

        indices, similarities = find_top_k_nearest(keyword, labels, k=99)

        assert len(indices) == 2
        assert len(similarities) == 2

    def test_zero_k_raises(self) -> None:
        """Verify k <= 0 is rejected with a clear error."""
        from plugins.labeling_evaluation._keyword_search import find_top_k_nearest

        with pytest.raises(ValueError, match="k must be positive"):
            find_top_k_nearest(np.array([1.0]), np.array([[1.0]]), k=0)

    def test_empty_corpus_raises(self) -> None:
        """Verify an empty label_embeddings matrix is rejected."""
        from plugins.labeling_evaluation._keyword_search import find_top_k_nearest

        with pytest.raises(ValueError, match="must not be empty"):
            find_top_k_nearest(np.array([1.0, 0.0]), np.empty((0, 2)), k=3)

    def test_normalisation_makes_magnitude_irrelevant(self) -> None:
        """Verify scaling label vectors does not affect the ranking."""
        from plugins.labeling_evaluation._keyword_search import find_top_k_nearest

        keyword = np.array([1.0, 0.0])
        labels = np.array(
            [
                [10.0, 0.0],  # same direction as keyword, sim 1
                [0.0, 100.0],  # orthogonal regardless of magnitude, sim 0
            ]
        )

        indices, _ = find_top_k_nearest(keyword, labels, k=2)
        np.testing.assert_array_equal(indices, [0, 1])

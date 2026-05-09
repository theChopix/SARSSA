"""Unit tests for plugins.labeling_evaluation._embedding_map."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestComputeLabelEmbeddingCoords:
    """Tests for the shared embedding + UMAP helper."""

    @patch("plugins.labeling_evaluation._embedding_map.umap.UMAP")
    @patch("plugins.labeling_evaluation._embedding_map.OpenAIEmbeddingLLM")
    def test_returns_umap_2d_coords(
        self,
        mock_embedder_cls: MagicMock,
        mock_umap_cls: MagicMock,
    ) -> None:
        """Verify the helper plumbs embeddings through UMAP and returns coords."""
        from plugins.labeling_evaluation._embedding_map import (
            compute_label_embedding_coords,
        )

        embedder = MagicMock()
        embedder.generate_embedding.side_effect = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        mock_embedder_cls.return_value = embedder

        reducer = MagicMock()
        expected_coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        reducer.fit_transform.return_value = expected_coords
        mock_umap_cls.return_value = reducer

        result = compute_label_embedding_coords(
            label_texts=["a", "b", "c"],
            embedding_model="text-embedding-3-small",
            umap_n_neighbors=10,
            umap_min_dist=0.05,
            umap_metric="cosine",
            umap_random_state=7,
        )

        np.testing.assert_array_equal(result, expected_coords)
        mock_embedder_cls.assert_called_once_with(model="text-embedding-3-small")
        assert embedder.generate_embedding.call_count == 3

        mock_umap_cls.assert_called_once_with(
            n_components=2,
            n_neighbors=10,
            min_dist=0.05,
            metric="cosine",
            random_state=7,
        )
        passed = reducer.fit_transform.call_args.args[0]
        assert passed.shape == (3, 3)

    @patch("plugins.labeling_evaluation._embedding_map.umap.UMAP")
    @patch("plugins.labeling_evaluation._embedding_map.OpenAIEmbeddingLLM")
    def test_preserves_input_order(
        self,
        mock_embedder_cls: MagicMock,
        mock_umap_cls: MagicMock,
    ) -> None:
        """Verify embeddings are passed to UMAP in the input order."""
        from plugins.labeling_evaluation._embedding_map import (
            compute_label_embedding_coords,
        )

        embedder = MagicMock()
        embedder.generate_embedding.side_effect = lambda t: [float(ord(t[0]))]
        mock_embedder_cls.return_value = embedder

        reducer = MagicMock()
        reducer.fit_transform.return_value = np.zeros((3, 2))
        mock_umap_cls.return_value = reducer

        compute_label_embedding_coords(
            label_texts=["c", "a", "b"],
            embedding_model="m",
            umap_n_neighbors=5,
            umap_min_dist=0.1,
            umap_metric="cosine",
            umap_random_state=1,
        )

        passed = reducer.fit_transform.call_args.args[0]
        assert passed[0, 0] == float(ord("c"))
        assert passed[1, 0] == float(ord("a"))
        assert passed[2, 0] == float(ord("b"))

    def test_empty_input_raises(self) -> None:
        """Verify an empty label list is rejected up front."""
        from plugins.labeling_evaluation._embedding_map import (
            compute_label_embedding_coords,
        )

        with pytest.raises(ValueError, match="label_texts"):
            compute_label_embedding_coords(
                label_texts=[],
                embedding_model="m",
                umap_n_neighbors=5,
                umap_min_dist=0.1,
                umap_metric="cosine",
                umap_random_state=1,
            )

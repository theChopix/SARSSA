"""Unit tests for labeling_evaluation.single.embedding_map_with_keyword_search.

Patches both ``embed_labels`` and ``compute_label_embedding_coords``
inside the plugin module so neither OpenAI nor real UMAP is
exercised.  Verifies top-k selection, trace structure (skip-from-
plain), sidebar item indices, and output params.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _build_plugin() -> Any:
    """Build a Plugin instance with state mirroring load_context().

    Returns:
        Plugin: Single-mode keyword-search plugin instance.
    """
    from plugins.labeling_evaluation.single.embedding_map_with_keyword_search.embedding_map_with_keyword_search import (  # noqa: E501
        Plugin,
    )

    plugin = Plugin()
    plugin.neuron_labels = {
        "0": "alpha",
        "1": "beta",
        "2": "gamma",
        "3": "delta",
    }
    plugin.neuron_ids = ["0", "1", "2", "3"]
    plugin.label_texts = ["alpha", "beta", "gamma", "delta"]
    return plugin


class TestSingleKeywordSearchRun:
    """Tests for the single plugin's run() method."""

    @patch(
        "plugins.labeling_evaluation.single.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.compute_label_embedding_coords"
    )
    @patch(
        "plugins.labeling_evaluation.single.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.embed_labels"
    )
    def test_top_k_picks_most_similar_labels(
        self,
        mock_embed_labels: MagicMock,
        mock_compute_coords: MagicMock,
    ) -> None:
        """Verify the top-k records contain the labels closest in cosine sim.

        Args:
            mock_embed_labels: Patched cache helper; we control the
                full embedding matrix it returns.
            mock_compute_coords: Patched UMAP helper; we control the
                shared coordinate space.
        """
        # Embeddings: 4 labels (rows 0-3) + keyword (row 4).
        # Keyword direction is [1, 0]; alpha/gamma share that
        # direction (similarity 1), beta/delta are orthogonal (0).
        embeddings = np.array(
            [
                [1.0, 0.0],  # alpha
                [0.0, 1.0],  # beta
                [1.0, 0.0],  # gamma
                [0.0, 1.0],  # delta
                [1.0, 0.0],  # keyword
            ]
        )
        mock_embed_labels.return_value = embeddings
        mock_compute_coords.return_value = np.array(
            [
                [10.0, 11.0],  # alpha
                [20.0, 21.0],  # beta
                [30.0, 31.0],  # gamma
                [40.0, 41.0],  # delta
                [50.0, 51.0],  # keyword
            ]
        )

        plugin = _build_plugin()
        plugin.run(keyword="search me", k=2)

        import json as _json

        records = _json.loads(plugin.top_k_matches)
        match_ids = sorted(r["neuron_id"] for r in records)
        # alpha (id 0) and gamma (id 2) tie for similarity 1; both
        # beat beta and delta (similarity 0).
        assert match_ids == ["0", "2"]

        assert plugin.keyword_param == "search me"
        assert plugin.k_param == 2
        assert plugin.num_top_k_matches_param == 2
        assert plugin.num_neurons == 4

    @patch(
        "plugins.labeling_evaluation.single.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.compute_label_embedding_coords"
    )
    @patch(
        "plugins.labeling_evaluation.single.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.embed_labels"
    )
    def test_figure_has_three_traces_with_skip_from_plain(
        self,
        mock_embed_labels: MagicMock,
        mock_compute_coords: MagicMock,
    ) -> None:
        """Verify plain trace excludes top-k indices and keyword is a star.

        Args:
            mock_embed_labels: Patched cache helper.
            mock_compute_coords: Patched UMAP helper.
        """
        embeddings = np.array(
            [
                [1.0, 0.0],  # alpha — top-1
                [0.0, 1.0],  # beta
                [0.0, 0.5],  # gamma
                [0.0, 0.2],  # delta
                [1.0, 0.0],  # keyword
            ]
        )
        mock_embed_labels.return_value = embeddings
        mock_compute_coords.return_value = np.array(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
            ]
        )

        plugin = _build_plugin()
        plugin.run(keyword="search me", k=1)

        traces = plugin._fig.data
        assert len(traces) == 3

        # Plain trace has 3 points (4 labels minus 1 highlighted).
        plain = traces[0]
        assert len(plain.x) == 3

        # Highlight trace has the 1 top match.
        highlight = traces[1]
        assert len(highlight.x) == 1
        assert highlight.x[0] == 10.0  # alpha's x coord

        # Keyword trace is a single star.
        keyword = traces[2]
        assert len(keyword.x) == 1
        assert keyword.marker.symbol == "star"
        assert keyword.x[0] == 50.0

    @patch(
        "plugins.labeling_evaluation.single.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.compute_label_embedding_coords"
    )
    @patch(
        "plugins.labeling_evaluation.single.embedding_map_with_keyword_search."
        "embedding_map_with_keyword_search.embed_labels"
    )
    def test_sidebar_items_target_highlight_trace_index(
        self,
        mock_embed_labels: MagicMock,
        mock_compute_coords: MagicMock,
    ) -> None:
        """Verify each sidebar item points at the highlight trace's index.

        With a non-empty plain trace the highlight trace lives at
        index 1; the JS handlers count on that.

        Args:
            mock_embed_labels: Patched cache helper.
            mock_compute_coords: Patched UMAP helper.
        """
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.2, 0.8],
                [1.0, 0.0],
            ]
        )
        mock_embed_labels.return_value = embeddings
        mock_compute_coords.return_value = np.zeros((5, 2))

        plugin = _build_plugin()
        plugin.run(keyword="search", k=2)

        sidebars = plugin._sidebars
        assert len(sidebars) == 1
        items = sidebars[0].items
        assert len(items) == 2
        for i, item in enumerate(items):
            assert item.trace_index == 1
            assert item.point_index == i

    def test_empty_keyword_raises(self) -> None:
        """Verify a blank keyword is rejected before any embedding work."""
        plugin = _build_plugin()
        with pytest.raises(ValueError, match="keyword must not be empty"):
            plugin.run(keyword="   ")

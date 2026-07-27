"""Unit tests for plugins.labeling_evaluation.single.dendrogram.

Patches only the embedding cache helper (no OpenAI call); scipy,
matplotlib and plotly run for real on tiny label sets so the tests
cover deduplication, both axis modes, and the rendered artifacts.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _build_plugin() -> Any:
    """Build a Plugin instance with duplicate and unlabeled neurons preset.

    Returns:
        Plugin: Dendrogram plugin instance ready for ``run()``, as
            ``load_context`` would have populated it.
    """
    from plugins.labeling_evaluation.single.dendrogram.dendrogram import Plugin

    plugin = Plugin()
    plugin.neuron_labels = {
        "0": {"label": "alpha"},
        "1": {"label": "beta"},
        "2": {"label": "alpha"},
        "3": {"label": None},
        "4": {"label": "gamma"},
    }
    plugin.neuron_ids = ["0", "1", "2", "3", "4"]
    plugin.unique_labels = ["alpha", "beta", "gamma"]
    plugin.label_id_groups = [["0", "2"], ["1"], ["4"]]
    plugin.num_unlabeled = 1
    return plugin


_EMBEDDINGS = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.4]])


class TestLoadContext:
    """Tests for label grouping in load_context."""

    @patch("plugins.plugin_interface.MLflowRunLoader")
    def test_groups_duplicates_and_skips_unlabeled(self, mock_loader_cls: MagicMock) -> None:
        """Verify identical labels group into one entry and None is excluded."""
        from plugins.labeling_evaluation.single.dendrogram.dendrogram import Plugin

        loader = MagicMock()
        loader.get_json_artifact.return_value = {
            "0": {"label": "alpha"},
            "1": {"label": "beta"},
            "2": {"label": "alpha"},
            "3": {"label": None},
        }
        mock_loader_cls.return_value = loader

        plugin = Plugin()
        plugin.load_context({"neuron_labeling": {"run_id": "nl"}})

        assert plugin.unique_labels == ["alpha", "beta"]
        assert plugin.label_id_groups == [["0", "2"], ["1"]]
        assert plugin.num_unlabeled == 1


class TestDendrogramRun:
    """Tests for the dendrogram plugin's run() method."""

    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.embed_labels")
    def test_embeds_unique_labels_only(self, mock_embed: MagicMock) -> None:
        """Verify the embedder receives the deduplicated label list."""
        mock_embed.return_value = _EMBEDDINGS

        plugin = _build_plugin()
        plugin.run(embedding_provider="openai", embedding_model="text-embedding-3-small")

        mock_embed.assert_called_once_with(
            ["alpha", "beta", "gamma"],
            "openai",
            "text-embedding-3-small",
            plugin.notifier,
        )

    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.embed_labels")
    def test_output_params_and_artifacts(self, mock_embed: MagicMock) -> None:
        """Verify counts and the HTML + PDF artifacts are produced."""
        mock_embed.return_value = _EMBEDDINGS

        plugin = _build_plugin()
        plugin.run()

        assert plugin.num_neurons == 5
        assert plugin.num_unique_labels == 3
        assert plugin.linkage_matrix.shape == (2, 4)
        assert plugin.dendrogram_pdf[:4] == b"%PDF"
        assert "<html" in plugin.dendrogram_html
        assert not hasattr(plugin, "dendrogram_svg")

    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.embed_labels")
    def test_html_has_leaf_labels_and_neuron_index(self, mock_embed: MagicMock) -> None:
        """Verify searchable leaf texts and the per-neuron index are in the HTML."""
        mock_embed.return_value = _EMBEDDINGS

        plugin = _build_plugin()
        plugin.run()

        html = plugin.dendrogram_html
        assert "alpha ×2 [n0, n2]" in html
        assert "beta [n1]" in html
        # Every labeled neuron stays findable via the index, duplicates included.
        for nid, label in [("0", "alpha"), ("2", "alpha"), ("1", "beta"), ("4", "gamma")]:
            assert f"[neuron {nid}] {label}" in html
        assert "[neuron 3]" not in html  # unlabeled neuron is excluded

    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.embed_labels")
    def test_depth_axis_mode_runs_and_titles_axis(self, mock_embed: MagicMock) -> None:
        """Verify axis_mode='depth' renders with the depth axis title."""
        mock_embed.return_value = _EMBEDDINGS

        plugin = _build_plugin()
        plugin.run(axis_mode="depth")

        assert "Depth below root" in plugin.dendrogram_html

    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.embed_labels")
    def test_unknown_axis_mode_raises(self, mock_embed: MagicMock) -> None:
        """Verify an invalid axis_mode fails fast."""
        mock_embed.return_value = _EMBEDDINGS

        plugin = _build_plugin()
        with pytest.raises(ValueError, match="axis_mode"):
            plugin.run(axis_mode="sideways")


class TestDepthLinkage:
    """Tests for the depth-based height transform."""

    def test_depth_heights_replace_distances(self) -> None:
        """Verify a chain tree gets uniform depth-spaced heights."""
        from plugins.labeling_evaluation.single.dendrogram.dendrogram import (
            _depth_linkage,
        )

        # Chain: (0,1) -> +2 -> +3 with skewed distances.
        Z = np.array(
            [
                [0.0, 1.0, 0.001, 2.0],
                [2.0, 4.0, 0.002, 3.0],
                [3.0, 5.0, 0.9, 4.0],
            ]
        )
        Zd = _depth_linkage(Z)
        # Depths from root: node6=0, node5=1, node4=2 -> inverted heights 1,2,3.
        np.testing.assert_array_equal(Zd[:, 2], [1.0, 2.0, 3.0])
        # Everything else is untouched.
        np.testing.assert_array_equal(Zd[:, [0, 1, 3]], Z[:, [0, 1, 3]])


class TestLeafTexts:
    """Tests for capped (HTML) vs expanded (PDF) leaf label ids."""

    def test_cap_and_expand(self) -> None:
        """Verify the default caps ids and max_ids=None expands them all."""
        from plugins.labeling_evaluation.single.dendrogram.dendrogram import (
            _MAX_IDS_IN_LEAF,
            Plugin,
        )

        big = [str(i) for i in range(_MAX_IDS_IN_LEAF + 5)]
        plugin = Plugin()
        plugin.unique_labels = ["shared", "solo"]
        plugin.label_id_groups = [big, ["99"]]

        capped = plugin._leaf_texts()
        assert "+5 more]" in capped[0]
        assert capped[1] == "solo [n99]"

        expanded = plugin._leaf_texts(max_ids=None)
        assert "more" not in expanded[0]
        for nid in big:
            assert f"n{nid}" in expanded[0]

    def test_longest_label_width_grows_with_text(self) -> None:
        """Verify the width helper is empty-safe and scales with length."""
        from plugins.labeling_evaluation.single.dendrogram.dendrogram import (
            _longest_label_width_in,
        )

        assert _longest_label_width_in([], 6) == 0.0
        short = _longest_label_width_in(["ab"], 6)
        wide = _longest_label_width_in(["ab", "a" * 200], 6)
        assert wide > short > 0.0


class TestDendrogramIOSpec:
    """Tests for the plugin's declarative I/O contract."""

    def test_output_params_and_artifacts_declared(self) -> None:
        """Verify declared outputs match the new artifact set."""
        from plugins.labeling_evaluation.single.dendrogram.dendrogram import Plugin

        param_keys = [spec.key for spec in Plugin.io_spec.output_params]
        assert param_keys == ["num_neurons", "num_unique_labels"]
        artifact_files = [spec.filename for spec in Plugin.io_spec.output_artifacts]
        assert "dendrogram.html" in artifact_files
        assert "dendrogram.pdf" in artifact_files
        assert "dendrogram.svg" not in artifact_files
        # The PDF is still produced but not shown in the results panel.
        display_files = [f.filename for f in Plugin.io_spec.display.files]
        assert display_files == ["dendrogram.html"]

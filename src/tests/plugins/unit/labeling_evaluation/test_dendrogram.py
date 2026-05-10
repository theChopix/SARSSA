"""Unit tests for plugins.labeling_evaluation.single.dendrogram.

Patches the cache helper so no OpenAI call is made and asserts the
plugin plumbs the new ``embedding_provider`` param through to the
helper, records it as an output param, and feeds the resulting
embeddings into ``scipy``'s linkage builder.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np


def _build_plugin() -> Any:
    """Build a Plugin instance with current-side state preset.

    Returns:
        Plugin: Dendrogram plugin instance ready for ``run()``,
            with ``neuron_labels`` / ``neuron_ids`` /
            ``label_texts`` filled in as ``load_context`` would.
    """
    from plugins.labeling_evaluation.single.dendrogram.dendrogram import Plugin

    plugin = Plugin()
    plugin.neuron_labels = {"0": "alpha", "1": "beta", "2": "gamma"}
    plugin.neuron_ids = ["0", "1", "2"]
    plugin.label_texts = ["alpha", "beta", "gamma"]
    return plugin


class TestDendrogramRun:
    """Tests for the dendrogram plugin's run() method."""

    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.dendrogram")
    @patch(
        "plugins.labeling_evaluation.single.dendrogram.dendrogram.linkage",
        return_value=np.zeros((2, 4)),
    )
    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.embed_labels")
    def test_calls_embed_labels_with_provider_then_model(
        self,
        mock_embed_labels: MagicMock,
        _mock_linkage: MagicMock,
        _mock_dendrogram_fn: MagicMock,
    ) -> None:
        """Verify the plugin forwards ``(label_texts, provider, model)`` positionally.

        Args:
            mock_embed_labels: Patched cache helper.
            _mock_linkage: Patched linkage builder (unused; just
                avoids running scipy on a zero-row array).
            _mock_dendrogram_fn: Patched scipy dendrogram drawer
                (unused; short-circuits validation of the mocked
                linkage matrix).
        """
        mock_embed_labels.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        plugin = _build_plugin()
        plugin.run(embedding_provider="openai", embedding_model="text-embedding-3-small")

        mock_embed_labels.assert_called_once_with(
            ["alpha", "beta", "gamma"],
            "openai",
            "text-embedding-3-small",
        )

    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.dendrogram")
    @patch(
        "plugins.labeling_evaluation.single.dendrogram.dendrogram.linkage",
        return_value=np.zeros((2, 4)),
    )
    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.embed_labels")
    def test_records_embedding_provider_param(
        self,
        mock_embed_labels: MagicMock,
        _mock_linkage: MagicMock,
        _mock_dendrogram_fn: MagicMock,
    ) -> None:
        """Verify the chosen provider is exposed as an MLflow output param.

        Args:
            mock_embed_labels: Patched cache helper.
            _mock_linkage: Patched linkage builder.
            _mock_dendrogram_fn: Patched scipy dendrogram drawer.
        """
        mock_embed_labels.return_value = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])

        plugin = _build_plugin()
        plugin.run(embedding_provider="gemini")

        assert plugin.embedding_provider_param == "gemini"
        assert plugin.embedding_model_param == "text-embedding-3-small"

    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.dendrogram")
    @patch(
        "plugins.labeling_evaluation.single.dendrogram.dendrogram.linkage",
    )
    @patch("plugins.labeling_evaluation.single.dendrogram.dendrogram.embed_labels")
    def test_linkage_receives_cosine_distances_of_embeddings(
        self,
        mock_embed_labels: MagicMock,
        mock_linkage: MagicMock,
        _mock_dendrogram_fn: MagicMock,
    ) -> None:
        """Verify ``linkage`` is fed the pairwise cosine distances of the embeddings.

        Args:
            mock_embed_labels: Patched cache helper.
            mock_linkage: Patched linkage builder; we inspect its
                first positional argument.
            _mock_dendrogram_fn: Patched scipy dendrogram drawer.
        """
        from scipy.spatial.distance import pdist

        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        mock_embed_labels.return_value = embeddings
        mock_linkage.return_value = np.zeros((2, 4))

        plugin = _build_plugin()
        plugin.run()

        passed_distances = mock_linkage.call_args.args[0]
        np.testing.assert_array_almost_equal(
            passed_distances,
            pdist(embeddings, metric="cosine"),
        )


class TestDendrogramIOSpec:
    """Tests for the plugin's declarative I/O contract."""

    def test_embedding_provider_in_output_params(self) -> None:
        """Verify the new ``embedding_provider`` shows up in ``output_params``."""
        from plugins.labeling_evaluation.single.dendrogram.dendrogram import Plugin

        keys = [spec.key for spec in Plugin.io_spec.output_params]
        assert "embedding_provider" in keys
        assert keys.index("embedding_provider") < keys.index("embedding_model")

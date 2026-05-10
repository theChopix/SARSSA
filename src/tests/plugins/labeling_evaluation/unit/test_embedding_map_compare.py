"""Unit tests for labeling_evaluation.compare.embedding_map.

Mocks ``MLflowRunLoader`` (so the past side never touches MLflow) and
the embedding helper (so OpenAI / UMAP are not invoked).  The
current-side ``neuron_labels`` is set on the plugin instance directly,
mirroring what ``load_context`` would have populated.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _build_plugin() -> Any:
    """Build a Plugin instance with current-side state preset.

    Returns:
        Plugin: Compare plugin instance ready for ``run()``; the past
            side is reached via the patched MLflowRunLoader.
    """
    from plugins.labeling_evaluation.compare.embedding_map.embedding_map import Plugin

    plugin = Plugin()
    plugin.neuron_labels = {"0": "current_a", "1": "current_b"}
    plugin.current_neuron_ids = ["0", "1"]
    plugin.current_label_texts = ["current_a", "current_b"]
    return plugin


def _make_past_loader(
    *,
    context: dict[str, Any],
    past_neuron_labels: dict[str, str],
) -> MagicMock:
    """Build a MagicMock that mimics MLflowRunLoader for the past side."""

    def get_json(filename: str) -> Any:
        match filename:
            case "context.json":
                return context
            case "neuron_labels.json":
                return past_neuron_labels
            case other:
                raise AssertionError(f"unexpected json artifact: {other}")

    loader = MagicMock()
    loader.get_json_artifact.side_effect = get_json
    return loader


class TestCompareEmbeddingMapRun:
    """Tests for the compare plugin's run() method."""

    @patch(
        "plugins.labeling_evaluation.compare.embedding_map.embedding_map."
        "compute_label_embedding_coords"
    )
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_run_splits_combined_coords_into_two_traces(
        self,
        mock_compare_loader_cls: MagicMock,
        mock_compute: MagicMock,
    ) -> None:
        """Verify combined UMAP output is split row-wise back into per-side coords."""
        past_context = {"neuron_labeling": {"run_id": "nl_past"}}
        past_loader = _make_past_loader(
            context=past_context,
            past_neuron_labels={"7": "past_a", "9": "past_b", "12": "past_c"},
        )
        mock_compare_loader_cls.return_value = past_loader

        combined = np.array(
            [
                [10.0, 11.0],  # current 0
                [20.0, 21.0],  # current 1
                [30.0, 31.0],  # past 7
                [40.0, 41.0],  # past 9
                [50.0, 51.0],  # past 12
            ]
        )
        mock_compute.return_value = combined

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            embedding_model="text-embedding-3-small",
            umap_n_neighbors=10,
            umap_min_dist=0.05,
            umap_metric="cosine",
            umap_random_state=7,
            point_size=12,
        )

        np.testing.assert_array_equal(plugin.current_umap_coords, combined[:2])
        np.testing.assert_array_equal(plugin.past_umap_coords, combined[2:])

        passed_kwargs = mock_compute.call_args.kwargs
        assert passed_kwargs["label_texts"] == [
            "current_a",
            "current_b",
            "past_a",
            "past_b",
            "past_c",
        ]
        assert passed_kwargs["embedding_provider"] == "openai"
        assert passed_kwargs["embedding_model"] == "text-embedding-3-small"
        assert passed_kwargs["umap_n_neighbors"] == 10
        assert passed_kwargs["umap_min_dist"] == 0.05
        assert passed_kwargs["umap_metric"] == "cosine"
        assert passed_kwargs["umap_random_state"] == 7

        assert plugin.past_run_id_param == "parent_xyz"
        assert plugin.embedding_provider_param == "openai"
        assert plugin.embedding_model_param == "text-embedding-3-small"
        assert plugin.num_neurons_current_param == 2
        assert plugin.num_neurons_past_param == 3

    @patch(
        "plugins.labeling_evaluation.compare.embedding_map.embedding_map."
        "compute_label_embedding_coords"
    )
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_run_builds_two_scatter_traces(
        self,
        mock_compare_loader_cls: MagicMock,
        mock_compute: MagicMock,
    ) -> None:
        """Verify the output figure has exactly two named scatter traces."""
        past_context = {"neuron_labeling": {"run_id": "nl_past"}}
        mock_compare_loader_cls.return_value = _make_past_loader(
            context=past_context,
            past_neuron_labels={"7": "past_a"},
        )
        mock_compute.return_value = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            point_size=8,
        )

        traces = plugin._fig.data
        assert len(traces) == 2
        names = sorted(t.name for t in traces)
        assert names == ["Current Run", "Past Run"]
        colors = {t.name: t.marker.color for t in traces}
        assert colors["Current Run"] != colors["Past Run"]

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_missing_past_run_id_kwarg_raises(
        self,
        mock_loader_cls: MagicMock,  # noqa: ARG002
    ) -> None:
        """Verify omitting past_run_id surfaces MissingContextError via the wrapper."""
        from plugins.plugin_interface import MissingContextError

        plugin = _build_plugin()
        kwargs_without_past_run_id: dict[str, Any] = {
            "embedding_model": "m",
        }
        with pytest.raises(MissingContextError, match="past_run_id"):
            plugin.run(**kwargs_without_past_run_id)


class TestCompareEmbeddingMapHints:
    """Tests for the auto-generated UI hint set."""

    def test_past_runs_dropdown_hint_present(self) -> None:
        """Verify BaseComparePlugin injected a PastRunsDropdownHint for past_run_id."""
        from plugins.labeling_evaluation.compare.embedding_map.embedding_map import (
            Plugin,
        )
        from plugins.plugin_interface import PastRunsDropdownHint

        hints = Plugin.io_spec.param_ui_hints
        past_hints = [
            h
            for h in hints
            if isinstance(h, PastRunsDropdownHint) and h.param_name == "past_run_id"
        ]
        assert len(past_hints) == 1
        assert past_hints[0].required_steps == ["neuron_labeling"]

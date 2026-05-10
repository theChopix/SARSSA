"""Unit tests for labeling_evaluation.compare.nearest_label_distance_bars.

Mocks ``MLflowRunLoader`` (so the past side never touches MLflow) and
the shared ``compute_nearest_distances`` helper (so OpenAI is not
invoked).  The current-side ``neuron_labels`` is set on the plugin
instance directly, mirroring what ``load_context`` would have
populated.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _build_plugin() -> Any:
    """Build a Plugin instance with current-side state preset.

    Returns:
        Plugin: Bars plugin instance ready for ``run()``.
    """
    from plugins.labeling_evaluation.compare.nearest_label_distance_bars.nearest_label_distance_bars import (  # noqa: E501
        Plugin,
    )

    plugin = Plugin()
    plugin.neuron_labels = {"0": "current_a", "1": "current_b", "2": "current_c"}
    plugin.current_neuron_ids = ["0", "1", "2"]
    plugin.current_label_texts = ["current_a", "current_b", "current_c"]
    return plugin


def _make_past_loader(
    *,
    context: dict[str, Any],
    past_neuron_labels: dict[str, str],
) -> MagicMock:
    """Build a MagicMock that mimics MLflowRunLoader for the past side.

    Args:
        context: Past-run pipeline context to return for ``context.json``.
        past_neuron_labels: Past-run neuron labels to return for
            ``neuron_labels.json``.

    Returns:
        MagicMock: Stand-in :class:`MLflowRunLoader` instance.
    """

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


class TestBarsRun:
    """Tests for the bars plugin's run() method."""

    @patch(
        "plugins.labeling_evaluation.compare.nearest_label_distance_bars."
        "nearest_label_distance_bars.compute_nearest_distances"
    )
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_run_records_per_label_distances_and_pairings(
        self,
        mock_compare_loader_cls: MagicMock,
        mock_compute: MagicMock,
    ) -> None:
        """Verify per-label JSON, sort order, and output params.

        Args:
            mock_compare_loader_cls: Patched ``MLflowRunLoader`` class
                used by the compare wrapper to load the past context
                and the past ``neuron_labels.json``.
            mock_compute: Patched compute helper; we control distances
                and the nearest-past index per current label.
        """
        past_context = {"neuron_labeling": {"run_id": "nl_past"}}
        mock_compare_loader_cls.return_value = _make_past_loader(
            context=past_context,
            past_neuron_labels={"7": "past_a", "9": "past_b"},
        )

        from plugins.labeling_evaluation._nearest_label_distance import (
            NearestLabelDistances,
        )

        # Current order: c0 → 0.10 (past 7), c1 → 0.50 (past 9), c2 → 0.30 (past 7).
        # Sorted desc, the bar chart should read c1 (0.50), c2 (0.30), c0 (0.10).
        mock_compute.return_value = NearestLabelDistances(
            distances=np.array([0.10, 0.50, 0.30]),
            nearest_past_indices=np.array([0, 1, 0]),
        )

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        records = json.loads(plugin.nearest_distances)
        neuron_order = [r["neuron_id"] for r in records]
        assert neuron_order == ["1", "2", "0"]

        record_by_id = {r["neuron_id"]: r for r in records}
        assert record_by_id["0"]["distance"] == pytest.approx(0.10)
        assert record_by_id["0"]["nearest_past_neuron_id"] == "7"
        assert record_by_id["0"]["nearest_past_label"] == "past_a"
        assert record_by_id["1"]["distance"] == pytest.approx(0.50)
        assert record_by_id["1"]["nearest_past_neuron_id"] == "9"
        assert record_by_id["1"]["nearest_past_label"] == "past_b"

        assert plugin.past_run_id_param == "parent_xyz"
        assert plugin.embedding_provider_param == "openai"
        assert plugin.embedding_model_param == "text-embedding-3-small"
        assert plugin.num_neurons_current_param == 3
        assert plugin.num_neurons_past_param == 2
        assert plugin.mean_distance_param == pytest.approx(0.30)
        assert plugin.median_distance_param == pytest.approx(0.30)

        passed_kwargs = mock_compute.call_args.kwargs
        assert passed_kwargs["current_label_texts"] == [
            "current_a",
            "current_b",
            "current_c",
        ]
        assert passed_kwargs["past_label_texts"] == ["past_a", "past_b"]
        assert passed_kwargs["embedding_provider"] == "openai"
        assert passed_kwargs["embedding_model"] == "text-embedding-3-small"

    @patch(
        "plugins.labeling_evaluation.compare.nearest_label_distance_bars."
        "nearest_label_distance_bars.compute_nearest_distances"
    )
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_figure_has_single_bar_trace_with_descending_y(
        self,
        mock_compare_loader_cls: MagicMock,
        mock_compute: MagicMock,
    ) -> None:
        """Verify the figure is one ``Bar`` trace whose y is sorted desc.

        Args:
            mock_compare_loader_cls: Patched ``MLflowRunLoader`` class.
            mock_compute: Patched compute helper.
        """
        past_context = {"neuron_labeling": {"run_id": "nl_past"}}
        mock_compare_loader_cls.return_value = _make_past_loader(
            context=past_context,
            past_neuron_labels={"7": "past_a", "9": "past_b"},
        )

        from plugins.labeling_evaluation._nearest_label_distance import (
            NearestLabelDistances,
        )

        mock_compute.return_value = NearestLabelDistances(
            distances=np.array([0.10, 0.50, 0.30]),
            nearest_past_indices=np.array([0, 1, 0]),
        )

        plugin = _build_plugin()
        plugin.run(past_run_id="parent_xyz")

        traces = plugin._fig.data
        assert len(traces) == 1
        ys = list(traces[0].y)
        assert ys == sorted(ys, reverse=True)

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_missing_past_run_id_kwarg_raises(
        self,
        mock_loader_cls: MagicMock,  # noqa: ARG002
    ) -> None:
        """Verify omitting past_run_id surfaces MissingContextError via the wrapper.

        Args:
            mock_loader_cls: Patched MLflowRunLoader (unused; just
                blocks any accidental real load).
        """
        from plugins.plugin_interface import MissingContextError

        plugin = _build_plugin()
        kwargs_without_past_run_id: dict[str, Any] = {
            "embedding_model": "m",
        }
        with pytest.raises(MissingContextError, match="past_run_id"):
            plugin.run(**kwargs_without_past_run_id)


class TestBarsHints:
    """Tests for the auto-generated UI hint set."""

    def test_past_runs_dropdown_hint_present(self) -> None:
        """Verify BaseComparePlugin injected a PastRunsDropdownHint for past_run_id."""
        from plugins.labeling_evaluation.compare.nearest_label_distance_bars.nearest_label_distance_bars import (  # noqa: E501
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

"""Unit tests for inspection.compare.sae_inspection.

Mocks ``MLflowRunLoader`` so the past-run side never touches MLflow.
The current-run side reads ``self.neuron_labels`` / ``self.items`` /
``self.item_acts`` directly, mirroring what ``load_context`` would
have populated.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from plugins.inspection.compare.sae_inspection.sae_inspection import Plugin


def _build_plugin() -> Plugin:
    """Build a Plugin instance with current-context attributes set.

    Returns:
        Plugin: A ready-to-run compare plugin instance with stubbed
            current-side data; the past side is reached via the
            mocked ``MLflowRunLoader``.
    """
    plugin = Plugin()
    plugin.neuron_labels = {"0": "current_a", "1": "current_b"}
    plugin.items = np.array(["c_item_0", "c_item_1", "c_item_2"])
    plugin.item_acts = torch.tensor(
        [
            [0.1, 0.7],
            [0.5, 0.2],
            [0.9, 0.4],
        ]
    )
    return plugin


def _make_past_loader(
    *,
    context: dict[str, Any],
    past_neuron_labels: dict[str, str],
    past_items: np.ndarray,
) -> MagicMock:
    """Build a MagicMock that mimics MLflowRunLoader for the past side.

    Args:
        context: Past parent run's ``context.json`` payload.
        past_neuron_labels: Stub ``neuron_labels.json`` for past run.
        past_items: Stub ``items.npy`` for past run.

    Returns:
        MagicMock: A loader whose ``get_*_artifact`` methods route
            by filename to the appropriate stub.  The ``pt`` artifact
            (``item_acts.pt``) is loaded via ``torch.load`` which the
            caller mocks separately.
    """

    def get_json(filename: str) -> Any:
        match filename:
            case "context.json":
                return context
            case "neuron_labels.json":
                return past_neuron_labels
            case other:
                raise AssertionError(f"unexpected json artifact requested: {other}")

    def get_npy(filename: str, allow_pickle: bool = False) -> np.ndarray:  # noqa: ARG001
        if filename == "items.npy":
            return past_items
        raise AssertionError(f"unexpected npy artifact: {filename}")

    loader = MagicMock()
    loader.get_json_artifact.side_effect = get_json
    loader.get_npy_artifact.side_effect = get_npy
    loader.download_artifact.return_value = "/tmp/item_acts.pt"
    return loader


class TestCompareSaeInspectionRun:
    """Tests for the compare plugin's run() method."""

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    @patch("plugins.plugin_interface.MLflowRunLoader")
    @patch("torch.load")
    def test_run_populates_both_sides(
        self,
        mock_torch_load: MagicMock,
        mock_base_loader_cls: MagicMock,
        mock_compare_loader_cls: MagicMock,
    ) -> None:
        """Verify both current and past outputs are populated correctly."""
        past_context = {
            "dataset_loading": {"run_id": "ds_past"},
            "neuron_labeling": {"run_id": "nl_past"},
        }
        past_loader = _make_past_loader(
            context=past_context,
            past_neuron_labels={"0": "past_a", "1": "past_b"},
            past_items=np.array(["p_item_0", "p_item_1", "p_item_2"]),
        )

        mock_compare_loader_cls.return_value = past_loader
        mock_base_loader_cls.return_value = past_loader
        mock_torch_load.return_value = torch.tensor(
            [
                [0.1, 0.4],
                [0.6, 0.3],
                [0.2, 0.9],
            ]
        )

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            neuron_id="0",
            past_neuron_id="1",
            k=2,
        )

        assert plugin.current_top_k_item_ids == ["c_item_2", "c_item_1"]
        assert plugin.current_top_k_activations == pytest.approx([0.9, 0.5])
        assert plugin.past_top_k_item_ids == ["p_item_2", "p_item_0"]
        assert plugin.past_top_k_activations == pytest.approx([0.9, 0.4])

        assert plugin.neuron_id_param == "0"
        assert plugin.label_param == "current_a"
        assert plugin.past_neuron_id_param == "1"
        assert plugin.past_label_param == "past_b"
        assert plugin.past_run_id_param == "parent_xyz"
        assert plugin.k_param == 2

        assert plugin.past_context == past_context

    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    def test_missing_past_run_id_kwarg_raises(
        self,
        mock_loader_cls: MagicMock,  # noqa: ARG002
    ) -> None:
        """Verify omitting past_run_id surfaces MissingContextError via the wrapper."""
        from plugins.plugin_interface import MissingContextError

        plugin = _build_plugin()
        kwargs_without_past_run_id: dict[str, Any] = {
            "neuron_id": "0",
            "past_neuron_id": "0",
            "k": 1,
        }
        with pytest.raises(MissingContextError, match="past_run_id"):
            plugin.run(**kwargs_without_past_run_id)


class TestCompareSaeInspectionFormatter:
    """Tests for the static _format_neuron_choices helper."""

    def test_formats_label_value_pairs(self) -> None:
        """Verify dropdown entries follow {label, value} contract."""
        result = Plugin._format_neuron_choices({"0": "concept_a", "5": "concept_b"})
        assert {"label": "concept_a [neuron id 0]", "value": "0"} in result
        assert {"label": "concept_b [neuron id 5]", "value": "5"} in result
        assert len(result) == 2

    def test_handles_empty_mapping(self) -> None:
        """Verify the formatter handles an empty input."""
        assert Plugin._format_neuron_choices({}) == []

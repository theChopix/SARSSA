"""Unit tests for steering.compare.sae_steering.

Mocks ``MLflowRunLoader`` and ``SteeredModel`` so neither MLflow nor
the real torch model layer is touched.  The current-side artifacts
are set on the plugin instance directly, mirroring what
``load_context`` would have populated.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix


def _make_csr(rows: list[list[int]], num_items: int) -> csr_matrix:
    """Build a binary csr_matrix from per-row column-index lists."""
    indptr = [0]
    indices: list[int] = []
    for row in rows:
        indices.extend(row)
        indptr.append(len(indices))
    data = [1] * len(indices)
    return csr_matrix((data, indices, indptr), shape=(len(rows), num_items))


def _make_recommender(top_indices: list[int]) -> MagicMock:
    """Build a model mock with deterministic ``recommend`` output."""
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    scores = torch.tensor([[0.0] * len(top_indices)])
    indices = torch.tensor([top_indices])
    model.recommend = MagicMock(return_value=(scores, indices))
    return model


def _build_plugin() -> Any:
    """Build a Plugin instance with current-side attributes preset.

    Returns:
        Plugin: Compare plugin instance with stubbed current-side
            data; the past side is reached via the patched
            ``MLflowRunLoader`` and ``SteeredModel``.
    """
    from plugins.steering.compare.sae_steering.sae_steering import Plugin

    plugin = Plugin()
    plugin.full_csr = _make_csr([[0, 1]], num_items=3)
    plugin.items = np.array(["c_a", "c_b", "c_c"])
    plugin.users = np.array(["c_user"])
    plugin.base_model = _make_recommender([2, 1])
    plugin.sae = MagicMock()
    plugin.neuron_labels = {"0": "current_a", "1": "current_b"}
    return plugin


def _make_past_loader(
    *,
    context: dict[str, Any],
    past_neuron_labels: dict[str, str],
    past_full_csr: csr_matrix,
    past_users: np.ndarray,
    past_items: np.ndarray,
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

    def get_npz(filename: str) -> Any:
        if filename == "full_csr.npz":
            return past_full_csr
        raise AssertionError(f"unexpected npz artifact: {filename}")

    def get_npy(filename: str, allow_pickle: bool = False) -> np.ndarray:  # noqa: ARG001
        match filename:
            case "users.npy":
                return past_users
            case "items.npy":
                return past_items
            case other:
                raise AssertionError(f"unexpected npy artifact: {other}")

    loader = MagicMock()
    loader.get_json_artifact.side_effect = get_json
    loader.get_npz_artifact.side_effect = get_npz
    loader.get_npy_artifact.side_effect = get_npy
    loader.download_artifact_dir.return_value = "/tmp/past_model"
    return loader


class TestCompareSaeSteeringRun:
    """Tests for the compare plugin's run() method."""

    @patch("plugins.steering._steer.SteeredModel")
    @patch("plugins.compare_plugin_interface.MLflowRunLoader")
    @patch("plugins.plugin_interface.MLflowRunLoader")
    @patch("utils.torch.models.model_loader.load_base_model")
    @patch("utils.torch.models.model_loader.load_sae_model")
    def test_run_populates_six_artifacts(
        self,
        mock_load_sae: MagicMock,
        mock_load_base: MagicMock,
        mock_base_loader_cls: MagicMock,
        mock_compare_loader_cls: MagicMock,
        mock_steered_cls: MagicMock,
    ) -> None:
        """Verify all six output attributes are set with the right item ids."""
        past_context = {
            "dataset_loading": {"run_id": "ds_past"},
            "training_cfm": {"run_id": "cfm_past"},
            "training_sae": {"run_id": "sae_past"},
            "neuron_labeling": {"run_id": "nl_past"},
        }
        past_loader = _make_past_loader(
            context=past_context,
            past_neuron_labels={"0": "past_a", "1": "past_b"},
            past_full_csr=_make_csr([[1]], num_items=3),
            past_users=np.array(["p_user"]),
            past_items=np.array(["p_a", "p_b", "p_c"]),
        )
        mock_compare_loader_cls.return_value = past_loader
        mock_base_loader_cls.return_value = past_loader

        past_base_model = _make_recommender([0, 2])
        mock_load_base.return_value = past_base_model
        mock_load_sae.return_value = MagicMock()

        current_steered = _make_recommender([1, 0])
        past_steered = _make_recommender([2, 1])
        mock_steered_cls.side_effect = [current_steered, past_steered]

        plugin = _build_plugin()
        plugin.run(
            past_run_id="parent_xyz",
            user_id=0,
            neuron_id="0",
            past_neuron_id="1",
            alpha=0.4,
            k=2,
        )

        assert plugin.current_interacted_items == ["c_a", "c_b"]
        assert plugin.past_interacted_items == ["p_b"]
        assert plugin.current_original_recommendations == ["c_c", "c_b"]
        assert plugin.past_original_recommendations == ["p_a", "p_c"]
        assert plugin.current_steered_recommendations == ["c_b", "c_a"]
        assert plugin.past_steered_recommendations == ["p_c", "p_b"]

        assert plugin.user_id_param == 0
        assert plugin.user_original_id_param == "c_user"
        assert plugin.past_user_original_id_param == "p_user"
        assert plugin.neuron_id_param == "0"
        assert plugin.label_param == "current_a"
        assert plugin.past_neuron_id_param == "1"
        assert plugin.past_label_param == "past_b"
        assert plugin.past_run_id_param == "parent_xyz"
        assert plugin.alpha_param == 0.4
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
            "user_id": 0,
            "neuron_id": "0",
            "past_neuron_id": "0",
            "alpha": 0.1,
            "k": 1,
        }
        with pytest.raises(MissingContextError, match="past_run_id"):
            plugin.run(**kwargs_without_past_run_id)


class TestCompareSaeSteeringFormatter:
    """Tests for the static _format_neuron_choices helper."""

    def test_formats_label_value_pairs(self) -> None:
        """Verify dropdown entries follow {label, value} contract."""
        from plugins.steering.compare.sae_steering.sae_steering import Plugin

        result = Plugin._format_neuron_choices({"0": "concept_a", "5": "concept_b"})
        assert {"label": "concept_a [neuron id 0]", "value": "0"} in result
        assert {"label": "concept_b [neuron id 5]", "value": "5"} in result
        assert len(result) == 2

    def test_handles_empty_mapping(self) -> None:
        """Verify the formatter handles an empty input."""
        from plugins.steering.compare.sae_steering.sae_steering import Plugin

        assert Plugin._format_neuron_choices({}) == []


class TestCompareSaeSteeringHints:
    """Tests for the auto-generated UI hint set."""

    def test_past_runs_dropdown_hint_present(self) -> None:
        """Verify BaseComparePlugin injected a PastRunsDropdownHint for past_run_id."""
        from plugins.plugin_interface import PastRunsDropdownHint
        from plugins.steering.compare.sae_steering.sae_steering import Plugin

        hints = Plugin.io_spec.param_ui_hints
        past_hints = [
            h
            for h in hints
            if isinstance(h, PastRunsDropdownHint) and h.param_name == "past_run_id"
        ]
        assert len(past_hints) == 1
        assert past_hints[0].required_steps == [
            "dataset_loading",
            "training_cfm",
            "training_sae",
            "neuron_labeling",
        ]

    def test_past_neuron_id_is_cascading_dropdown(self) -> None:
        """Verify past_neuron_id's dropdown hint sources from past_run_id."""
        from plugins.plugin_interface import DynamicDropdownHint
        from plugins.steering.compare.sae_steering.sae_steering import Plugin

        hints = Plugin.io_spec.param_ui_hints
        past_neuron_hints = [
            h
            for h in hints
            if isinstance(h, DynamicDropdownHint) and h.param_name == "past_neuron_id"
        ]
        assert len(past_neuron_hints) == 1
        assert past_neuron_hints[0].source_run_param == "past_run_id"

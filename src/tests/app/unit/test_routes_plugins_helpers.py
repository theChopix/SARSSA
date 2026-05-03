"""Unit tests for the helper functions in app.api.routes_plugins.

Covers the pure helpers that back the param-choices endpoint —
``_find_dropdown_hint``, ``_load_hint_artifact``, and
``_resolve_artifact_run_id`` — using mocked plugin instances and
MLflow loaders.  Endpoint-level tests live in
``tests/app/integration/test_routes_plugins.py``.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.routes_plugins import (
    _find_dropdown_hint,
    _load_hint_artifact,
    _resolve_artifact_run_id,
)
from plugins.plugin_interface import (
    DynamicDropdownHint,
    ParamUIHint,
    PluginIOSpec,
)

# ── Helpers ───────────────────────────────────────────────────────


def _make_mock_plugin_with_hints(
    hints: list[ParamUIHint],
    formatter_name: str = "_fmt",
    formatter_fn: Any = None,
) -> MagicMock:
    """Create a mock plugin with param_ui_hints and an optional formatter.

    Args:
        hints: List of ParamUIHint instances for io_spec.
        formatter_name: Name of the formatter static method.
        formatter_fn: Callable to use as the formatter. If None,
            no formatter attribute is set on the class.

    Returns:
        MagicMock: A mock plugin instance.
    """
    plugin = MagicMock()
    plugin.io_spec = PluginIOSpec(param_ui_hints=hints)
    if formatter_fn is not None:
        setattr(plugin.__class__, formatter_name, formatter_fn)
    else:
        # Ensure getattr returns None for missing formatter
        if hasattr(plugin.__class__, formatter_name):
            delattr(plugin.__class__, formatter_name)
    return plugin


# ── Tests for _find_dropdown_hint ─────────────────────────────────


class TestFindDropdownHint:
    """Tests for _find_dropdown_hint."""

    def test_returns_matching_hint(self) -> None:
        """Verify the matching DynamicDropdownHint is returned."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_fmt",
        )
        plugin = _make_mock_plugin_with_hints([hint])
        result = _find_dropdown_hint(plugin, "neuron_id")
        assert result is hint

    def test_returns_none_when_no_match(self) -> None:
        """Verify None when param_name does not match any hint."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            formatter="_fmt",
        )
        plugin = _make_mock_plugin_with_hints([hint])
        assert _find_dropdown_hint(plugin, "other_param") is None

    def test_returns_none_when_no_hints(self) -> None:
        """Verify None when plugin has no hints at all."""
        plugin = _make_mock_plugin_with_hints([])
        assert _find_dropdown_hint(plugin, "neuron_id") is None

    def test_ignores_base_param_ui_hint(self) -> None:
        """Verify base ParamUIHint is not returned as a dropdown."""
        hint = ParamUIHint(param_name="neuron_id")
        plugin = _make_mock_plugin_with_hints([hint])
        assert _find_dropdown_hint(plugin, "neuron_id") is None

    def test_returns_first_matching_hint(self) -> None:
        """Verify the first matching hint wins when duplicates exist."""
        hint_a = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_file="a.json",
            formatter="_fmt_a",
        )
        hint_b = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_file="b.json",
            formatter="_fmt_b",
        )
        plugin = _make_mock_plugin_with_hints([hint_a, hint_b])
        result = _find_dropdown_hint(plugin, "neuron_id")
        assert result is hint_a


# ── Tests for _load_hint_artifact ─────────────────────────────────


class TestLoadHintArtifact:
    """Tests for _load_hint_artifact."""

    @patch("app.api.routes_plugins.MLflowRunLoader")
    def test_loads_json_artifact(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify JSON loader is called for artifact_loader='json'."""
        mock_loader = MagicMock()
        mock_loader.get_json_artifact.return_value = {"0": "sci-fi"}
        mock_loader_cls.return_value = mock_loader

        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_file="neuron_labels.json",
            artifact_loader="json",
            formatter="_fmt",
        )
        result = _load_hint_artifact(hint, "run123")

        mock_loader.get_json_artifact.assert_called_once_with(
            "neuron_labels.json",
        )
        assert result == {"0": "sci-fi"}

    @patch("app.api.routes_plugins.MLflowRunLoader")
    def test_loads_npy_artifact(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify NPY loader is called for artifact_loader='npy'."""
        mock_loader = MagicMock()
        mock_loader.get_npy_artifact.return_value = [1, 2, 3]
        mock_loader_cls.return_value = mock_loader

        hint = DynamicDropdownHint(
            param_name="x",
            artifact_file="data.npy",
            artifact_loader="npy",
            formatter="_fmt",
        )
        result = _load_hint_artifact(hint, "run123")

        mock_loader.get_npy_artifact.assert_called_once_with("data.npy")
        assert result == [1, 2, 3]

    @patch("app.api.routes_plugins.MLflowRunLoader")
    def test_loads_npz_artifact(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify NPZ loader is called for artifact_loader='npz'."""
        mock_loader = MagicMock()
        mock_loader.get_npz_artifact.return_value = "sparse_matrix"
        mock_loader_cls.return_value = mock_loader

        hint = DynamicDropdownHint(
            param_name="x",
            artifact_file="matrix.npz",
            artifact_loader="npz",
            formatter="_fmt",
        )
        result = _load_hint_artifact(hint, "run123")

        mock_loader.get_npz_artifact.assert_called_once_with("matrix.npz")
        assert result == "sparse_matrix"

    @patch("app.api.routes_plugins.MLflowRunLoader")
    def test_unsupported_loader_raises_404(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify unsupported artifact_loader raises HTTPException."""
        mock_loader_cls.return_value = MagicMock()

        hint = DynamicDropdownHint(
            param_name="x",
            artifact_file="data.csv",
            artifact_loader="csv",
            formatter="_fmt",
        )
        with pytest.raises(HTTPException) as exc_info:
            _load_hint_artifact(hint, "run123")
        assert exc_info.value.status_code == 404
        assert "Unsupported artifact loader" in exc_info.value.detail

    @patch("app.api.routes_plugins.MLflowRunLoader")
    def test_mlflow_error_raises_404(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify MLflow loading failures raise HTTPException 404."""
        mock_loader = MagicMock()
        mock_loader.get_json_artifact.side_effect = FileNotFoundError(
            "not found",
        )
        mock_loader_cls.return_value = mock_loader

        hint = DynamicDropdownHint(
            param_name="x",
            artifact_file="missing.json",
            artifact_loader="json",
            formatter="_fmt",
        )
        with pytest.raises(HTTPException) as exc_info:
            _load_hint_artifact(hint, "run123")
        assert exc_info.value.status_code == 404
        assert "missing.json" in exc_info.value.detail


# ── Tests for _resolve_artifact_run_id ────────────────────────────


class TestResolveArtifactRunId:
    """Tests for _resolve_artifact_run_id."""

    def test_returns_run_id_unchanged_when_not_cascading(self) -> None:
        """Verify non-cascading hints pass run_id through untouched."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_fmt",
        )
        assert _resolve_artifact_run_id(hint, "step_run_xyz") == "step_run_xyz"

    @patch("app.api.routes_plugins.get_run_context")
    def test_resolves_from_parent_context_when_cascading(
        self,
        mock_get_context: MagicMock,
    ) -> None:
        """Verify cascading hints resolve via the parent run's context."""
        mock_get_context.return_value = {
            "dataset_loading": {"run_id": "ds_step"},
            "neuron_labeling": {"run_id": "nl_step"},
        }
        hint = DynamicDropdownHint(
            param_name="past_neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_fmt",
            source_run_param="past_run_id",
        )
        result = _resolve_artifact_run_id(hint, "parent_run_abc")
        assert result == "nl_step"
        mock_get_context.assert_called_once_with("parent_run_abc")

    @patch("app.api.routes_plugins.get_run_context")
    def test_raises_404_when_parent_context_missing(
        self,
        mock_get_context: MagicMock,
    ) -> None:
        """Verify a missing context.json bubbles as a 404."""
        mock_get_context.side_effect = FileNotFoundError("no context")
        hint = DynamicDropdownHint(
            param_name="past_neuron_id",
            artifact_step="neuron_labeling",
            formatter="_fmt",
            source_run_param="past_run_id",
        )
        with pytest.raises(HTTPException) as exc_info:
            _resolve_artifact_run_id(hint, "parent_run_abc")
        assert exc_info.value.status_code == 404
        assert "context.json" in exc_info.value.detail

    @patch("app.api.routes_plugins.get_run_context")
    def test_raises_404_when_step_missing_from_context(
        self,
        mock_get_context: MagicMock,
    ) -> None:
        """Verify a context without the required step yields 404."""
        mock_get_context.return_value = {"dataset_loading": {"run_id": "ds"}}
        hint = DynamicDropdownHint(
            param_name="past_neuron_id",
            artifact_step="neuron_labeling",
            formatter="_fmt",
            source_run_param="past_run_id",
        )
        with pytest.raises(HTTPException) as exc_info:
            _resolve_artifact_run_id(hint, "parent_run_abc")
        assert exc_info.value.status_code == 404
        assert "neuron_labeling" in exc_info.value.detail

    @patch("app.api.routes_plugins.get_run_context")
    def test_raises_404_when_step_entry_has_no_run_id(
        self,
        mock_get_context: MagicMock,
    ) -> None:
        """Verify a malformed step entry without run_id yields 404."""
        mock_get_context.return_value = {"neuron_labeling": {"foo": "bar"}}
        hint = DynamicDropdownHint(
            param_name="past_neuron_id",
            artifact_step="neuron_labeling",
            formatter="_fmt",
            source_run_param="past_run_id",
        )
        with pytest.raises(HTTPException) as exc_info:
            _resolve_artifact_run_id(hint, "parent_run_abc")
        assert exc_info.value.status_code == 404

"""Unit tests for app.api.routes_plugins.

Tests cover the param-choices endpoint and its helper functions
using mocked plugin instances and MLflow loaders.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.routes_plugins import _find_dropdown_hint, _load_hint_artifact
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


# ── Tests for GET /plugins/param-choices endpoint ─────────────────


class TestGetParamChoicesEndpoint:
    """Tests for the param-choices endpoint via TestClient."""

    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_returns_formatted_choices(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify the endpoint returns formatted dropdown options."""
        from fastapi.testclient import TestClient

        from app.main import app

        def _fmt(data: dict) -> list[dict[str, str]]:
            return [{"label": f"{v} [id {k}]", "value": k} for k, v in data.items()]

        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            artifact_loader="json",
            formatter="_fmt",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        plugin.__class__._fmt = staticmethod(_fmt)
        mock_pm.load.return_value = plugin

        mock_loader = MagicMock()
        mock_loader.get_json_artifact.return_value = {
            "0": "sci-fi",
            "42": "comedy",
        }
        mock_loader_cls.return_value = mock_loader

        client = TestClient(app)
        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/neuron_id",
            params={"run_id": "abc123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert {"label": "sci-fi [id 0]", "value": "0"} in data
        assert {"label": "comedy [id 42]", "value": "42"} in data

    @patch("app.api.routes_plugins.PluginManager")
    def test_returns_404_for_unknown_plugin(
        self,
        mock_pm: MagicMock,
    ) -> None:
        """Verify 404 when the plugin cannot be loaded."""
        from fastapi.testclient import TestClient

        from app.main import app

        mock_pm.load.side_effect = ImportError("no such module")

        client = TestClient(app)
        response = client.get(
            "/plugins/param-choices/steering/steering.fake.fake/neuron_id",
            params={"run_id": "abc123"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch("app.api.routes_plugins.PluginManager")
    def test_returns_404_for_missing_hint(
        self,
        mock_pm: MagicMock,
    ) -> None:
        """Verify 404 when no DynamicDropdownHint exists for param."""
        from fastapi.testclient import TestClient

        from app.main import app

        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[])
        mock_pm.load.return_value = plugin

        client = TestClient(app)
        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/neuron_id",
            params={"run_id": "abc123"},
        )

        assert response.status_code == 404
        assert "No DynamicDropdownHint" in response.json()["detail"]

    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_returns_404_for_missing_formatter(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify 404 when the formatter method doesn't exist."""
        from fastapi.testclient import TestClient

        from app.main import app

        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_file="neuron_labels.json",
            artifact_loader="json",
            formatter="_nonexistent_formatter",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        # Ensure the formatter attribute does not exist
        plugin.__class__ = type(
            "FakePlugin",
            (),
            {"io_spec": plugin.io_spec},
        )
        mock_pm.load.return_value = plugin

        mock_loader = MagicMock()
        mock_loader.get_json_artifact.return_value = {}
        mock_loader_cls.return_value = mock_loader

        client = TestClient(app)
        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/neuron_id",
            params={"run_id": "abc123"},
        )

        assert response.status_code == 404
        assert "Formatter" in response.json()["detail"]

    def test_returns_422_when_run_id_missing(self) -> None:
        """Verify 422 when run_id query param is not provided."""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/neuron_id",
        )

        assert response.status_code == 422

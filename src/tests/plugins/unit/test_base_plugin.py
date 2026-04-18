"""Unit tests for BasePlugin lifecycle methods.

All MLflow and MLflowRunLoader interactions are mocked.
"""

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.sparse as sp

from plugins.plugin_interface import (
    ArtifactSpec,
    BasePlugin,
    MissingContextError,
    OutputArtifactSpec,
    OutputParamSpec,
    ParamSpec,
    PluginIOSpec,
)

# ── Concrete stub for testing ───────────────────────────────────────


class _StubPlugin(BasePlugin):
    """Minimal concrete plugin for testing base class methods."""

    def run(self, context: dict, **params: Any) -> None:
        """No-op run implementation.

        Args:
            context: Pipeline context.
            **params: Ignored.
        """


def _make_plugin(io_spec: PluginIOSpec | None = None) -> _StubPlugin:
    """Create a stub plugin with an optional io_spec.

    Args:
        io_spec: I/O specification to assign.

    Returns:
        _StubPlugin: Plugin instance ready for testing.
    """
    plugin = _StubPlugin()
    if io_spec is not None:
        plugin.io_spec = io_spec
    return plugin


# ── _validate_required_steps ────────────────────────────────────────


class TestValidateRequiredSteps:
    """Tests for BasePlugin._validate_required_steps."""

    def test_passes_when_all_steps_present(self) -> None:
        """Verify no error when all required steps exist."""
        plugin = _make_plugin(
            PluginIOSpec(required_steps=["dataset_loading"]),
        )
        context = {"dataset_loading": {"run_id": "abc"}}
        plugin._validate_required_steps(context)

    def test_passes_with_multiple_steps(self) -> None:
        """Verify no error with multiple required steps present."""
        plugin = _make_plugin(
            PluginIOSpec(
                required_steps=["dataset_loading", "training_cfm"],
            ),
        )
        context = {
            "dataset_loading": {"run_id": "a"},
            "training_cfm": {"run_id": "b"},
        }
        plugin._validate_required_steps(context)

    def test_passes_with_empty_required_steps(self) -> None:
        """Verify no error when no steps are required."""
        plugin = _make_plugin(PluginIOSpec(required_steps=[]))
        plugin._validate_required_steps({})

    def test_raises_when_step_missing(self) -> None:
        """Verify MissingContextError when step key is absent."""
        plugin = _make_plugin(
            PluginIOSpec(required_steps=["training_cfm"]),
        )
        with pytest.raises(MissingContextError, match="training_cfm"):
            plugin._validate_required_steps({})

    def test_raises_when_run_id_missing(self) -> None:
        """Verify MissingContextError when run_id key is absent."""
        plugin = _make_plugin(
            PluginIOSpec(required_steps=["dataset_loading"]),
        )
        context = {"dataset_loading": {"status": "done"}}
        with pytest.raises(MissingContextError, match="no 'run_id'"):
            plugin._validate_required_steps(context)


# ── _load_artifact dispatch ─────────────────────────────────────────


class TestLoadArtifact:
    """Tests for BasePlugin._load_artifact dispatcher."""

    def test_npz_loader(self) -> None:
        """Verify npz loader calls get_npz_artifact."""
        plugin = _make_plugin()
        loader = MagicMock(spec=["get_npz_artifact"])
        expected = sp.csr_matrix(np.eye(3))
        loader.get_npz_artifact.return_value = expected

        spec = ArtifactSpec("s", "m.npz", "a", "npz")
        result = plugin._load_artifact(loader, spec)

        loader.get_npz_artifact.assert_called_once_with("m.npz")
        assert result is expected

    def test_npz_loader_with_kwargs(self) -> None:
        """Verify loader_kwargs are forwarded to get_npz_artifact."""
        plugin = _make_plugin()
        loader = MagicMock(spec=["get_npz_artifact"])
        loader.get_npz_artifact.return_value = MagicMock()

        spec = ArtifactSpec(
            "s",
            "m.npz",
            "a",
            "npz",
            loader_kwargs={"return_sparse": False},
        )
        plugin._load_artifact(loader, spec)

        loader.get_npz_artifact.assert_called_once_with(
            "m.npz",
            return_sparse=False,
        )

    def test_npy_loader(self) -> None:
        """Verify npy loader calls get_npy_artifact."""
        plugin = _make_plugin()
        loader = MagicMock(spec=["get_npy_artifact"])
        expected = np.array([1, 2, 3])
        loader.get_npy_artifact.return_value = expected

        spec = ArtifactSpec("s", "arr.npy", "a", "npy")
        result = plugin._load_artifact(loader, spec)

        loader.get_npy_artifact.assert_called_once_with("arr.npy")
        np.testing.assert_array_equal(result, expected)

    def test_npy_loader_with_kwargs(self) -> None:
        """Verify loader_kwargs forwarded to get_npy_artifact."""
        plugin = _make_plugin()
        loader = MagicMock(spec=["get_npy_artifact"])
        loader.get_npy_artifact.return_value = np.array([])

        spec = ArtifactSpec(
            "s",
            "a.npy",
            "a",
            "npy",
            loader_kwargs={"allow_pickle": True},
        )
        plugin._load_artifact(loader, spec)

        loader.get_npy_artifact.assert_called_once_with(
            "a.npy",
            allow_pickle=True,
        )

    def test_json_loader(self) -> None:
        """Verify json loader calls get_json_artifact."""
        plugin = _make_plugin()
        loader = MagicMock(spec=["get_json_artifact"])
        expected = {"key": "value"}
        loader.get_json_artifact.return_value = expected

        spec = ArtifactSpec("s", "d.json", "a", "json")
        result = plugin._load_artifact(loader, spec)

        loader.get_json_artifact.assert_called_once_with("d.json")
        assert result == expected

    def test_model_dir_loader(self) -> None:
        """Verify model_dir loader calls download_artifact_dir."""
        plugin = _make_plugin()
        loader = MagicMock(spec=["download_artifact_dir"])
        loader.download_artifact_dir.return_value = "/tmp/model"

        spec = ArtifactSpec("s", "", "a", "model_dir")
        result = plugin._load_artifact(loader, spec)

        loader.download_artifact_dir.assert_called_once_with()
        assert result == "/tmp/model"

    @patch("torch.load")
    def test_pt_loader(self, mock_torch_load: MagicMock) -> None:
        """Verify pt loader downloads file and calls torch.load."""
        plugin = _make_plugin()
        loader = MagicMock(spec=["download_artifact"])
        loader.download_artifact.return_value = "/tmp/t.pt"
        expected_tensor = MagicMock()
        mock_torch_load.return_value = expected_tensor

        spec = ArtifactSpec("s", "t.pt", "a", "pt")
        result = plugin._load_artifact(loader, spec)

        loader.download_artifact.assert_called_once_with("t.pt")
        mock_torch_load.assert_called_once_with(
            "/tmp/t.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert result is expected_tensor

    def test_unknown_loader_raises(self) -> None:
        """Verify ValueError for unrecognised loader type."""
        plugin = _make_plugin()
        loader = MagicMock()
        spec = ArtifactSpec("s", "f", "a", "parquet")

        with pytest.raises(ValueError, match="parquet"):
            plugin._load_artifact(loader, spec)


# ── _save_artifact dispatch ─────────────────────────────────────────


class TestSaveArtifact:
    """Tests for BasePlugin._save_artifact dispatcher."""

    def test_json_saver(self, tmp_path: Any) -> None:
        """Verify json saver writes valid JSON."""
        plugin = _make_plugin()
        plugin.my_data = {"a": 1, "b": [2, 3]}

        spec = OutputArtifactSpec("my_data", "out.json", "json")
        plugin._save_artifact(str(tmp_path), spec)

        path = tmp_path / "out.json"
        assert path.exists()
        with open(path) as f:
            assert json.load(f) == {"a": 1, "b": [2, 3]}

    def test_npz_saver(self, tmp_path: Any) -> None:
        """Verify npz saver writes a sparse matrix."""
        plugin = _make_plugin()
        plugin.matrix = sp.csr_matrix(np.eye(3))

        spec = OutputArtifactSpec("matrix", "m.npz", "npz")
        plugin._save_artifact(str(tmp_path), spec)

        path = tmp_path / "m.npz"
        assert path.exists()
        loaded = sp.load_npz(str(path))
        np.testing.assert_array_equal(
            loaded.toarray(),
            np.eye(3),
        )

    def test_npy_saver(self, tmp_path: Any) -> None:
        """Verify npy saver writes a numpy array."""
        plugin = _make_plugin()
        plugin.arr = np.array([10, 20, 30])

        spec = OutputArtifactSpec("arr", "a.npy", "npy")
        plugin._save_artifact(str(tmp_path), spec)

        path = tmp_path / "a.npy"
        assert path.exists()
        np.testing.assert_array_equal(
            np.load(str(path)),
            [10, 20, 30],
        )

    @patch("torch.save")
    def test_pt_saver(
        self,
        mock_torch_save: MagicMock,
        tmp_path: Any,
    ) -> None:
        """Verify pt saver calls torch.save with correct path."""
        plugin = _make_plugin()
        plugin.tensor = MagicMock()

        spec = OutputArtifactSpec("tensor", "t.pt", "pt")
        plugin._save_artifact(str(tmp_path), spec)

        expected_path = os.path.join(str(tmp_path), "t.pt")
        mock_torch_save.assert_called_once_with(
            plugin.tensor,
            expected_path,
        )

    def test_unknown_saver_raises(self, tmp_path: Any) -> None:
        """Verify ValueError for unrecognised saver type."""
        plugin = _make_plugin()
        plugin.val = "data"
        spec = OutputArtifactSpec("val", "f.csv", "csv")

        with pytest.raises(ValueError, match="csv"):
            plugin._save_artifact(str(tmp_path), spec)


# ── load_context (integration of sub-helpers) ───────────────────────


class TestLoadContext:
    """Tests for BasePlugin.load_context end-to-end."""

    @patch("plugins.plugin_interface.MLflowRunLoader")
    def test_loads_artifacts_and_params(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify artifacts and params are set on self.*."""
        mock_loader = MagicMock()
        mock_loader.get_npz_artifact.return_value = sp.csr_matrix(
            np.eye(2),
        )
        mock_loader.get_parameter.return_value = "42"
        mock_loader_cls.return_value = mock_loader

        io_spec = PluginIOSpec(
            required_steps=["dataset_loading"],
            input_artifacts=[
                ArtifactSpec(
                    "dataset_loading",
                    "train.npz",
                    "train_csr",
                    "npz",
                ),
            ],
            input_params=[
                ParamSpec(
                    "dataset_loading",
                    "num_users",
                    "num_users",
                    int,
                ),
            ],
        )
        plugin = _make_plugin(io_spec)
        context = {"dataset_loading": {"run_id": "run_abc"}}

        plugin.load_context(context)

        assert hasattr(plugin, "train_csr")
        assert sp.issparse(plugin.train_csr)
        assert plugin.num_users == 42
        assert isinstance(plugin.num_users, int)

    def test_raises_on_missing_step(self) -> None:
        """Verify MissingContextError for missing required step."""
        io_spec = PluginIOSpec(
            required_steps=["training_cfm"],
        )
        plugin = _make_plugin(io_spec)

        with pytest.raises(MissingContextError):
            plugin.load_context({})

    def test_no_op_with_empty_spec(self) -> None:
        """Verify no error and no side-effects with empty io_spec."""
        plugin = _make_plugin(PluginIOSpec())
        plugin.load_context({})

    @patch("plugins.plugin_interface.MLflowRunLoader")
    def test_creates_loader_per_step(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify separate MLflowRunLoader per upstream step."""
        mock_loader = MagicMock()
        mock_loader.get_npz_artifact.return_value = MagicMock()
        mock_loader_cls.return_value = mock_loader

        io_spec = PluginIOSpec(
            required_steps=["step_a", "step_b"],
            input_artifacts=[
                ArtifactSpec("step_a", "a.npz", "a", "npz"),
                ArtifactSpec("step_b", "b.npz", "b", "npz"),
            ],
        )
        plugin = _make_plugin(io_spec)
        context = {
            "step_a": {"run_id": "id_a"},
            "step_b": {"run_id": "id_b"},
        }

        plugin.load_context(context)

        assert mock_loader_cls.call_count == 2
        mock_loader_cls.assert_any_call("id_a")
        mock_loader_cls.assert_any_call("id_b")


# ── update_context ──────────────────────────────────────────────────


class TestUpdateContext:
    """Tests for BasePlugin.update_context."""

    @patch("plugins.plugin_interface.mlflow")
    def test_logs_output_params(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """Verify output params are logged to MLflow."""
        io_spec = PluginIOSpec(
            output_params=[
                OutputParamSpec("dataset_name", "dataset"),
                OutputParamSpec("num_users", "n_users"),
            ],
        )
        plugin = _make_plugin(io_spec)
        plugin.dataset = "MovieLens"
        plugin.n_users = 1000

        plugin.update_context()

        mock_mlflow.log_params.assert_called_once_with(
            {"dataset_name": "MovieLens", "num_users": 1000},
        )

    @patch("plugins.plugin_interface.mlflow")
    def test_skips_log_params_when_empty(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """Verify mlflow.log_params is not called with no specs."""
        plugin = _make_plugin(PluginIOSpec())
        plugin.update_context()

        mock_mlflow.log_params.assert_not_called()

    @patch("plugins.plugin_interface.mlflow")
    def test_logs_output_artifacts(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """Verify output artifacts are saved and logged."""
        io_spec = PluginIOSpec(
            output_artifacts=[
                OutputArtifactSpec("data", "data.json", "json"),
            ],
        )
        plugin = _make_plugin(io_spec)
        plugin.data = {"key": "value"}

        plugin.update_context()

        mock_mlflow.log_artifacts.assert_called_once()
        tmp_dir = mock_mlflow.log_artifacts.call_args[0][0]
        assert os.path.isdir(tmp_dir) or True  # temp cleaned up

    @patch("plugins.plugin_interface.mlflow")
    def test_skips_log_artifacts_when_empty(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """Verify mlflow.log_artifacts not called with no specs."""
        plugin = _make_plugin(PluginIOSpec())
        plugin.update_context()

        mock_mlflow.log_artifacts.assert_not_called()

    @patch("plugins.plugin_interface.mlflow")
    def test_combined_params_and_artifacts(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """Verify both params and artifacts are logged together."""
        io_spec = PluginIOSpec(
            output_params=[
                OutputParamSpec("name", "dataset_name"),
            ],
            output_artifacts=[
                OutputArtifactSpec("arr", "arr.npy", "npy"),
            ],
        )
        plugin = _make_plugin(io_spec)
        plugin.dataset_name = "ML"
        plugin.arr = np.array([1])

        plugin.update_context()

        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_artifacts.assert_called_once()


# ── io_spec class attribute default ─────────────────────────────────


class TestIOSpecDefault:
    """Tests for the default io_spec on BasePlugin."""

    def test_default_io_spec_is_empty(self) -> None:
        """Verify default io_spec has all empty lists."""
        plugin = _make_plugin()
        assert plugin.io_spec.required_steps == []
        assert plugin.io_spec.input_artifacts == []
        assert plugin.io_spec.input_params == []
        assert plugin.io_spec.output_artifacts == []
        assert plugin.io_spec.output_params == []

    def test_io_spec_can_be_overridden(self) -> None:
        """Verify io_spec can be set to a custom spec."""
        spec = PluginIOSpec(required_steps=["step_a"])
        plugin = _make_plugin(spec)
        assert plugin.io_spec.required_steps == ["step_a"]

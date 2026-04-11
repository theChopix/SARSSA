"""Unit tests for utils.mlflow_manager.

All MLflow interactions are mocked to keep tests fast and isolated.
"""

from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import scipy.sparse as sp

from utils.mlflow_manager import MLflowRunLoader


def _make_loader(run_id: str = "test_run_123") -> MLflowRunLoader:
    """Create an MLflowRunLoader with a given run ID.

    Args:
        run_id: MLflow run ID.

    Returns:
        MLflowRunLoader: Loader instance (no MLflow calls made yet).
    """
    return MLflowRunLoader(run_id)


def _make_mock_run(
    params: dict[str, str] | None = None,
    metrics: dict[str, float] | None = None,
) -> MagicMock:
    """Create a mock MLflow Run object.

    Args:
        params: Run parameters.
        metrics: Run metrics.

    Returns:
        MagicMock: A mock with ``data.params`` and ``data.metrics``.
    """
    mock_run = MagicMock()
    mock_run.data.params = params or {}
    mock_run.data.metrics = metrics or {}
    return mock_run


# ── Run property ──────────────────────────────────────────────────────


class TestRunProperty:
    """Tests for the lazy-loaded ``run`` property."""

    @patch("utils.mlflow_manager.mlflow")
    def test_caches_run_object(self, mock_mlflow: MagicMock) -> None:
        """Verify mlflow.get_run is called only once on repeated access."""
        mock_mlflow.get_run.return_value = _make_mock_run()
        loader = _make_loader()

        _ = loader.run
        _ = loader.run

        mock_mlflow.get_run.assert_called_once_with("test_run_123")


# ── download_artifact ─────────────────────────────────────────────────


class TestDownloadArtifact:
    """Tests for download_artifact (single file)."""

    @patch("utils.mlflow_manager.mlflow")
    def test_downloads_with_filename_only(self, mock_mlflow: MagicMock) -> None:
        """Verify correct artifact_path when no subdirectory is given."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/data.json"
        loader = _make_loader("run_abc")

        result = loader.download_artifact("data.json")

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_abc", artifact_path="data.json"
        )
        assert result == "/tmp/data.json"

    @patch("utils.mlflow_manager.mlflow")
    def test_downloads_with_artifact_path(self, mock_mlflow: MagicMock) -> None:
        """Verify subdirectory is prepended to filename."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/model/config.json"
        loader = _make_loader("run_abc")

        result = loader.download_artifact("config.json", artifact_path="model")

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_abc", artifact_path="model/config.json"
        )
        assert result == "/tmp/model/config.json"


# ── download_artifact_dir ─────────────────────────────────────────────


class TestDownloadArtifactDir:
    """Tests for download_artifact_dir (directory)."""

    @patch("utils.mlflow_manager.mlflow")
    def test_downloads_entire_artifact_dir(self, mock_mlflow: MagicMock) -> None:
        """Verify empty string is used when no subdirectory is given."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/artifacts"
        loader = _make_loader("run_abc")

        result = loader.download_artifact_dir()

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_abc", artifact_path=""
        )
        assert result == "/tmp/artifacts"

    @patch("utils.mlflow_manager.mlflow")
    def test_downloads_subdirectory(self, mock_mlflow: MagicMock) -> None:
        """Verify specific subdirectory is passed through."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/model"
        loader = _make_loader("run_abc")

        result = loader.download_artifact_dir(artifact_path="model")

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_abc", artifact_path="model"
        )
        assert result == "/tmp/model"


# ── Parameters & Metrics ──────────────────────────────────────────────


class TestParameters:
    """Tests for get_parameters and get_parameter."""

    @patch("utils.mlflow_manager.mlflow")
    def test_get_parameters_returns_all(self, mock_mlflow: MagicMock) -> None:
        """Verify all parameters are returned."""
        mock_mlflow.get_run.return_value = _make_mock_run(params={"lr": "0.01", "epochs": "100"})
        loader = _make_loader()

        assert loader.get_parameters() == {"lr": "0.01", "epochs": "100"}

    @patch("utils.mlflow_manager.mlflow")
    def test_get_parameter_existing(self, mock_mlflow: MagicMock) -> None:
        """Verify a specific parameter is returned."""
        mock_mlflow.get_run.return_value = _make_mock_run(params={"lr": "0.01"})
        loader = _make_loader()

        assert loader.get_parameter("lr") == "0.01"

    @patch("utils.mlflow_manager.mlflow")
    def test_get_parameter_missing_returns_default(self, mock_mlflow: MagicMock) -> None:
        """Verify default is returned for missing parameter."""
        mock_mlflow.get_run.return_value = _make_mock_run(params={})
        loader = _make_loader()

        assert loader.get_parameter("missing", default="fallback") == "fallback"

    @patch("utils.mlflow_manager.mlflow")
    def test_get_parameter_missing_returns_none(self, mock_mlflow: MagicMock) -> None:
        """Verify None is returned when no default is given."""
        mock_mlflow.get_run.return_value = _make_mock_run(params={})
        loader = _make_loader()

        assert loader.get_parameter("missing") is None


class TestMetrics:
    """Tests for get_metrics and get_metric."""

    @patch("utils.mlflow_manager.mlflow")
    def test_get_metrics_returns_all(self, mock_mlflow: MagicMock) -> None:
        """Verify all metrics are returned."""
        mock_mlflow.get_run.return_value = _make_mock_run(metrics={"loss": 0.5, "accuracy": 0.95})
        loader = _make_loader()

        assert loader.get_metrics() == {"loss": 0.5, "accuracy": 0.95}

    @patch("utils.mlflow_manager.mlflow")
    def test_get_metric_existing(self, mock_mlflow: MagicMock) -> None:
        """Verify a specific metric is returned."""
        mock_mlflow.get_run.return_value = _make_mock_run(metrics={"loss": 0.5})
        loader = _make_loader()

        assert loader.get_metric("loss") == 0.5

    @patch("utils.mlflow_manager.mlflow")
    def test_get_metric_missing_returns_default(self, mock_mlflow: MagicMock) -> None:
        """Verify default is returned for missing metric."""
        mock_mlflow.get_run.return_value = _make_mock_run(metrics={})
        loader = _make_loader()

        assert loader.get_metric("missing", default=-1.0) == -1.0

    @patch("utils.mlflow_manager.mlflow")
    def test_get_metric_missing_returns_none(self, mock_mlflow: MagicMock) -> None:
        """Verify None is returned when no default is given."""
        mock_mlflow.get_run.return_value = _make_mock_run(metrics={})
        loader = _make_loader()

        assert loader.get_metric("missing") is None


# ── JSON artifact ─────────────────────────────────────────────────────


class TestGetJsonArtifact:
    """Tests for get_json_artifact."""

    @patch("builtins.open", mock_open(read_data='{"key": "value"}'))
    @patch("utils.mlflow_manager.mlflow")
    def test_returns_parsed_json(self, mock_mlflow: MagicMock) -> None:
        """Verify JSON file is downloaded and parsed."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/data.json"
        loader = _make_loader()

        result = loader.get_json_artifact("data.json")

        assert result == {"key": "value"}

    @patch("builtins.open", mock_open(read_data='{"key": "value"}'))
    @patch("utils.mlflow_manager.mlflow")
    def test_passes_artifact_path(self, mock_mlflow: MagicMock) -> None:
        """Verify artifact_path is forwarded to download."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/sub/data.json"
        loader = _make_loader("run_x")

        loader.get_json_artifact("data.json", artifact_path="sub")

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_x", artifact_path="sub/data.json"
        )

    @patch("utils.mlflow_manager.mlflow")
    def test_raises_on_missing_file(self, mock_mlflow: MagicMock) -> None:
        """Verify FileNotFoundError propagates when artifact is missing."""
        mock_mlflow.artifacts.download_artifacts.side_effect = FileNotFoundError
        loader = _make_loader()

        with pytest.raises(FileNotFoundError):
            loader.get_json_artifact("missing.json")


# ── NPY artifact ──────────────────────────────────────────────────────


class TestGetNpyArtifact:
    """Tests for get_npy_artifact."""

    @patch("utils.mlflow_manager.np.load")
    @patch("utils.mlflow_manager.mlflow")
    def test_loads_npy_file(self, mock_mlflow: MagicMock, mock_np_load: MagicMock) -> None:
        """Verify numpy array is downloaded and loaded."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/data.npy"
        expected = np.array([1, 2, 3])
        mock_np_load.return_value = expected
        loader = _make_loader()

        result = loader.get_npy_artifact("data.npy")

        mock_np_load.assert_called_once_with("/tmp/data.npy", allow_pickle=False)
        np.testing.assert_array_equal(result, expected)

    @patch("utils.mlflow_manager.np.load")
    @patch("utils.mlflow_manager.mlflow")
    def test_passes_allow_pickle(self, mock_mlflow: MagicMock, mock_np_load: MagicMock) -> None:
        """Verify allow_pickle flag is forwarded."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/data.npy"
        mock_np_load.return_value = np.array([1])
        loader = _make_loader()

        loader.get_npy_artifact("data.npy", allow_pickle=True)

        mock_np_load.assert_called_once_with("/tmp/data.npy", allow_pickle=True)


# ── NPZ artifact ──────────────────────────────────────────────────────


class TestGetNpzArtifact:
    """Tests for get_npz_artifact."""

    @patch("utils.mlflow_manager.sp.load_npz")
    @patch("utils.mlflow_manager.mlflow")
    def test_loads_sparse_matrix(self, mock_mlflow: MagicMock, mock_load_npz: MagicMock) -> None:
        """Verify sparse matrix is loaded when return_sparse=True."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/matrix.npz"
        expected = sp.csr_matrix(np.eye(3))
        mock_load_npz.return_value = expected
        loader = _make_loader()

        result = loader.get_npz_artifact("matrix.npz")

        mock_load_npz.assert_called_once_with("/tmp/matrix.npz")
        assert sp.issparse(result)

    @patch("utils.mlflow_manager.np.load")
    @patch("utils.mlflow_manager.mlflow")
    def test_loads_raw_npz(self, mock_mlflow: MagicMock, mock_np_load: MagicMock) -> None:
        """Verify np.load is used when return_sparse=False."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/data.npz"
        mock_np_load.return_value = MagicMock()
        loader = _make_loader()

        loader.get_npz_artifact("data.npz", return_sparse=False)

        mock_np_load.assert_called_once_with("/tmp/data.npz")


# ── artifact_exists ───────────────────────────────────────────────────


class TestArtifactExists:
    """Tests for artifact_exists."""

    @patch("utils.mlflow_manager.mlflow")
    def test_returns_true_when_artifact_found(self, mock_mlflow: MagicMock) -> None:
        """Verify True when download succeeds."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/data.json"
        loader = _make_loader()

        assert loader.artifact_exists("data.json") is True

    @patch("utils.mlflow_manager.mlflow")
    def test_returns_false_on_file_not_found(self, mock_mlflow: MagicMock) -> None:
        """Verify False when FileNotFoundError is raised."""
        mock_mlflow.artifacts.download_artifacts.side_effect = FileNotFoundError
        loader = _make_loader()

        assert loader.artifact_exists("missing.json") is False

    @patch("utils.mlflow_manager.mlflow")
    def test_returns_false_on_os_error(self, mock_mlflow: MagicMock) -> None:
        """Verify False when OSError is raised."""
        mock_mlflow.artifacts.download_artifacts.side_effect = OSError
        loader = _make_loader()

        assert loader.artifact_exists("broken.json") is False


# ── Module-level convenience functions ────────────────────────────────


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @patch("utils.mlflow_manager.mlflow")
    def test_get_run_parameters(self, mock_mlflow: MagicMock) -> None:
        """Verify get_run_parameters delegates to MLflowRunLoader."""
        from utils.mlflow_manager import get_run_parameters

        mock_mlflow.get_run.return_value = _make_mock_run(params={"lr": "0.01"})

        assert get_run_parameters("run_1") == {"lr": "0.01"}

    @patch("utils.mlflow_manager.mlflow")
    def test_get_run_parameter(self, mock_mlflow: MagicMock) -> None:
        """Verify get_run_parameter delegates to MLflowRunLoader."""
        from utils.mlflow_manager import get_run_parameter

        mock_mlflow.get_run.return_value = _make_mock_run(params={"lr": "0.01"})

        assert get_run_parameter("run_1", "lr") == "0.01"

    @patch("builtins.open", mock_open(read_data='{"k": "v"}'))
    @patch("utils.mlflow_manager.mlflow")
    def test_get_json_artifact(self, mock_mlflow: MagicMock) -> None:
        """Verify get_json_artifact delegates to MLflowRunLoader."""
        from utils.mlflow_manager import get_json_artifact

        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/f.json"

        assert get_json_artifact("run_1", "f.json") == {"k": "v"}

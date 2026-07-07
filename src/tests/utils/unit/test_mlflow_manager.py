"""Unit tests for utils.mlflow_manager.

All MLflow interactions are mocked to keep tests fast and isolated.
"""

from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
import scipy.sparse as sp
from mlflow.exceptions import MlflowException

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


def _write_on_download(relative: str, content: str = "x"):
    """Create a download side effect writing *relative* under dst_path.

    Args:
        relative: Relative artifact path to create.
        content: File content to write.

    Returns:
        Callable mimicking ``mlflow.artifacts.download_artifacts``.
    """

    def _download(run_id: str, artifact_path: str, dst_path: str) -> str:  # noqa: ARG001
        target = Path(dst_path) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return str(target)

    return _download


def _file_info(path: str) -> MagicMock:
    """Create a mock FileInfo with the given artifact path."""
    info = MagicMock()
    info.path = path
    return info


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
    """Tests for download_artifact (single file, cached)."""

    @patch("utils.mlflow_manager.mlflow")
    def test_downloads_with_filename_only(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """Verify correct artifact_path when no subdirectory is given."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download(
                "data.json", '{"a": 1}'
            )
            loader = _make_loader("run_abc")

            result = loader.download_artifact("data.json")

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_abc", artifact_path="data.json", dst_path=ANY
        )
        assert result.endswith("data.json")
        assert Path(result).read_text() == '{"a": 1}'

    @patch("utils.mlflow_manager.mlflow")
    def test_downloads_with_artifact_path(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """Verify subdirectory is prepended to filename."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download(
                "model/config.json"
            )
            loader = _make_loader("run_abc")

            result = loader.download_artifact("config.json", artifact_path="model")

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_abc", artifact_path="model/config.json", dst_path=ANY
        )
        assert result.endswith("model/config.json")
        assert Path(result).exists()

    @patch("utils.mlflow_manager.mlflow")
    def test_repeated_download_hits_cache(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """Verify the second read reuses the cache without downloading."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download("data.json")
            loader = _make_loader("run_abc")

            first = loader.download_artifact("data.json")
            second = loader.download_artifact("data.json")

        assert first == second
        mock_mlflow.artifacts.download_artifacts.assert_called_once()

    @patch("utils.mlflow_manager.mlflow")
    def test_cache_is_per_run(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """Verify the same filename from another run downloads again."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download("data.json")

            first = _make_loader("run_a").download_artifact("data.json")
            second = _make_loader("run_b").download_artifact("data.json")

        assert first != second
        assert mock_mlflow.artifacts.download_artifacts.call_count == 2

    @patch("utils.mlflow_manager.mlflow")
    def test_file_entry_does_not_satisfy_dir_request(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        """Verify a cached single file never masquerades as its directory."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download(
                "model/weights.pt"
            )
            loader = _make_loader("run_abc")

            loader.download_artifact("weights.pt", artifact_path="model")
            loader.download_artifact_dir(artifact_path="model")

        assert mock_mlflow.artifacts.download_artifacts.call_count == 2

    @patch("utils.mlflow_manager.mlflow")
    def test_failed_download_is_not_cached(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """Verify an error leaves no cache entry, so a retry downloads again."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = FileNotFoundError("boom")
            loader = _make_loader("run_abc")

            with pytest.raises(FileNotFoundError):
                loader.download_artifact("data.json")

            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download("data.json")
            result = loader.download_artifact("data.json")

        assert Path(result).exists()
        assert mock_mlflow.artifacts.download_artifacts.call_count == 2


# ── download_artifact_dir ─────────────────────────────────────────────


class TestDownloadArtifactDir:
    """Tests for download_artifact_dir (directory, cached)."""

    @patch("utils.mlflow_manager.mlflow")
    def test_downloads_entire_artifact_dir(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """Verify empty string is used when no subdirectory is given."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download("a.txt")
            loader = _make_loader("run_abc")

            result = loader.download_artifact_dir()

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_abc", artifact_path="", dst_path=ANY
        )
        assert (Path(result) / "a.txt").exists()

    @patch("utils.mlflow_manager.mlflow")
    def test_downloads_subdirectory(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """Verify specific subdirectory is passed through."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download(
                "model/weights.pt"
            )
            loader = _make_loader("run_abc")

            result = loader.download_artifact_dir(artifact_path="model")

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_abc", artifact_path="model", dst_path=ANY
        )
        assert result.endswith("model")
        assert (Path(result) / "weights.pt").exists()

    @patch("utils.mlflow_manager.mlflow")
    def test_repeated_download_hits_cache(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """Verify the second directory read reuses the cache."""
        with patch("utils.mlflow_manager._CACHE_ROOT", tmp_path):
            mock_mlflow.artifacts.download_artifacts.side_effect = _write_on_download(
                "model/weights.pt"
            )
            loader = _make_loader("run_abc")

            first = loader.download_artifact_dir(artifact_path="model")
            second = loader.download_artifact_dir(artifact_path="model")

        assert first == second
        mock_mlflow.artifacts.download_artifacts.assert_called_once()


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

    @patch.object(MLflowRunLoader, "download_artifact")
    def test_returns_parsed_json(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Verify JSON file is downloaded and parsed."""
        json_file = tmp_path / "data.json"
        json_file.write_text('{"key": "value"}')
        mock_download.return_value = str(json_file)
        loader = _make_loader()

        result = loader.get_json_artifact("data.json")

        assert result == {"key": "value"}

    @patch.object(MLflowRunLoader, "download_artifact")
    def test_passes_artifact_path(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Verify artifact_path is forwarded to download."""
        json_file = tmp_path / "data.json"
        json_file.write_text("{}")
        mock_download.return_value = str(json_file)
        loader = _make_loader("run_x")

        loader.get_json_artifact("data.json", artifact_path="sub")

        mock_download.assert_called_once_with("data.json", "sub")

    @patch.object(MLflowRunLoader, "download_artifact")
    def test_raises_on_missing_file(self, mock_download: MagicMock) -> None:
        """Verify FileNotFoundError propagates when artifact is missing."""
        mock_download.side_effect = FileNotFoundError
        loader = _make_loader()

        with pytest.raises(FileNotFoundError):
            loader.get_json_artifact("missing.json")


# ── NPY artifact ──────────────────────────────────────────────────────


class TestGetNpyArtifact:
    """Tests for get_npy_artifact."""

    @patch.object(MLflowRunLoader, "download_artifact")
    def test_loads_npy_file(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Verify numpy array is downloaded and loaded."""
        npy_file = tmp_path / "data.npy"
        np.save(npy_file, np.array([1, 2, 3]))
        mock_download.return_value = str(npy_file)
        loader = _make_loader()

        result = loader.get_npy_artifact("data.npy")

        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    @patch("utils.mlflow_manager.np.load")
    @patch.object(MLflowRunLoader, "download_artifact")
    def test_passes_allow_pickle(self, mock_download: MagicMock, mock_np_load: MagicMock) -> None:
        """Verify allow_pickle flag is forwarded."""
        mock_download.return_value = "/tmp/data.npy"
        mock_np_load.return_value = np.array([1])
        loader = _make_loader()

        loader.get_npy_artifact("data.npy", allow_pickle=True)

        mock_np_load.assert_called_once_with("/tmp/data.npy", allow_pickle=True)


# ── NPZ artifact ──────────────────────────────────────────────────────


class TestGetNpzArtifact:
    """Tests for get_npz_artifact."""

    @patch.object(MLflowRunLoader, "download_artifact")
    def test_loads_sparse_matrix(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Verify sparse matrix is loaded when return_sparse=True."""
        npz_file = tmp_path / "matrix.npz"
        sp.save_npz(npz_file, sp.csr_matrix(np.eye(3)))
        mock_download.return_value = str(npz_file)
        loader = _make_loader()

        result = loader.get_npz_artifact("matrix.npz")

        assert sp.issparse(result)

    @patch("utils.mlflow_manager.np.load")
    @patch.object(MLflowRunLoader, "download_artifact")
    def test_loads_raw_npz(self, mock_download: MagicMock, mock_np_load: MagicMock) -> None:
        """Verify np.load is used when return_sparse=False."""
        mock_download.return_value = "/tmp/data.npz"
        mock_np_load.return_value = MagicMock()
        loader = _make_loader()

        loader.get_npz_artifact("data.npz", return_sparse=False)

        mock_np_load.assert_called_once_with("/tmp/data.npz")


# ── artifact_exists ───────────────────────────────────────────────────


class TestArtifactExists:
    """Tests for artifact_exists (metadata-only query)."""

    @patch("utils.mlflow_manager.mlflow")
    def test_returns_true_when_artifact_found(self, mock_mlflow: MagicMock) -> None:
        """Verify True when the listing contains the file."""
        client = mock_mlflow.tracking.MlflowClient.return_value
        client.list_artifacts.return_value = [_file_info("data.json")]
        loader = _make_loader("run_abc")

        assert loader.artifact_exists("data.json") is True
        client.list_artifacts.assert_called_once_with("run_abc", "")
        mock_mlflow.artifacts.download_artifacts.assert_not_called()

    @patch("utils.mlflow_manager.mlflow")
    def test_returns_false_when_absent(self, mock_mlflow: MagicMock) -> None:
        """Verify False when the listing lacks the file."""
        client = mock_mlflow.tracking.MlflowClient.return_value
        client.list_artifacts.return_value = [_file_info("other.json")]
        loader = _make_loader()

        assert loader.artifact_exists("missing.json") is False

    @patch("utils.mlflow_manager.mlflow")
    def test_checks_subdirectory(self, mock_mlflow: MagicMock) -> None:
        """Verify the subdirectory is listed and paths compared in full."""
        client = mock_mlflow.tracking.MlflowClient.return_value
        client.list_artifacts.return_value = [_file_info("model/weights.pt")]
        loader = _make_loader("run_abc")

        assert loader.artifact_exists("weights.pt", artifact_path="model") is True
        client.list_artifacts.assert_called_once_with("run_abc", "model")

    @patch("utils.mlflow_manager.mlflow")
    def test_returns_false_on_mlflow_error(self, mock_mlflow: MagicMock) -> None:
        """Verify False when the listing itself fails."""
        client = mock_mlflow.tracking.MlflowClient.return_value
        client.list_artifacts.side_effect = MlflowException("bad run")
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

    @patch.object(MLflowRunLoader, "download_artifact")
    def test_get_json_artifact(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Verify get_json_artifact delegates to MLflowRunLoader."""
        from utils.mlflow_manager import get_json_artifact

        json_file = tmp_path / "f.json"
        json_file.write_text('{"k": "v"}')
        mock_download.return_value = str(json_file)

        assert get_json_artifact("run_1", "f.json") == {"k": "v"}

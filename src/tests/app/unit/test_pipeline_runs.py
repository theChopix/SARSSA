"""Unit tests for app.core.pipeline_runs.

All MLflow interactions are mocked to keep tests fast and isolated.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.config.config import EXPERIMENT_NAME
from app.core.pipeline_runs import (
    get_pipeline_runs,
    get_run_context,
)


def _make_mock_run(
    run_id: str,
    run_name: str,
    status: str = "FINISHED",
    start_time: int = 1700000000000,
    parent_run_id: str | None = None,
) -> MagicMock:
    """Create a mock MLflow Run object.

    Args:
        run_id: Unique run identifier.
        run_name: Human-readable run name.
        status: Run status string.
        start_time: Start time in epoch milliseconds.
        parent_run_id: If set, simulates a nested (child) run.

    Returns:
        MagicMock: A mock with ``info``, ``data.tags``.
    """
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_run.info.run_name = run_name
    mock_run.info.status = status
    mock_run.info.start_time = start_time
    tags: dict[str, str] = {}
    if parent_run_id is not None:
        tags["mlflow.parentRunId"] = parent_run_id
    mock_run.data.tags = tags
    return mock_run


class TestGetPipelineRuns:
    """Tests for get_pipeline_runs."""

    @patch("app.core.pipeline_runs.mlflow")
    def test_returns_runs_with_expected_fields(self, mock_mlflow: MagicMock) -> None:
        """Verify each returned dict has run_id, run_name, status, start_time."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "42"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = [
            _make_mock_run("r1", "pipeline_run_2025-01-01"),
        ]

        runs = get_pipeline_runs()

        assert len(runs) == 1
        assert runs[0]["run_id"] == "r1"
        assert runs[0]["run_name"] == "pipeline_run_2025-01-01"
        assert runs[0]["status"] == "FINISHED"
        assert runs[0]["start_time"] == 1700000000000

    @patch("app.core.pipeline_runs.mlflow")
    def test_returns_multiple_runs(self, mock_mlflow: MagicMock) -> None:
        """Verify multiple runs are returned in order."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "42"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = [
            _make_mock_run("r1", "run_a"),
            _make_mock_run("r2", "run_b"),
            _make_mock_run("r3", "run_c"),
        ]

        runs = get_pipeline_runs()
        assert len(runs) == 3
        assert [r["run_id"] for r in runs] == ["r1", "r2", "r3"]

    @patch("app.core.pipeline_runs.mlflow")
    def test_returns_empty_list_when_no_runs(self, mock_mlflow: MagicMock) -> None:
        """Verify empty list is returned when experiment has no runs."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "42"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = []

        runs = get_pipeline_runs()
        assert runs == []

    @patch("app.core.pipeline_runs.mlflow")
    def test_raises_when_experiment_not_found(self, mock_mlflow: MagicMock) -> None:
        """Verify ValueError when the experiment does not exist."""
        mock_mlflow.get_experiment_by_name.return_value = None

        with pytest.raises(ValueError, match=EXPERIMENT_NAME):
            get_pipeline_runs()

    @patch("app.core.pipeline_runs.mlflow")
    def test_filters_out_nested_runs(self, mock_mlflow: MagicMock) -> None:
        """Verify nested (child) runs are excluded from results."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "42"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = [
            _make_mock_run("parent", "pipeline_run_1"),
            _make_mock_run("child", "step_run", parent_run_id="parent"),
        ]

        runs = get_pipeline_runs()

        assert len(runs) == 1
        assert runs[0]["run_id"] == "parent"


class TestGetRunContext:
    """Tests for get_run_context."""

    @patch("builtins.open", mock_open(read_data='{"dataset_loading": {"run_id": "abc123"}}'))
    @patch("app.core.pipeline_runs.mlflow")
    def test_returns_parsed_context(self, mock_mlflow: MagicMock) -> None:
        """Verify context.json contents are parsed and returned."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/context.json"

        context = get_run_context("parent_run_id")

        assert context == {"dataset_loading": {"run_id": "abc123"}}

    @patch("builtins.open", mock_open(read_data='{"dataset_loading": {"run_id": "abc123"}}'))
    @patch("app.core.pipeline_runs.mlflow")
    def test_calls_download_artifacts_correctly(self, mock_mlflow: MagicMock) -> None:
        """Verify download_artifacts is called with correct args."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/context.json"

        get_run_context("my_run_id")

        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="my_run_id", artifact_path="context.json"
        )

    @patch("app.core.pipeline_runs.mlflow")
    def test_raises_on_missing_artifact(self, mock_mlflow: MagicMock) -> None:
        """Verify FileNotFoundError propagates when artifact is missing."""
        mock_mlflow.artifacts.download_artifacts.side_effect = FileNotFoundError(
            "context.json not found"
        )

        with pytest.raises(FileNotFoundError):
            get_run_context("bad_run_id")

    @patch("builtins.open", mock_open(read_data="{}"))
    @patch("app.core.pipeline_runs.mlflow")
    def test_returns_empty_dict_for_empty_context(self, mock_mlflow: MagicMock) -> None:
        """Verify empty context.json returns empty dict."""
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/context.json"

        context = get_run_context("run_id")
        assert context == {}

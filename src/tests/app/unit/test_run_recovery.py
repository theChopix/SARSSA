"""Unit tests for app.core.run_recovery.

All MLflow interactions are mocked.
"""

from unittest.mock import MagicMock, call, patch

from app.core.run_recovery import RECOVERY_TAG, fail_orphaned_runs


class _Page(list):
    """Stand-in for MLflow's PagedList: a list carrying a ``.token``."""

    def __init__(self, runs: list, token: str | None = None) -> None:
        super().__init__(runs)
        self.token = token


def _run(run_id: str) -> MagicMock:
    """Create a mock run with the given ``info.run_id``."""
    run = MagicMock()
    run.info.run_id = run_id
    return run


def _experiment(experiment_id: str, name: str) -> MagicMock:
    """Create a mock experiment (``name`` set post-hoc — it's reserved
    in the MagicMock constructor)."""
    experiment = MagicMock(experiment_id=experiment_id)
    experiment.name = name
    return experiment


def _arm(mock_mlflow: MagicMock, pages: list[_Page]) -> MagicMock:
    """Wire ``mlflow`` so the experiments resolve and search returns *pages*.

    Returns the mock ``MlflowClient`` instance for assertions.
    """
    mock_mlflow.search_experiments.return_value = [
        _experiment("0", "Default"),
        _experiment("1", "pipeline_experiments"),
    ]
    client = mock_mlflow.tracking.MlflowClient.return_value
    client.search_runs.side_effect = pages
    return client


class TestFailOrphanedRuns:
    """Tests for fail_orphaned_runs."""

    @patch("app.core.run_recovery.mlflow")
    def test_marks_running_runs_failed_with_tag(self, mock_mlflow: MagicMock) -> None:
        """Each RUNNING run is tagged and terminated as FAILED."""
        client = _arm(mock_mlflow, [_Page([_run("parent"), _run("child")])])

        result = fail_orphaned_runs("terminated_at_shutdown")

        assert result == ["parent", "child"]
        client.set_tag.assert_has_calls(
            [
                call("parent", RECOVERY_TAG, "terminated_at_shutdown"),
                call("child", RECOVERY_TAG, "terminated_at_shutdown"),
            ]
        )
        client.set_terminated.assert_has_calls(
            [call("parent", status="FAILED"), call("child", status="FAILED")]
        )

    @patch("app.core.run_recovery.mlflow")
    def test_no_experiment_returns_empty(self, mock_mlflow: MagicMock) -> None:
        """Only Default (excluded) existing is a no-op, not an error."""
        mock_mlflow.search_experiments.return_value = [_experiment("0", "Default")]

        assert fail_orphaned_runs("terminated_at_shutdown") == []
        mock_mlflow.tracking.MlflowClient.assert_not_called()

    @patch("app.core.run_recovery.mlflow")
    def test_no_running_runs_is_noop(self, mock_mlflow: MagicMock) -> None:
        """An empty result set terminates nothing."""
        client = _arm(mock_mlflow, [_Page([])])

        assert fail_orphaned_runs("terminated_at_shutdown") == []
        client.set_terminated.assert_not_called()

    @patch("app.core.run_recovery.mlflow")
    def test_paginates_until_token_exhausted(self, mock_mlflow: MagicMock) -> None:
        """All pages are swept until ``token`` is falsy."""
        client = _arm(
            mock_mlflow,
            [_Page([_run("a")], token="next"), _Page([_run("b")], token=None)],
        )

        assert fail_orphaned_runs("terminated_at_shutdown") == ["a", "b"]
        assert client.search_runs.call_count == 2

    @patch("app.core.run_recovery.mlflow")
    def test_continues_after_failure_on_one_run(self, mock_mlflow: MagicMock) -> None:
        """A failure terminating one run does not abort the sweep."""
        client = _arm(mock_mlflow, [_Page([_run("bad"), _run("good")])])
        client.set_terminated.side_effect = [RuntimeError("boom"), None]

        assert fail_orphaned_runs("terminated_at_shutdown") == ["good"]

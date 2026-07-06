"""Unit tests for app.core.pipeline_engine.

All MLflow and PluginManager interactions are mocked.
"""

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.core.pipeline_engine import PipelineEngine


def _mock_start_run(run_id: str = "parent_id") -> MagicMock:
    """Create a mock MLflow ActiveRun context manager.

    Args:
        run_id: Run ID to assign to the mock.
    Returns:
        MagicMock: Context manager with ``info.run_id``.
    """
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    return mock_run


def _arm_execution_order(mock_mlflow: MagicMock, existing: int = 0, offset: int = 0) -> None:
    """Make ``_next_execution_order`` resolve under a mocked ``mlflow``.

    ``_next_execution_order`` counts the parent run's existing nested children
    via ``MlflowClient.search_runs``, adds one, and adds the inherited-step
    offset read from the parent run's tags.  Under a bare mock, ``search_runs``
    returns a ``MagicMock`` that ``len()`` rejects and ``get_run().data.tags``
    is not a real dict, so tests must supply both.

    Args:
        mock_mlflow: The patched ``mlflow`` module mock.
        existing: How many nested children already exist.
        offset: Inherited-step offset stored on the parent run.
    """
    client = mock_mlflow.tracking.MlflowClient.return_value
    client.search_runs.return_value = [MagicMock() for _ in range(existing)]
    client.get_run.return_value.data.tags = {"sarssa.order_offset": str(offset)} if offset else {}


class TestStartRun:
    """Tests for PipelineEngine.start_run."""

    @patch("app.core.pipeline_engine.mlflow")
    def test_returns_run_id(self, mock_mlflow: MagicMock) -> None:
        """Verify start_run returns the parent run ID."""
        mock_mlflow.start_run.return_value = _mock_start_run("abc123")

        engine = PipelineEngine()
        run_id = engine.start_run()

        assert run_id == "abc123"

    @patch("app.core.pipeline_engine.mlflow")
    def test_sets_experiment(self, mock_mlflow: MagicMock) -> None:
        """Verify the MLflow experiment is set."""
        mock_mlflow.start_run.return_value = _mock_start_run()

        engine = PipelineEngine()
        engine.start_run()

        mock_mlflow.set_experiment.assert_called_once()

    @patch("app.core.pipeline_engine.mlflow")
    def test_ends_run_after_creation(self, mock_mlflow: MagicMock) -> None:
        """Verify end_run is called to release the active run context."""
        mock_mlflow.start_run.return_value = _mock_start_run()

        engine = PipelineEngine()
        engine.start_run()

        mock_mlflow.end_run.assert_called_once()

    @patch("app.core.pipeline_engine.mlflow")
    def test_raises_if_run_already_active(self, mock_mlflow: MagicMock) -> None:
        """Verify RuntimeError when a run is already in progress."""
        mock_mlflow.start_run.return_value = _mock_start_run()

        engine = PipelineEngine()
        engine.start_run()

        with pytest.raises(RuntimeError, match="already in progress"):
            engine.start_run()

    @patch("app.core.pipeline_engine.mlflow")
    def test_prefixes_tags_with_sarssa(self, mock_mlflow: MagicMock) -> None:
        """Verify user tags are prefixed with sarssa. in mlflow.start_run()."""
        mock_mlflow.start_run.return_value = _mock_start_run("run_42")

        engine = PipelineEngine()
        engine.start_run(tags={"dataset": "MovieLens", "model": "ELSA"})

        _, kwargs = mock_mlflow.start_run.call_args
        assert kwargs["tags"] == {
            "sarssa.dataset": "MovieLens",
            "sarssa.model": "ELSA",
        }

    @patch("app.core.pipeline_engine.mlflow")
    def test_passes_description_to_mlflow(self, mock_mlflow: MagicMock) -> None:
        """Verify description is passed to mlflow.start_run()."""
        mock_mlflow.start_run.return_value = _mock_start_run("run_42")

        engine = PipelineEngine()
        engine.start_run(description="Baseline run")

        _, kwargs = mock_mlflow.start_run.call_args
        assert kwargs["description"] == "Baseline run"

    @patch("app.core.pipeline_engine.mlflow")
    def test_passes_none_when_no_tags_or_description(self, mock_mlflow: MagicMock) -> None:
        """Verify tags=None and description=None when not provided."""
        mock_mlflow.start_run.return_value = _mock_start_run("run_42")

        engine = PipelineEngine()
        engine.start_run()

        _, kwargs = mock_mlflow.start_run.call_args
        assert kwargs["tags"] is None
        assert kwargs["description"] is None

    @patch("app.core.pipeline_engine.mlflow")
    def test_stores_order_offset_tag_and_marks_name(self, mock_mlflow: MagicMock) -> None:
        """Verify a positive order_offset is tagged and marks the run name."""
        mock_mlflow.start_run.return_value = _mock_start_run("run_42")

        engine = PipelineEngine()
        engine.start_run(order_offset=2)

        _, kwargs = mock_mlflow.start_run.call_args
        assert kwargs["tags"]["sarssa.order_offset"] == "2"
        assert "( inherited )" in kwargs["run_name"]

    @patch("app.core.pipeline_engine.mlflow")
    def test_no_offset_tag_or_marker_when_zero(self, mock_mlflow: MagicMock) -> None:
        """Verify a zero order_offset adds no tag and no name marker."""
        mock_mlflow.start_run.return_value = _mock_start_run("run_42")

        engine = PipelineEngine()
        engine.start_run(order_offset=0)

        _, kwargs = mock_mlflow.start_run.call_args
        assert kwargs["tags"] is None
        assert "( inherited )" not in kwargs["run_name"]

    @patch("app.core.pipeline_engine.datetime")
    @patch("app.core.pipeline_engine.mlflow")
    def test_run_name_uses_configured_timezone(
        self, mock_mlflow: MagicMock, mock_datetime: MagicMock
    ) -> None:
        """Verify the run name timestamp is built in the configured timezone."""
        from app.config.config import TIMEZONE

        mock_mlflow.start_run.return_value = _mock_start_run("run_42")
        fixed = datetime.datetime(2026, 5, 31, 14, 7)
        mock_datetime.datetime.now.return_value = fixed

        engine = PipelineEngine()
        engine.start_run(pipeline_name="Baseline ELSA")

        # now() must be called with the configured tz, not naive/UTC.
        mock_datetime.datetime.now.assert_called_once_with(TIMEZONE)
        _, kwargs = mock_mlflow.start_run.call_args
        assert kwargs["run_name"] == "Pipeline Run | Baseline ELSA [ 31/05/2026 | 14:07 ]"


class TestExecuteStep:
    """Tests for PipelineEngine.execute_step."""

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_updates_context_with_run_id(self, mock_mlflow: MagicMock, _mock_pm: MagicMock) -> None:
        """Verify context gets the step's run_id under its category."""
        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
            # dataset_loading steps reopen the parent run once more to log
            # the dataset input (populates the MLflow "Dataset" column).
            _mock_start_run("parent_id"),
        ]
        _arm_execution_order(mock_mlflow)

        engine = PipelineEngine()
        engine.start_run()

        context: dict[str, Any] = {}
        result = engine.execute_step("dataset_loading.loader.loader", {}, context)

        assert result["dataset_loading"] == {"run_id": "step_id"}

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_nested_run_name_uses_execution_order(
        self,
        mock_mlflow: MagicMock,
        mock_pm: MagicMock,
    ) -> None:
        """Verify the nested run name is '[<order>] <category> / <plugin>'.

        The bracket number is the execution sequence (existing children
        + 1), not the category's config order; the plugin's own ``name``
        is preferred for the label.
        """
        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
            # dataset_loading steps reopen the parent run once more to log
            # the dataset input (populates the MLflow "Dataset" column).
            _mock_start_run("parent_id"),
        ]
        # Three nested children already exist → this step is the 4th.
        _arm_execution_order(mock_mlflow, existing=3)
        mock_plugin = MagicMock()
        mock_plugin.name = "MovieLens Loader"
        mock_pm.load.return_value = mock_plugin

        engine = PipelineEngine()
        engine.start_run()
        engine.execute_step("dataset_loading.movieLens_loader.movieLens_loader", {}, {})

        # The nested step run is the call carrying nested=True (no longer the
        # last start_run call, since dataset-input logging reopens the parent).
        nested_call = next(
            c for c in mock_mlflow.start_run.call_args_list if c.kwargs.get("nested")
        )
        assert nested_call.kwargs["run_name"] == "[4] Dataset Loading / MovieLens Loader"

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_calls_plugin_run_with_params(self, mock_mlflow: MagicMock, mock_pm: MagicMock) -> None:
        """Verify plugin.run() is called with only keyword params."""
        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)
        mock_plugin = MagicMock()
        mock_pm.load.return_value = mock_plugin

        engine = PipelineEngine()
        engine.start_run()

        context: dict[str, Any] = {}
        engine.execute_step("training_cfm.trainer.trainer", {"epochs": 10}, context)

        mock_plugin.run.assert_called_once_with(epochs=10)

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_calls_load_context_before_run(
        self,
        mock_mlflow: MagicMock,
        mock_pm: MagicMock,
    ) -> None:
        """Verify load_context(context) is called before run()."""
        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)
        mock_plugin = MagicMock()
        mock_pm.load.return_value = mock_plugin

        engine = PipelineEngine()
        engine.start_run()

        context: dict[str, Any] = {"existing": {"run_id": "prev"}}
        engine.execute_step("cat.impl.impl", {}, context)

        mock_plugin.load_context.assert_called_once_with(context)

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_calls_update_context_after_run(
        self,
        mock_mlflow: MagicMock,
        mock_pm: MagicMock,
    ) -> None:
        """Verify update_context() is called after run()."""
        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)
        mock_plugin = MagicMock()
        mock_pm.load.return_value = mock_plugin

        engine = PipelineEngine()
        engine.start_run()
        engine.execute_step("cat.impl.impl", {}, {})

        mock_plugin.update_context.assert_called_once()

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_lifecycle_call_order(
        self,
        mock_mlflow: MagicMock,
        mock_pm: MagicMock,
    ) -> None:
        """Verify load_context → run → update_context ordering."""
        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)
        call_order: list[str] = []
        mock_plugin = MagicMock()
        mock_plugin.load_context.side_effect = (
            lambda ctx: call_order.append("load_context")  # noqa: ARG005
        )
        mock_plugin.run.side_effect = (
            lambda **kw: call_order.append("run")  # noqa: ARG005
        )
        mock_plugin.update_context.side_effect = lambda: call_order.append("update_context")
        mock_pm.load.return_value = mock_plugin

        engine = PipelineEngine()
        engine.start_run()
        engine.execute_step("cat.impl.impl", {"x": 1}, {})

        assert call_order == ["load_context", "run", "update_context"]

    @patch("app.core.pipeline_engine.mlflow")
    def test_raises_without_active_run(self, _mock_mlflow: MagicMock) -> None:
        """Verify RuntimeError when no parent run is active."""
        engine = PipelineEngine()

        with pytest.raises(RuntimeError, match="start_run"):
            engine.execute_step("plugin.name.name", {}, {})

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_loads_correct_plugin(self, mock_mlflow: MagicMock, mock_pm: MagicMock) -> None:
        """Verify PluginManager.load is called with the plugin name."""
        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)

        engine = PipelineEngine()
        engine.start_run()
        engine.execute_step("inspection.inspector.inspector", {}, {})

        mock_pm.load.assert_called_once_with("inspection.inspector.inspector")

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_injects_notifier_when_provided(
        self,
        mock_mlflow: MagicMock,
        mock_pm: MagicMock,
    ) -> None:
        """Verify plugin.notifier is set to the provided notifier."""
        from utils.plugin_notifier import PluginNotifier

        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)
        mock_plugin = MagicMock()
        mock_pm.load.return_value = mock_plugin
        notifier = PluginNotifier()

        engine = PipelineEngine()
        engine.start_run()
        engine.execute_step("cat.impl.impl", {}, {}, notifier=notifier)

        assert mock_plugin.notifier is notifier

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_injects_cancellation_when_provided(
        self,
        mock_mlflow: MagicMock,
        mock_pm: MagicMock,
    ) -> None:
        """Verify plugin.cancellation is set to the provided token."""
        import threading

        from utils.cancellation import CancellationToken

        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)
        mock_plugin = MagicMock()
        mock_pm.load.return_value = mock_plugin
        token = CancellationToken(threading.Event())

        engine = PipelineEngine()
        engine.start_run()
        engine.execute_step("cat.impl.impl", {}, {}, cancellation=token)

        assert mock_plugin.cancellation is token

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_does_not_set_notifier_when_none(
        self,
        mock_mlflow: MagicMock,
        mock_pm: MagicMock,
    ) -> None:
        """Verify plugin.notifier is untouched when notifier=None."""
        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)
        mock_plugin = MagicMock()
        original_notifier = mock_plugin.notifier
        mock_pm.load.return_value = mock_plugin

        engine = PipelineEngine()
        engine.start_run()
        engine.execute_step("cat.impl.impl", {}, {}, notifier=None)

        assert mock_plugin.notifier is original_notifier

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_notifier_injected_before_run(
        self,
        mock_mlflow: MagicMock,
        mock_pm: MagicMock,
    ) -> None:
        """Verify notifier is set before load_context/run() are called."""
        from utils.plugin_notifier import PluginNotifier

        mock_mlflow.start_run.side_effect = [
            _mock_start_run("parent_id"),
            _mock_start_run("parent_id"),
            _mock_start_run("step_id"),
        ]
        _arm_execution_order(mock_mlflow)
        notifier = PluginNotifier()
        seen_notifier: list[Any] = []

        mock_plugin = MagicMock()
        mock_plugin.load_context.side_effect = lambda ctx: seen_notifier.append(  # noqa: ARG005
            mock_plugin.notifier
        )
        mock_pm.load.return_value = mock_plugin

        engine = PipelineEngine()
        engine.start_run()
        engine.execute_step("cat.impl.impl", {}, {}, notifier=notifier)

        assert seen_notifier[0] is notifier


class TestFinalizeRun:
    """Tests for PipelineEngine.finalize_run."""

    @patch("app.core.pipeline_engine.mlflow")
    def test_logs_context_json(self, mock_mlflow: MagicMock) -> None:
        """Verify context.json is logged to the parent run."""
        mock_mlflow.start_run.return_value = _mock_start_run("parent_id")

        engine = PipelineEngine()
        engine.start_run()

        context = {"dataset_loading": {"run_id": "abc"}}
        engine.finalize_run(context)

        mock_mlflow.log_dict.assert_called_once_with(context, "context.json")

    @patch("app.core.pipeline_engine.mlflow")
    def test_clears_parent_run_id(self, mock_mlflow: MagicMock) -> None:
        """Verify _parent_run_id is reset after finalization."""
        mock_mlflow.start_run.return_value = _mock_start_run("parent_id")

        engine = PipelineEngine()
        engine.start_run()
        engine.finalize_run({})

        assert engine._parent_run_id is None

    @patch("app.core.pipeline_engine.mlflow")
    def test_raises_without_active_run(self, _mock_mlflow: MagicMock) -> None:
        """Verify RuntimeError when no parent run is active."""
        engine = PipelineEngine()

        with pytest.raises(RuntimeError, match="start_run"):
            engine.finalize_run({})

    @patch("app.core.pipeline_engine.mlflow")
    def test_allows_new_run_after_finalize(self, mock_mlflow: MagicMock) -> None:
        """Verify a new run can be started after finalization."""
        mock_mlflow.start_run.return_value = _mock_start_run("run_1")

        engine = PipelineEngine()
        engine.start_run()
        engine.finalize_run({})

        mock_mlflow.start_run.return_value = _mock_start_run("run_2")
        run_id = engine.start_run()
        assert run_id == "run_2"


class TestResumeRun:
    """Tests for PipelineEngine.resume_run."""

    @patch("app.core.pipeline_engine.mlflow")
    def test_sets_parent_run_id(self, _mock_mlflow: MagicMock) -> None:
        """Verify _parent_run_id is set to the given run_id."""
        engine = PipelineEngine()
        engine.resume_run("existing_run_42")

        assert engine._parent_run_id == "existing_run_42"

    @patch("app.core.pipeline_engine.mlflow")
    def test_raises_if_run_already_active(self, mock_mlflow: MagicMock) -> None:
        """Verify RuntimeError when a run is already in progress."""
        mock_mlflow.start_run.return_value = _mock_start_run()

        engine = PipelineEngine()
        engine.start_run()

        with pytest.raises(RuntimeError, match="already in progress"):
            engine.resume_run("other_run")

    @patch("app.core.pipeline_engine.mlflow")
    def test_allows_execute_step_after_resume(self, mock_mlflow: MagicMock) -> None:
        """Verify execute_step works after resume_run."""
        mock_mlflow.start_run.return_value = _mock_start_run("step_id")
        _arm_execution_order(mock_mlflow)

        engine = PipelineEngine()
        engine.resume_run("parent_id")

        with patch("app.core.pipeline_engine.PluginManager"):
            ctx: dict[str, Any] = {}
            engine.execute_step("cat.impl.impl", {}, ctx)

        assert "cat" in ctx


class TestFailRun:
    """Tests for PipelineEngine.fail_run."""

    @patch("app.core.pipeline_engine.mlflow")
    def test_logs_context_json(self, mock_mlflow: MagicMock) -> None:
        """Verify context.json is logged to the parent run."""
        mock_mlflow.start_run.return_value = _mock_start_run("parent_id")
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        engine = PipelineEngine()
        engine.start_run()

        context = {"dataset_loading": {"run_id": "abc"}}
        engine.fail_run(context)

        mock_mlflow.log_dict.assert_called_once_with(context, "context.json")

    @patch("app.core.pipeline_engine.mlflow")
    def test_sets_cancellation_tag(self, mock_mlflow: MagicMock) -> None:
        """Verify cancellation tag is set on the parent run."""
        mock_mlflow.start_run.return_value = _mock_start_run("parent_id")
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        engine = PipelineEngine()
        engine.start_run()

        engine.fail_run({})

        mock_mlflow.set_tag.assert_called_once_with("cancellation", "cancelled_by_user")

    @patch("app.core.pipeline_engine.mlflow")
    def test_marks_run_as_failed(self, mock_mlflow: MagicMock) -> None:
        """Verify the run is marked FAILED via MlflowClient."""
        mock_mlflow.start_run.return_value = _mock_start_run("parent_id")
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        engine = PipelineEngine()
        engine.start_run()
        run_id = engine._parent_run_id

        engine.fail_run({})

        mock_client.set_terminated.assert_called_once_with(run_id, status="FAILED")

    @patch("app.core.pipeline_engine.mlflow")
    def test_clears_parent_run_id(self, mock_mlflow: MagicMock) -> None:
        """Verify _parent_run_id is reset after fail_run."""
        mock_mlflow.start_run.return_value = _mock_start_run("parent_id")
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        engine = PipelineEngine()
        engine.start_run()
        engine.fail_run({})

        assert engine._parent_run_id is None

    @patch("app.core.pipeline_engine.mlflow")
    def test_raises_without_active_run(self, _mock_mlflow: MagicMock) -> None:
        """Verify RuntimeError when no parent run is active."""
        engine = PipelineEngine()

        with pytest.raises(RuntimeError, match="start_run"):
            engine.fail_run({})


class TestBatchRun:
    """Tests for PipelineEngine.run (batch mode)."""

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_executes_all_steps(self, mock_mlflow: MagicMock, mock_pm: MagicMock) -> None:
        """Verify all steps are executed and context is returned."""
        step_counter = {"n": 0}

        def mock_start(**_kwargs: Any) -> MagicMock:
            step_counter["n"] += 1
            return _mock_start_run(f"run_{step_counter['n']}")

        mock_mlflow.start_run.side_effect = mock_start

        steps = [
            {"plugin": "cat_a.impl.impl", "params": {"x": 1}},
            {"plugin": "cat_b.impl.impl", "params": {}},
        ]

        engine = PipelineEngine(steps)
        context = engine.run({})

        assert "cat_a" in context
        assert "cat_b" in context
        assert mock_pm.load.call_count == 2

    @patch("app.core.pipeline_engine.PluginManager")
    @patch("app.core.pipeline_engine.mlflow")
    def test_finalizes_after_all_steps(self, mock_mlflow: MagicMock, _mock_pm: MagicMock) -> None:
        """Verify _parent_run_id is None after batch run completes."""
        mock_mlflow.start_run.return_value = _mock_start_run("parent")

        engine = PipelineEngine([{"plugin": "cat.p.p", "params": {}}])
        engine.run({})

        assert engine._parent_run_id is None


class TestNextExecutionOrder:
    """Tests for the inherited-step offset in nested step numbering."""

    @patch("app.core.pipeline_engine.mlflow")
    def test_adds_inherited_offset(self, mock_mlflow: MagicMock) -> None:
        """Verify order == offset + existing children + 1."""
        engine = PipelineEngine()
        engine._parent_run_id = "parent"
        _arm_execution_order(mock_mlflow, existing=1, offset=2)

        assert engine._next_execution_order() == 4

    @patch("app.core.pipeline_engine.mlflow")
    def test_no_offset_when_tag_absent(self, mock_mlflow: MagicMock) -> None:
        """Verify a fresh run (no offset tag) numbers from children + 1."""
        engine = PipelineEngine()
        engine._parent_run_id = "parent"
        _arm_execution_order(mock_mlflow, existing=2, offset=0)

        assert engine._next_execution_order() == 3

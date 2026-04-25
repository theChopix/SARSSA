"""Unit tests for app.core.pipeline_worker.

All PipelineEngine interactions are mocked.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from app.core.pipeline_worker import run_pipeline_worker, run_step_worker
from app.models.pipeline import TaskState


def _make_task(steps: list[dict[str, Any]] | None = None) -> TaskState:
    """Create a TaskState with sensible defaults for testing.

    Args:
        steps: Optional step dicts. Defaults to two dummy steps.

    Returns:
        TaskState: A fresh task in "running" status.
    """
    if steps is None:
        steps = [
            {"plugin": "cat_a.impl.impl", "params": {"x": 1}},
            {"plugin": "cat_b.impl.impl", "params": {}},
        ]
    return TaskState(task_id="test-task", steps_requested=steps)


class TestWorkerSuccess:
    """Tests for successful pipeline execution."""

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_status_completed(self, mock_engine_cls: MagicMock) -> None:
        """Verify status is set to 'completed' on success."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "parent_id"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            category = plugin.split(".")[0]
            ctx[category] = {"run_id": f"{category}_run"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert task.status == "completed"

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_run_id_set(self, mock_engine_cls: MagicMock) -> None:
        """Verify run_id is set from engine.start_run()."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "my_run_42"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert task.run_id == "my_run_42"

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_completed_steps_populated(self, mock_engine_cls: MagicMock) -> None:
        """Verify completed_steps has an entry for each step."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            category = plugin.split(".")[0]
            ctx[category] = {"run_id": f"{category}_run"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert len(task.completed_steps) == 2
        assert task.completed_steps[0] == {"category": "cat_a", "run_id": "cat_a_run"}
        assert task.completed_steps[1] == {"category": "cat_b", "run_id": "cat_b_run"}

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_context_set(self, mock_engine_cls: MagicMock) -> None:
        """Verify task.context is set on completion."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            category = plugin.split(".")[0]
            ctx[category] = {"run_id": f"{category}_run"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert task.context is not None
        assert "cat_a" in task.context
        assert "cat_b" in task.context

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_current_step_none_after_completion(self, mock_engine_cls: MagicMock) -> None:
        """Verify current_step is cleared after success."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert task.current_step is None

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_finalize_run_called(self, mock_engine_cls: MagicMock) -> None:
        """Verify finalize_run is called with the final context."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        mock_engine.finalize_run.assert_called_once()
        finalized_ctx = mock_engine.finalize_run.call_args[0][0]
        assert "cat_a" in finalized_ctx
        assert "cat_b" in finalized_ctx

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_current_step_updated_during_execution(self, mock_engine_cls: MagicMock) -> None:
        """Verify current_step and current_step_index are set for each step."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        observed_steps: list[tuple[str | None, int]] = []

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            # Capture what current_step was set to before execute_step ran.
            observed_steps.append((task.current_step, task.current_step_index))
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert observed_steps == [("cat_a", 0), ("cat_b", 1)]


class TestWorkerTags:
    """Tests for tags/description forwarding to the engine."""

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_passes_tags_to_engine(self, mock_engine_cls: MagicMock) -> None:
        """Verify tags are forwarded to engine.start_run()."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "run_123"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        task.tags = {"dataset": "MovieLens", "model": "ELSA"}
        run_pipeline_worker(task)

        mock_engine.start_run.assert_called_once_with(
            tags={"dataset": "MovieLens", "model": "ELSA"},
            description="",
        )

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_passes_description_to_engine(self, mock_engine_cls: MagicMock) -> None:
        """Verify description is forwarded to engine.start_run()."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "run_123"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        task.description = "Baseline run"
        run_pipeline_worker(task)

        mock_engine.start_run.assert_called_once_with(tags={}, description="Baseline run")

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_passes_empty_defaults_to_engine(self, mock_engine_cls: MagicMock) -> None:
        """Verify empty tags/description are passed when not set."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "run_123"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        mock_engine.start_run.assert_called_once_with(tags={}, description="")


class TestWorkerCancellation:
    """Tests for cooperative cancellation via cancel_event."""

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_cancel_before_first_step(self, mock_engine_cls: MagicMock) -> None:
        """Verify immediate cancellation when cancel_event is set before start."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "parent_id"

        task = _make_task()
        task.cancel_event.set()
        run_pipeline_worker(task)

        assert task.status == "cancelled"
        assert task.error == "Pipeline cancelled by user."
        mock_engine.execute_step.assert_not_called()

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_cancel_between_steps(self, mock_engine_cls: MagicMock) -> None:
        """Verify cancellation after first step completes but before second."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "parent_id"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            category = plugin.split(".")[0]
            ctx[category] = {"run_id": f"{category}_run"}
            # Set cancel after the first step executes.
            task.cancel_event.set()
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert task.status == "cancelled"
        assert len(task.completed_steps) == 1
        assert task.completed_steps[0]["category"] == "cat_a"

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_cancel_calls_fail_run(self, mock_engine_cls: MagicMock) -> None:
        """Verify fail_run is called with partial context on cancellation."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "parent_id"

        task = _make_task()
        task.cancel_event.set()
        run_pipeline_worker(task)

        mock_engine.fail_run.assert_called_once()
        mock_engine.finalize_run.assert_not_called()

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_cancel_does_not_set_context(self, mock_engine_cls: MagicMock) -> None:
        """Verify task.context remains None after cancellation."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "parent_id"

        task = _make_task()
        task.cancel_event.set()
        run_pipeline_worker(task)

        assert task.context is None

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_no_cancel_runs_normally(self, mock_engine_cls: MagicMock) -> None:
        """Verify pipeline completes normally when cancel_event is never set."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "parent_id"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert task.status == "completed"
        mock_engine.fail_run.assert_not_called()
        mock_engine.finalize_run.assert_called_once()


class TestWorkerFailure:
    """Tests for pipeline execution failure."""

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_status_error_on_exception(self, mock_engine_cls: MagicMock) -> None:
        """Verify status is set to 'error' when a step raises."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"
        mock_engine.execute_step.side_effect = RuntimeError("GPU OOM")

        task = _make_task()
        run_pipeline_worker(task)

        assert task.status == "error"

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_error_message_stored(self, mock_engine_cls: MagicMock) -> None:
        """Verify the exception message is stored in task.error."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"
        mock_engine.execute_step.side_effect = ValueError("bad param")

        task = _make_task()
        run_pipeline_worker(task)

        assert task.error == "bad param"

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_partial_completed_steps_on_failure(self, mock_engine_cls: MagicMock) -> None:
        """Verify completed_steps contains only steps that finished before the error."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        call_count = {"n": 0}

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("step 2 failed")
            ctx[plugin.split(".")[0]] = {"run_id": "r1"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task()
        run_pipeline_worker(task)

        assert len(task.completed_steps) == 1
        assert task.completed_steps[0]["category"] == "cat_a"


def _make_step_task(
    run_id: str = "parent_run_1",
    plugin: str = "labeling_evaluation.impl.impl",
    params: dict[str, Any] | None = None,
) -> TaskState:
    """Create a TaskState pre-configured for run_step_worker testing.

    Args:
        run_id: Pre-existing MLflow parent run ID.
        plugin: Plugin module path for the single step.
        params: Optional params dict for the step.

    Returns:
        TaskState: A fresh task with run_id and one step set.
    """
    step = {"plugin": plugin, "params": params or {}}
    task = TaskState(task_id="step-task", steps_requested=[step])
    task.run_id = run_id
    return task


class TestStepWorkerSuccess:
    """Tests for successful single-step execution via run_step_worker."""

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_status_completed(self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock) -> None:
        """Verify status is set to 'completed' on success."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "step_run_1"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task()
        run_step_worker(task)

        assert task.status == "completed"

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_completed_steps_populated(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify completed_steps has one entry after the step finishes."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            category = plugin.split(".")[0]
            ctx[category] = {"run_id": "step_run_42"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task(plugin="labeling_evaluation.impl.impl")
        run_step_worker(task)

        assert len(task.completed_steps) == 1
        assert task.completed_steps[0] == {
            "category": "labeling_evaluation",
            "run_id": "step_run_42",
        }

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_context_set(self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock) -> None:
        """Verify task.context is set to the updated context on completion."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        initial_ctx = {"dataset_loading": {"run_id": "ds_run"}}
        mock_get_ctx.return_value = initial_ctx

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "new_run"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task()
        run_step_worker(task)

        assert task.context is not None
        assert "dataset_loading" in task.context
        assert "labeling_evaluation" in task.context

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_resume_run_called_with_run_id(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify engine.resume_run is called with task.run_id."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task(run_id="my_parent_run")
        run_step_worker(task)

        mock_engine.resume_run.assert_called_once_with("my_parent_run")

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_finalize_run_called(self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock) -> None:
        """Verify finalize_run is called to re-persist context.json."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task()
        run_step_worker(task)

        mock_engine.finalize_run.assert_called_once()

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_current_step_none_after_completion(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify current_step is cleared after the step completes."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task()
        run_step_worker(task)

        assert task.current_step is None


class TestStepWorkerFailure:
    """Tests for single-step execution failure via run_step_worker."""

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_status_error_on_exception(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify status is set to 'error' when the step raises."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}
        mock_engine.execute_step.side_effect = RuntimeError("GPU OOM")

        task = _make_step_task()
        run_step_worker(task)

        assert task.status == "error"

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_error_message_stored(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify the exception message is stored in task.error."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}
        mock_engine.execute_step.side_effect = ValueError("bad param")

        task = _make_step_task()
        run_step_worker(task)

        assert task.error == "bad param"

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_context_not_set_on_failure(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify task.context remains None when the step fails."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}
        mock_engine.execute_step.side_effect = RuntimeError("crash")

        task = _make_step_task()
        run_step_worker(task)

        assert task.context is None


class TestPipelineWorkerNotifier:
    """Tests for notifier wiring in run_pipeline_worker."""

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_messages_shared_with_task(self, mock_engine_cls: MagicMock) -> None:
        """Verify task.messages is the same list as the notifier's messages."""
        from utils.plugin_notifier import PluginNotifier

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        captured_notifier: list[PluginNotifier] = []

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            captured_notifier.append(kwargs["notifier"])
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task([{"plugin": "cat_a.impl.impl", "params": {}}])
        run_pipeline_worker(task)

        assert task.messages is captured_notifier[0].messages

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_notifier_passed_to_execute_step(self, mock_engine_cls: MagicMock) -> None:
        """Verify execute_step is called with a PluginNotifier keyword arg."""
        from utils.plugin_notifier import PluginNotifier

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task([{"plugin": "cat_a.impl.impl", "params": {}}])
        run_pipeline_worker(task)

        _, call_kwargs = mock_engine.execute_step.call_args
        assert isinstance(call_kwargs["notifier"], PluginNotifier)

    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_messages_visible_via_task_when_notifier_appends(
        self, mock_engine_cls: MagicMock
    ) -> None:
        """Verify messages appended by the notifier are visible via task.messages."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "p"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            kwargs["notifier"].info("Epoch 1")
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_task([{"plugin": "cat_a.impl.impl", "params": {}}])
        run_pipeline_worker(task)

        assert len(task.messages) == 1
        assert task.messages[0]["text"] == "Epoch 1"


class TestStepWorkerNotifier:
    """Tests for notifier wiring in run_step_worker."""

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_messages_shared_with_task(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify task.messages is the same list as the notifier's messages."""
        from utils.plugin_notifier import PluginNotifier

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}

        captured_notifier: list[PluginNotifier] = []

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            captured_notifier.append(kwargs["notifier"])
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task()
        run_step_worker(task)

        assert task.messages is captured_notifier[0].messages

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_notifier_passed_to_execute_step(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify execute_step is called with a PluginNotifier keyword arg."""
        from utils.plugin_notifier import PluginNotifier

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **_kwargs: Any
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task()
        run_step_worker(task)

        _, call_kwargs = mock_engine.execute_step.call_args
        assert isinstance(call_kwargs["notifier"], PluginNotifier)

    @patch("app.core.pipeline_worker.get_run_context")
    @patch("app.core.pipeline_worker.PipelineEngine")
    def test_messages_visible_via_task_when_notifier_appends(
        self, mock_engine_cls: MagicMock, mock_get_ctx: MagicMock
    ) -> None:
        """Verify messages appended by the notifier are visible via task.messages."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_get_ctx.return_value = {}

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            kwargs["notifier"].success("Step done")
            ctx[plugin.split(".")[0]] = {"run_id": "r"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        task = _make_step_task()
        run_step_worker(task)

        assert len(task.messages) == 1
        assert task.messages[0]["level"] == "success"

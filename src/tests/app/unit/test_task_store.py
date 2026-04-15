"""Unit tests for app.core.task_store."""

import threading

import pytest

from app.core.task_store import _tasks, cancel_task, create_task, get_task, task_to_response
from app.models.pipeline import TaskState


def _clear_store() -> None:
    """Remove all tasks from the store between tests."""
    _tasks.clear()


class TestCreateTask:
    """Tests for create_task."""

    def test_returns_task_state(self) -> None:
        """Verify create_task returns a TaskState instance."""
        _clear_store()
        task = create_task([{"plugin": "cat.impl.impl", "params": {}}])
        assert isinstance(task, TaskState)

    def test_generates_unique_ids(self) -> None:
        """Verify two tasks get different IDs."""
        _clear_store()
        t1 = create_task([])
        t2 = create_task([])
        assert t1.task_id != t2.task_id

    def test_status_is_running(self) -> None:
        """Verify new task has status 'running'."""
        _clear_store()
        task = create_task([])
        assert task.status == "running"

    def test_stores_steps_requested(self) -> None:
        """Verify steps_requested is stored on the task."""
        _clear_store()
        steps = [{"plugin": "a.b.c", "params": {"x": 1}}]
        task = create_task(steps)
        assert task.steps_requested == steps

    def test_task_is_stored(self) -> None:
        """Verify the task is retrievable from the internal dict."""
        _clear_store()
        task = create_task([])
        assert _tasks[task.task_id] is task

    def test_stores_tags(self) -> None:
        """Verify tags are forwarded to the TaskState."""
        _clear_store()
        task = create_task([], tags={"dataset": "MovieLens", "model": "ELSA"})
        assert task.tags == {"dataset": "MovieLens", "model": "ELSA"}

    def test_stores_description(self) -> None:
        """Verify description is forwarded to the TaskState."""
        _clear_store()
        task = create_task([], description="Baseline run")
        assert task.description == "Baseline run"

    def test_tags_default_to_empty_dict(self) -> None:
        """Verify tags default to an empty dict when not provided."""
        _clear_store()
        task = create_task([])
        assert task.tags == {}

    def test_description_defaults_to_empty_string(self) -> None:
        """Verify description defaults to empty string when not provided."""
        _clear_store()
        task = create_task([])
        assert task.description == ""

    def test_tags_none_becomes_empty_dict(self) -> None:
        """Verify passing None for tags results in an empty dict."""
        _clear_store()
        task = create_task([], tags=None)
        assert task.tags == {}

    def test_cancel_event_is_threading_event(self) -> None:
        """Verify cancel_event is a threading.Event instance."""
        _clear_store()
        task = create_task([])
        assert isinstance(task.cancel_event, threading.Event)

    def test_cancel_event_not_set_by_default(self) -> None:
        """Verify cancel_event is not set on a new task."""
        _clear_store()
        task = create_task([])
        assert not task.cancel_event.is_set()

    def test_cancel_event_unique_per_task(self) -> None:
        """Verify each task gets its own cancel_event instance."""
        _clear_store()
        t1 = create_task([])
        t2 = create_task([])
        assert t1.cancel_event is not t2.cancel_event


class TestGetTask:
    """Tests for get_task."""

    def test_returns_none_for_unknown_id(self) -> None:
        """Verify None is returned for a non-existent task ID."""
        _clear_store()
        assert get_task("does-not-exist") is None

    def test_returns_existing_task(self) -> None:
        """Verify the same TaskState object is returned."""
        _clear_store()
        task = create_task([])
        retrieved = get_task(task.task_id)
        assert retrieved is task


class TestCancelTask:
    """Tests for cancel_task."""

    def test_returns_none_for_unknown_id(self) -> None:
        """Verify None is returned for a non-existent task ID."""
        _clear_store()
        assert cancel_task("does-not-exist") is None

    def test_sets_cancel_event(self) -> None:
        """Verify cancel_event is set on a running task."""
        _clear_store()
        task = create_task([])
        result = cancel_task(task.task_id)
        assert result is task
        assert task.cancel_event.is_set()

    def test_raises_for_completed_task(self) -> None:
        """Verify ValueError when the task is already completed."""
        _clear_store()
        task = create_task([])
        task.status = "completed"

        with pytest.raises(ValueError, match="not cancellable"):
            cancel_task(task.task_id)

    def test_raises_for_errored_task(self) -> None:
        """Verify ValueError when the task has errored."""
        _clear_store()
        task = create_task([])
        task.status = "error"

        with pytest.raises(ValueError, match="not cancellable"):
            cancel_task(task.task_id)

    def test_raises_for_already_cancelled_task(self) -> None:
        """Verify ValueError when the task is already cancelled."""
        _clear_store()
        task = create_task([])
        task.status = "cancelled"

        with pytest.raises(ValueError, match="not cancellable"):
            cancel_task(task.task_id)


class TestTaskToResponse:
    """Tests for task_to_response."""

    def test_includes_total_steps(self) -> None:
        """Verify total_steps is computed from steps_requested."""
        _clear_store()
        task = create_task(
            [
                {"plugin": "a.b.c", "params": {}},
                {"plugin": "d.e.f", "params": {}},
                {"plugin": "g.h.i", "params": {}},
            ]
        )
        resp = task_to_response(task)
        assert resp.total_steps == 3

    def test_maps_all_fields(self) -> None:
        """Verify all TaskState fields appear in the response."""
        _clear_store()
        task = create_task([{"plugin": "cat.p.p", "params": {}}])
        task.run_id = "mlflow_run_1"
        task.current_step = "cat"
        task.current_step_index = 0
        task.completed_steps = [{"category": "cat", "run_id": "r1"}]
        task.context = {"cat": {"run_id": "r1"}}
        task.status = "completed"

        resp = task_to_response(task)
        assert resp.task_id == task.task_id
        assert resp.status == "completed"
        assert resp.run_id == "mlflow_run_1"
        assert resp.current_step == "cat"
        assert resp.current_step_index == 0
        assert resp.total_steps == 1
        assert resp.completed_steps == [{"category": "cat", "run_id": "r1"}]
        assert resp.context == {"cat": {"run_id": "r1"}}
        assert resp.error is None

    def test_error_state(self) -> None:
        """Verify error field is propagated."""
        _clear_store()
        task = create_task([])
        task.status = "error"
        task.error = "Something broke"

        resp = task_to_response(task)
        assert resp.status == "error"
        assert resp.error == "Something broke"

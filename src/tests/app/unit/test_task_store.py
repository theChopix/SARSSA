"""Unit tests for app.core.task_store."""

from app.core.task_store import _tasks, create_task, get_task, task_to_response
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

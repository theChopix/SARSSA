"""Unit tests for app.core.tasks.task_store."""

import threading

import pytest

from app.core.tasks.task_store import (
    _tasks,
    cancel_task,
    create_task,
    get_task,
    list_active_tasks,
    task_to_response,
    task_to_summary,
)
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

    def test_status_is_queued(self) -> None:
        """Verify a new task waits as 'queued' until the dispatcher picks it."""
        _clear_store()
        task = create_task([])
        assert task.status == "queued"

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

    def test_stores_pipeline_name(self) -> None:
        """Verify pipeline_name is forwarded to the TaskState."""
        _clear_store()
        task = create_task([], pipeline_name="Baseline ELSA")
        assert task.pipeline_name == "Baseline ELSA"

    def test_pipeline_name_defaults_to_empty_string(self) -> None:
        """Verify pipeline_name defaults to empty string when not provided."""
        _clear_store()
        task = create_task([])
        assert task.pipeline_name == ""

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

    def test_run_id_none_by_default(self) -> None:
        """Verify run_id is None when not provided."""
        _clear_store()
        task = create_task([])
        assert task.run_id is None

    def test_run_id_set_when_provided(self) -> None:
        """Verify run_id is pre-populated when passed to create_task."""
        _clear_store()
        task = create_task([], run_id="abc")
        assert task.run_id == "abc"


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

    def test_graceful_does_not_set_abort_event(self) -> None:
        """Verify the default (graceful) cancel leaves abort_event unset."""
        _clear_store()
        task = create_task([])
        cancel_task(task.task_id)
        assert task.cancel_event.is_set()
        assert not task.abort_event.is_set()

    def test_hard_sets_both_events(self) -> None:
        """Verify hard cancel ('Cancel now') sets both events."""
        _clear_store()
        task = create_task([])
        cancel_task(task.task_id, hard=True)
        assert task.cancel_event.is_set()
        assert task.abort_event.is_set()

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

    def test_queued_task_is_resolved_immediately(self) -> None:
        """Verify a queued task is marked cancelled right away (no worker owns it)."""
        _clear_store()
        task = create_task([])

        cancel_task(task.task_id)

        assert task.status == "cancelled"
        assert task.error == "Cancelled while queued."

    def test_running_task_keeps_status(self) -> None:
        """Verify a running task's status is left for its worker to resolve."""
        _clear_store()
        task = create_task([])
        task.status = "running"

        cancel_task(task.task_id)

        assert task.status == "running"
        assert task.cancel_event.is_set()


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

    def test_messages_empty_by_default(self) -> None:
        """Verify messages is an empty list when no notifications were emitted."""
        _clear_store()
        task = create_task([])

        resp = task_to_response(task)

        assert resp.messages == []

    def test_messages_included_in_response(self) -> None:
        """Verify messages on TaskState appear in the response."""
        _clear_store()
        task = create_task([])
        task.messages.append({"timestamp": 1.0, "level": "info", "text": "Epoch 1"})
        task.messages.append({"timestamp": 2.0, "level": "success", "text": "Done"})

        resp = task_to_response(task)

        assert len(resp.messages) == 2
        assert resp.messages[0]["text"] == "Epoch 1"
        assert resp.messages[1]["text"] == "Done"

    def test_messages_is_snapshot_not_same_reference(self) -> None:
        """Verify task_to_response returns a copy of messages, not the same list."""
        _clear_store()
        task = create_task([])
        task.messages.append({"timestamp": 1.0, "level": "info", "text": "msg"})

        resp = task_to_response(task)
        task.messages.append({"timestamp": 2.0, "level": "info", "text": "late"})

        assert len(resp.messages) == 1


class TestListActiveTasks:
    """Tests for list_active_tasks."""

    def test_returns_only_running_tasks(self) -> None:
        """Verify completed/errored tasks are excluded."""
        _clear_store()
        running = create_task([])
        done = create_task([])
        done.status = "completed"

        active = list_active_tasks()

        assert running in active
        assert done not in active

    def test_empty_when_none_running(self) -> None:
        """Verify an empty list when no task is running."""
        _clear_store()
        task = create_task([])
        task.status = "error"

        assert list_active_tasks() == []

    def test_newest_first(self) -> None:
        """Verify the most recently created task comes first."""
        _clear_store()
        first = create_task([])
        second = create_task([])

        assert list_active_tasks() == [second, first]

    def test_includes_queued_and_running(self) -> None:
        """Verify both queued and running tasks count as active."""
        _clear_store()
        queued = create_task([])
        running = create_task([])
        running.status = "running"

        active = list_active_tasks()

        assert queued in active
        assert running in active


class TestTaskToSummary:
    """Tests for task_to_summary."""

    def test_maps_fields_and_steps(self) -> None:
        """Verify all summary fields, including steps_requested."""
        _clear_store()
        steps = [
            {"plugin": "a.b.c", "params": {"x": 1}},
            {"plugin": "d.e.f", "params": {}},
        ]
        initial_context = {"dataset_loading": {"run_id": "upstream1"}}
        task = create_task(steps, initial_context=initial_context, pipeline_name="Baseline")
        task.run_id = "run1"
        task.current_step = "a"
        task.current_step_index = 1

        summary = task_to_summary(task)

        assert summary.task_id == task.task_id
        assert summary.run_id == "run1"
        assert summary.pipeline_name == "Baseline"
        assert summary.status == "queued"
        assert summary.current_step == "a"
        assert summary.current_step_index == 1
        assert summary.total_steps == 2
        assert summary.steps_requested == steps
        assert summary.initial_context == initial_context


class TestTaskStateMessages:
    """Tests for TaskState.messages field."""

    def test_messages_defaults_to_empty_list(self) -> None:
        """Verify new TaskState has an empty messages list."""
        _clear_store()
        task = create_task([])

        assert task.messages == []

    def test_messages_list_is_unique_per_task(self) -> None:
        """Verify each task gets its own messages list instance."""
        _clear_store()
        t1 = create_task([])
        t2 = create_task([])

        t1.messages.append({"timestamp": 1.0, "level": "info", "text": "x"})

        assert t2.messages == []


class TestTaskTimings:
    """Tests for the created_at / started_at / last_message fields."""

    def test_created_at_set_on_creation(self) -> None:
        """Verify a fresh task is stamped and not yet started."""
        _clear_store()
        task = create_task([{"plugin": "a.b.c", "params": {}}])

        assert task.created_at > 0
        assert task.started_at is None

    def test_summary_exposes_timings(self) -> None:
        """Verify all three stamps reach the summary the menu polls."""
        _clear_store()
        task = create_task([{"plugin": "a.b.c", "params": {}}])
        task.started_at = task.created_at + 5
        task.current_step_started_at = task.created_at + 8

        summary = task_to_summary(task)

        assert summary.created_at == task.created_at
        assert summary.started_at == task.created_at + 5
        assert summary.current_step_started_at == task.created_at + 8

    def test_summary_last_message_is_latest_only(self) -> None:
        """Verify the summary carries the newest message, not the list."""
        _clear_store()
        task = create_task([{"plugin": "a.b.c", "params": {}}])
        task.messages = [
            {"timestamp": 1.0, "level": "info", "text": "first"},
            {"timestamp": 2.0, "level": "progress", "text": "Epoch 2/10"},
        ]

        summary = task_to_summary(task)

        assert summary.last_message == task.messages[-1]

    def test_summary_last_message_none_when_silent(self) -> None:
        """Verify a plugin that never notifies yields None, not an error."""
        _clear_store()
        task = create_task([{"plugin": "a.b.c", "params": {}}])

        assert task_to_summary(task).last_message is None

    def test_response_exposes_timings(self) -> None:
        """Verify the detailed status response carries the stamps too."""
        _clear_store()
        task = create_task([{"plugin": "a.b.c", "params": {}}])
        task.started_at = task.created_at + 1

        response = task_to_response(task)

        assert response.created_at == task.created_at
        assert response.started_at == task.created_at + 1

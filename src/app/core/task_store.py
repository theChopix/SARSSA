"""In-memory task store for background pipeline executions.

Provides a simple dict-backed store where each task is a
:class:`~app.models.pipeline.TaskState` object.  The background worker
thread mutates the ``TaskState`` in-place; the ``GET /tasks/{id}``
endpoint reads it via :func:`task_to_response`.
"""

import uuid
from typing import Any

from app.models.pipeline import TaskState, TaskStatusResponse, TaskSummary

_tasks: dict[str, TaskState] = {}


def create_task(
    steps: list[dict[str, Any]],
    initial_context: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    description: str = "",
    pipeline_name: str = "",
    run_id: str | None = None,
) -> TaskState:
    """Create a new task, store it, and return it.

    Args:
        steps: Serialised step dicts (each has ``plugin`` and ``params``).
        initial_context: Optional pre-populated context from a previous run.
        tags: User-provided key-value tags for the pipeline run.
        description: User-provided free-text description.
        pipeline_name: Optional user-provided label woven into the
            parent run name.
        run_id: Optional pre-existing MLflow parent run ID.  When provided,
            ``task.run_id`` is set immediately so the worker can resume an
            existing run rather than starting a fresh one.

    Returns:
        TaskState: The newly created task with status ``"running"``.
    """
    task_id = uuid.uuid4().hex
    task = TaskState(
        task_id=task_id,
        steps_requested=steps,
        initial_context=initial_context or {},
        tags=tags or {},
        description=description,
        pipeline_name=pipeline_name,
    )
    if run_id is not None:
        task.run_id = run_id
    _tasks[task_id] = task
    return task


def get_task(task_id: str) -> TaskState | None:
    """Look up a task by its ID.

    Args:
        task_id: Unique task identifier.

    Returns:
        TaskState | None: The task, or ``None`` if not found.
    """
    return _tasks.get(task_id)


def cancel_task(task_id: str, hard: bool = False) -> TaskState | None:
    """Request cancellation of a task by setting its cancel event(s).

    Args:
        task_id: Unique task identifier.
        hard: When ``True`` ("Cancel now"), also set ``abort_event`` so a
            cooperating plugin stops the currently executing step; the
            graceful ``cancel_event`` is set either way, so no further
            steps start.

    Returns:
        TaskState | None: The task if found and cancellation was
            requested, ``None`` if not found.

    Raises:
        ValueError: If the task is not in a cancellable state.
    """
    task = _tasks.get(task_id)
    if task is None:
        return None
    if task.status != "running":
        raise ValueError(f"Task '{task_id}' is '{task.status}', not cancellable.")
    task.cancel_event.set()
    if hard:
        task.abort_event.set()
    return task


def list_active_tasks() -> list[TaskState]:
    """Return all tasks still in the ``"running"`` state, newest first.

    Returns:
        list[TaskState]: Running tasks, most recently created first.
    """
    running = [task for task in _tasks.values() if task.status == "running"]
    running.reverse()
    return running


def task_to_summary(task: TaskState) -> TaskSummary:
    """Convert a TaskState to a compact summary of an active task.

    Args:
        task: The task to serialise.

    Returns:
        TaskSummary: Pydantic model ready for JSON serialisation.
    """
    return TaskSummary(
        task_id=task.task_id,
        run_id=task.run_id,
        pipeline_name=task.pipeline_name,
        status=task.status,
        current_step=task.current_step,
        current_step_index=task.current_step_index,
        total_steps=len(task.steps_requested),
        steps_requested=task.steps_requested,
        initial_context=task.initial_context,
    )


def task_to_response(task: TaskState) -> TaskStatusResponse:
    """Convert a TaskState to a serialisable API response.

    Adds the computed ``total_steps`` field derived from
    ``len(task.steps_requested)``.

    Args:
        task: The task to serialise.

    Returns:
        TaskStatusResponse: Pydantic model ready for JSON serialisation.
    """
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        run_id=task.run_id,
        current_step=task.current_step,
        current_step_index=task.current_step_index,
        total_steps=len(task.steps_requested),
        completed_steps=task.completed_steps,
        context=task.context,
        error=task.error,
        messages=list(task.messages),
    )

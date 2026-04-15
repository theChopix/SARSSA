"""In-memory task store for background pipeline executions.

Provides a simple dict-backed store where each task is a
:class:`~app.models.pipeline.TaskState` object.  The background worker
thread mutates the ``TaskState`` in-place; the ``GET /tasks/{id}``
endpoint reads it via :func:`task_to_response`.
"""

import uuid
from typing import Any

from app.models.pipeline import TaskState, TaskStatusResponse

_tasks: dict[str, TaskState] = {}


def create_task(
    steps: list[dict[str, Any]],
    initial_context: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    description: str = "",
) -> TaskState:
    """Create a new task, store it, and return it.

    Args:
        steps: Serialised step dicts (each has ``plugin`` and ``params``).
        initial_context: Optional pre-populated context from a previous run.
        tags: User-provided key-value tags for the pipeline run.
        description: User-provided free-text description.

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
    )
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


def cancel_task(task_id: str) -> TaskState | None:
    """Request cancellation of a task by setting its cancel event.

    Args:
        task_id: Unique task identifier.

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
    return task


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
    )

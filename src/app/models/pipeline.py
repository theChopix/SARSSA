"""Pydantic models and dataclasses for pipeline execution."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class StepDefinition(BaseModel):
    """A single plugin step to execute."""

    plugin: str
    params: dict[str, Any] | None = {}


class PipelineRequest(BaseModel):
    """Request body for pipeline execution endpoints."""

    steps: list[StepDefinition]


# ── Task state (in-memory, mutated by background worker) ──


@dataclass
class TaskState:
    """Mutable state of a background pipeline task.

    Created by the task store when ``POST /run-async`` is called.
    The background worker thread mutates this object in-place as
    it progresses through pipeline steps.  The ``GET /tasks/{id}``
    endpoint reads it to return the current status.

    Attributes:
        task_id: Unique identifier for this task.
        status: One of ``"running"``, ``"completed"``, ``"error"``.
        run_id: MLflow parent run ID (set after ``engine.start_run()``).
        steps_requested: The original step dicts submitted by the user.
        current_step: Category key of the step currently executing.
        current_step_index: 0-based index into *steps_requested*.
        completed_steps: Steps that have finished so far.
        context: Final pipeline context (set on completion).
        error: Error message (set on failure).
    """

    task_id: str
    status: str = "running"
    run_id: str | None = None
    steps_requested: list[dict[str, Any]] = field(default_factory=list)
    current_step: str | None = None
    current_step_index: int = 0
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] | None = None
    error: str | None = None


# ── API response model ────────────────────────────────────


class TaskStatusResponse(BaseModel):
    """Serialisation model returned by ``GET /tasks/{task_id}``.

    This is a read-only Pydantic model built from a :class:`TaskState`.
    It adds the computed ``total_steps`` field so the frontend can
    display progress like "Step 2 / 4".
    """

    task_id: str
    status: str
    run_id: str | None = None
    current_step: str | None = None
    current_step_index: int = 0
    total_steps: int = 0
    completed_steps: list[dict[str, Any]] = []
    context: dict[str, Any] | None = None
    error: str | None = None

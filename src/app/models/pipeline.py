"""Pydantic models and dataclasses for pipeline execution."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class StepDefinition(BaseModel):
    """A single plugin step to execute."""

    plugin: str
    params: dict[str, Any] | None = {}


class PipelineRequest(BaseModel):
    """Request body for pipeline execution endpoints."""

    steps: list[StepDefinition]
    context: dict[str, Any] = {}
    tags: dict[str, str] = {}
    description: str = ""
    pipeline_name: str = Field(default="", max_length=60)
    experiment_name: str = ""


class ExperimentCreateRequest(BaseModel):
    """Request body for creating an MLflow experiment."""

    name: str = Field(max_length=100)


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
        status: One of ``"queued"``, ``"running"``, ``"completed"``,
            ``"error"``, ``"cancelled"``.
        run_id: MLflow parent run ID (set after ``engine.start_run()``).
        steps_requested: The original step dicts submitted by the user.
        initial_context: Pre-populated context from a previous run (empty for fresh runs).
        tags: User-provided key-value tags for the pipeline run.
        description: User-provided free-text description.
        pipeline_name: Optional user-provided label woven into the
            parent run name.
        experiment_name: MLflow experiment the run logs to (empty =
            shared experiment).
        current_step: Category key of the step currently executing.
        current_step_index: 0-based index into *steps_requested*.
        completed_steps: Steps that have finished so far.
        context: Final pipeline context (set on completion).
        error: Error message (set on failure).
        cancel_event: Thread-safe flag for graceful cancellation
            (stop before the next step).
        abort_event: Thread-safe flag for immediate ("Cancel now")
            cancellation — observed by a cooperating plugin to stop the
            currently executing step.
        messages: Ordered list of notification dicts pushed by the
            executing plugin via ``PluginNotifier``.  Shared with the
            notifier's own list (same object) so new entries are
            visible to the polling endpoint immediately.
        created_at: Unix epoch seconds when the task was submitted.
        started_at: Unix epoch seconds when the worker picked it up,
            ``None`` while still queued.  Kept separate from
            *created_at* so the UI can tell "queued for 20 min" from
            "running for 20 min".
        current_step_started_at: Unix epoch seconds when the step named
            by *current_step* began, ``None`` between steps.
    """

    task_id: str
    status: str = "running"
    cancel_event: threading.Event = field(default_factory=threading.Event)
    abort_event: threading.Event = field(default_factory=threading.Event)
    run_id: str | None = None
    steps_requested: list[dict[str, Any]] = field(default_factory=list)
    initial_context: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    description: str = ""
    pipeline_name: str = ""
    experiment_name: str = ""
    current_step: str | None = None
    current_step_index: int = 0
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] | None = None
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    current_step_started_at: float | None = None


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
    messages: list[dict[str, Any]] = []
    created_at: float = 0.0
    started_at: float | None = None
    current_step_started_at: float | None = None


class TaskSummary(BaseModel):
    """Compact view of an active task, returned by ``GET /tasks``.

    Carries enough to render a progress row and to rebuild the pipeline
    layout when the task is loaded — hence ``steps_requested`` (the
    newly-run steps) plus ``initial_context`` (the upstream steps that
    were loaded from a previous run and are not in ``steps_requested``).

    Carries only the *latest* notification, not the whole list: this
    endpoint is polled for every active task, so the payload must stay
    small.  Consumers needing the full history use
    :class:`TaskStatusResponse`.
    """

    task_id: str
    run_id: str | None = None
    pipeline_name: str = ""
    experiment_name: str = ""
    status: str
    current_step: str | None = None
    current_step_index: int = 0
    total_steps: int = 0
    steps_requested: list[dict[str, Any]] = []
    initial_context: dict[str, Any] = {}
    created_at: float = 0.0
    started_at: float | None = None
    current_step_started_at: float | None = None
    last_message: dict[str, Any] | None = None

"""API routes for pipeline execution and run management."""

from typing import Any, Literal

import mlflow
from fastapi import APIRouter, HTTPException, Query

from app.config.config import MLFLOW_UI_BASE_URL, SHARED_EXPERIMENT_NAME
from app.core.pipeline_engine import PipelineEngine
from app.core.pipeline_runs import (
    get_eligible_pipeline_runs,
    get_pipeline_runs,
    get_run_context,
)
from app.core.pipeline_worker import run_pipeline_worker, run_step_worker
from app.core.tasks.task_queue import submit
from app.core.tasks.task_store import (
    cancel_task,
    create_task,
    get_task,
    list_active_tasks,
    task_to_response,
    task_to_summary,
)
from app.models.pipeline import (
    ExperimentCreateRequest,
    PipelineRequest,
    StepDefinition,
    TaskStatusResponse,
    TaskSummary,
)

router = APIRouter()


@router.get("/mlflow-info")
def get_mlflow_info(experiment: str = "") -> dict[str, str]:
    """Return MLflow UI connection info for frontend deep links.

    Resolves the selected (or shared) experiment name to its numeric
    MLflow ID and pairs it with the UI base URL from config.

    Args:
        experiment: Optional user-selected experiment name; empty
            means the shared experiment.

    Returns:
        dict[str, str]: Contains ``ui_base_url`` and ``experiment_id``.

    Raises:
        HTTPException: 404 if the requested experiment is not found.
    """
    name = experiment or SHARED_EXPERIMENT_NAME
    mlflow_experiment = mlflow.get_experiment_by_name(name)
    if mlflow_experiment is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{name}' not found.",
        )
    return {
        "ui_base_url": MLFLOW_UI_BASE_URL,
        "experiment_id": mlflow_experiment.experiment_id,
    }


@router.get("/experiments")
def list_experiments() -> list[dict[str, Any]]:
    """Return active MLflow experiments selectable in the UI.

    MLflow's catch-all ``Default`` experiment is excluded; the shared
    experiment is flagged and listed first, the rest alphabetically.

    Returns:
        list[dict[str, Any]]: Each dict has ``name``,
            ``experiment_id`` and ``shared``.
    """
    experiments = [
        {
            "name": experiment.name,
            "experiment_id": experiment.experiment_id,
            "shared": experiment.name == SHARED_EXPERIMENT_NAME,
        }
        for experiment in mlflow.search_experiments()
        if experiment.name != "Default"
    ]
    return sorted(experiments, key=lambda e: (not e["shared"], e["name"]))


@router.post("/experiments")
def create_experiment(request: ExperimentCreateRequest) -> dict[str, Any]:
    """Create an MLflow experiment and return it.

    Args:
        request: Contains the experiment ``name``.

    Returns:
        dict[str, Any]: ``name``, ``experiment_id`` and ``shared``
            of the (possibly pre-existing) experiment.

    Raises:
        HTTPException: 422 if the name is empty or ``Default``.
    """
    name = request.name.strip()
    if not name or name == "Default":
        raise HTTPException(status_code=422, detail="Invalid experiment name.")
    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(name)
    return {
        "name": name,
        "experiment_id": experiment_id,
        "shared": name == SHARED_EXPERIMENT_NAME,
    }


@router.get("/runs")
def list_runs(
    required_steps: list[str] | None = Query(
        default=None,
        description=(
            "Filter to runs whose context.json contains every listed step. "
            "Pass repeated query params (?required_steps=a&required_steps=b)."
        ),
    ),
    experiment: str = "",
) -> list[dict[str, Any]]:
    """Return top-level pipeline runs, newest first.

    When *required_steps* is provided, only runs whose
    ``context.json`` contains every listed step key are returned —
    used by the compare-plugin past-runs dropdown to surface only
    runs that already completed the prerequisite stages.

    Args:
        required_steps: Optional list of step keys that must be
            present in each returned run's context.
        experiment: Optional user-selected experiment searched in
            addition to the shared one.

    Returns:
        list[dict[str, Any]]: Each dict has ``run_id``,
            ``run_name``, ``status``, ``start_time``, ``shared``.

    Raises:
        HTTPException: 404 if the requested experiment is not found.
    """
    try:
        if required_steps:
            return get_eligible_pipeline_runs(required_steps, experiment)
        return get_pipeline_runs(experiment)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/context")
def get_context(run_id: str) -> dict[str, Any]:
    """Return the context.json artifact from a pipeline run.

    Args:
        run_id: MLflow run ID of the parent pipeline run.

    Returns:
        dict[str, Any]: The context dictionary.

    Raises:
        HTTPException: 404 if the context artifact is not found.
    """
    try:
        return get_run_context(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/run-async")
def run_pipeline_async(pipeline_request: PipelineRequest) -> dict[str, str]:
    """Queue a pipeline for execution and return the task ID.

    Compute tasks run one at a time; the task waits as ``"queued"`` until
    the queue reaches it. The caller should poll ``GET /tasks/{task_id}``
    to track progress.

    Args:
        pipeline_request: Steps to execute.

    Returns:
        dict[str, str]: ``{"task_id": "..."}``.
    """
    steps = [step.model_dump() for step in pipeline_request.steps]
    task = create_task(
        steps,
        initial_context=pipeline_request.context,
        tags=pipeline_request.tags,
        description=pipeline_request.description,
        pipeline_name=pipeline_request.pipeline_name,
        experiment_name=pipeline_request.experiment_name,
    )

    submit(task, run_pipeline_worker)

    return {"task_id": task.task_id}


@router.get("/tasks")
def list_running_tasks() -> list[TaskSummary]:
    """Return all currently running pipeline tasks, newest first.

    Only tasks still in the ``"running"`` state are returned;
    completed, failed, and cancelled tasks are omitted.

    Returns:
        list[TaskSummary]: Compact summaries of the active tasks.
    """
    return [task_to_summary(task) for task in list_active_tasks()]


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Return the current status of a background pipeline task.

    Args:
        task_id: Unique task identifier returned by ``POST /run-async``.

    Returns:
        TaskStatusResponse: Current task state including progress info.

    Raises:
        HTTPException: 404 if the task ID is not found.
    """
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return task_to_response(task)


@router.post("/tasks/{task_id}/cancel")
def cancel_task_endpoint(
    task_id: str,
    mode: Literal["graceful", "now"] = "graceful",
) -> dict[str, str]:
    """Request cancellation of a running pipeline task.

    Two modes:

    - ``graceful`` (default) — the currently executing step runs to
      completion, then the pipeline stops before the next step.
    - ``now`` — additionally signals a cooperating plugin to abort the
      currently executing step at its next safe checkpoint. A plugin
      that does not observe the signal falls back to graceful behaviour.

    Either way the task status transitions to ``"cancelled"`` once the
    worker acknowledges the request.

    Args:
        task_id: Unique task identifier.
        mode: ``"graceful"`` or ``"now"`` (see above).

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: 404 if the task is not found.
        HTTPException: 409 if the task is not in a cancellable state.
    """
    try:
        task = cancel_task(task_id, hard=(mode == "now"))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )

    message = (
        "Abort requested. The current step will stop at its next safe checkpoint."
        if mode == "now"
        else "Cancellation requested. The current step will finish before the pipeline stops."
    )
    return {"message": message}


@router.post("/runs/{run_id}/execute-step")
def execute_step(run_id: str, step: StepDefinition) -> dict[str, Any]:
    """Execute a single plugin step on an existing pipeline run.

    Used for phase-2 (multi-run) plugins where the user triggers
    individual steps from the UI on a completed pipeline run.

    Blocks until the step completes, then re-persists ``context.json``
    on the parent run so the new step is visible to downstream steps.
    This is the synchronous equivalent of
    ``POST /runs/{run_id}/execute-step-async``; both leave the parent
    run's context in the same state.

    Args:
        run_id: MLflow run ID of the parent pipeline run.
        step: Plugin name and parameters to execute.

    Returns:
        dict[str, Any]: ``{"category": "...", "step_run_id": "..."}``.
    """
    context = get_run_context(run_id)

    engine = PipelineEngine()
    engine.resume_run(run_id)
    engine.execute_step(step.plugin, step.params or {}, context)
    engine.finalize_run(context)

    category = step.plugin.split(".")[0]
    return {
        "category": category,
        "step_run_id": context[category]["run_id"],
    }


@router.post("/runs/{run_id}/execute-step-async")
def execute_step_async(run_id: str, step: StepDefinition) -> dict[str, str]:
    """Queue a single plugin step for execution and return the task ID.

    The queued worker resumes the existing pipeline run identified by
    *run_id*, executes the requested step, and re-persists
    ``context.json``.  Compute tasks run one at a time, so the step may
    wait as ``"queued"`` behind an in-flight pipeline.  The caller
    should poll ``GET /tasks/{task_id}`` every 2 seconds until the task
    reaches a terminal state.

    The synchronous ``POST /runs/{run_id}/execute-step`` endpoint is
    kept as-is for scripting and testing.

    Args:
        run_id: MLflow run ID of the parent pipeline run to resume.
        step: Plugin name and parameters to execute.

    Returns:
        dict[str, str]: ``{"task_id": "..."}``.  Poll
            ``GET /tasks/{task_id}`` for progress.
    """
    task = create_task(steps=[step.model_dump()], run_id=run_id)

    submit(task, run_step_worker)

    return {"task_id": task.task_id}


@router.post("/run")
def run_pipeline(pipeline_request: PipelineRequest) -> dict[str, Any]:
    """Execute all pipeline steps synchronously (legacy endpoint).

    Args:
        pipeline_request: Steps and optional initial context.

    Returns:
        dict[str, Any]: Completion message and final context.
    """
    steps = [step.model_dump() for step in pipeline_request.steps]

    engine = PipelineEngine(steps)
    result = engine.run(dict(pipeline_request.context))

    return {"message": "Pipeline finished", "result": result}

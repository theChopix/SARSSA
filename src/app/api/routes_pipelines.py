"""API routes for pipeline execution and run management."""

import threading
from typing import Any

import mlflow
from fastapi import APIRouter, HTTPException

from app.config.config import EXPERIMENT_NAME, MLFLOW_UI_BASE_URL
from app.core.pipeline_engine import PipelineEngine
from app.core.pipeline_runs import get_pipeline_runs, get_run_context
from app.core.pipeline_worker import run_pipeline_worker
from app.core.task_store import create_task, get_task, task_to_response
from app.models.pipeline import PipelineRequest, StepDefinition, TaskStatusResponse

router = APIRouter()


@router.get("/mlflow-info")
def get_mlflow_info() -> dict[str, str]:
    """Return MLflow UI connection info for frontend deep links.

    Resolves the configured experiment name to its numeric MLflow ID
    and pairs it with the UI base URL from config.

    Returns:
        dict[str, str]: Contains ``ui_base_url`` and ``experiment_id``.

    Raises:
        HTTPException: 500 if the configured experiment is not found.
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise HTTPException(
            status_code=500,
            detail=f"Experiment '{EXPERIMENT_NAME}' not found.",
        )
    return {
        "ui_base_url": MLFLOW_UI_BASE_URL,
        "experiment_id": experiment.experiment_id,
    }


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    """Return all top-level pipeline runs, newest first.

    Returns:
        list[dict[str, Any]]: Each dict has ``run_id``,
            ``run_name``, ``status``, ``start_time``.
    """
    return get_pipeline_runs()


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
    """Start a pipeline in a background thread and return the task ID.

    The caller should poll ``GET /tasks/{task_id}`` to track progress.

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
    )

    thread = threading.Thread(
        target=run_pipeline_worker,
        args=(task,),
        daemon=True,
    )
    thread.start()

    return {"task_id": task.task_id}


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


@router.post("/runs/{run_id}/execute-step")
def execute_step(run_id: str, step: StepDefinition) -> dict[str, Any]:
    """Execute a single plugin step on an existing pipeline run.

    Used for phase-2 (multi-run) plugins where the user triggers
    individual steps from the UI on a completed pipeline run.

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

    category = step.plugin.split(".")[0]
    return {
        "category": category,
        "step_run_id": context[category]["run_id"],
    }


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

"""API routes for pipeline execution and run management."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.pipeline_engine import PipelineEngine
from app.core.pipeline_runs import get_pipeline_runs, get_run_context
from app.models.pipeline import PipelineRequest, StepDefinition

router = APIRouter()


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


@router.post("/run-stream")
async def run_pipeline_stream(
    pipeline_request: PipelineRequest,
) -> EventSourceResponse:
    """Execute a pipeline with SSE progress events.

    Emits the following event types:

    - ``run_started``: ``{"run_id": "..."}``
    - ``step_started``: ``{"category": "...", "plugin": "..."}``
    - ``step_completed``: ``{"category": "...", "run_id": "..."}``
    - ``run_completed``: ``{"run_id": "...", "context": {...}}``

    Args:
        pipeline_request: Steps to execute.

    Returns:
        EventSourceResponse: SSE stream of pipeline progress.
    """
    engine = PipelineEngine()

    async def event_generator() -> Any:
        run_id = await asyncio.to_thread(engine.start_run)
        yield {
            "event": "run_started",
            "data": json.dumps({"run_id": run_id}),
        }

        context: dict[str, Any] = {}

        for step in pipeline_request.steps:
            category = step.plugin.split(".")[0]
            yield {
                "event": "step_started",
                "data": json.dumps({"category": category, "plugin": step.plugin}),
            }

            await asyncio.to_thread(
                engine.execute_step,
                step.plugin,
                step.params or {},
                context,
            )

            yield {
                "event": "step_completed",
                "data": json.dumps(
                    {
                        "category": category,
                        "run_id": context[category]["run_id"],
                    }
                ),
            }

        await asyncio.to_thread(engine.finalize_run, context)
        yield {
            "event": "run_completed",
            "data": json.dumps({"run_id": run_id, "context": context}),
        }

    return EventSourceResponse(event_generator())


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
def run_pipeline(context: dict[str, Any], pipeline_request: PipelineRequest) -> dict[str, Any]:
    """Execute all pipeline steps synchronously (legacy endpoint).

    Args:
        context: Initial pipeline context.
        pipeline_request: Steps to execute.

    Returns:
        dict[str, Any]: Completion message and final context.
    """
    steps = [step.model_dump() for step in pipeline_request.steps]

    engine = PipelineEngine(steps)
    result = engine.run(context)

    return {"message": "Pipeline finished", "result": result}

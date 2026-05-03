"""Functions for querying past pipeline runs from MLflow."""

import json
from typing import Any, cast

import mlflow
from mlflow.entities import Run

from app.config.config import EXPERIMENT_NAME


def get_pipeline_runs() -> list[dict[str, Any]]:
    """List all top-level pipeline runs from the tracking experiment.

    Returns:
        list[dict[str, Any]]: Each dict contains ``run_id``,
            ``run_name``, ``status``, and ``start_time`` (ISO string).
            Results are ordered newest-first.

    Raises:
        ValueError: If the MLflow experiment does not exist.
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{EXPERIMENT_NAME}' not found.")

    all_runs = cast(
        list[Run],
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            output_format="list",
            order_by=["start_time DESC"],
        ),
    )

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
        }
        for run in all_runs
        if "mlflow.parentRunId" not in run.data.tags
    ]


def get_eligible_pipeline_runs(
    required_steps: list[str],
) -> list[dict[str, Any]]:
    """List past pipeline runs that contain all *required_steps*.

    A run is eligible when its ``context.json`` artifact maps every
    entry in *required_steps* to a value (typically a per-step run
    id).  Runs whose ``context.json`` is missing or unreadable are
    silently dropped — they cannot be used as a compare source.

    Args:
        required_steps: Step keys that must be present in the
            past run's ``context.json``.  An empty list returns
            every top-level pipeline run.

    Returns:
        list[dict[str, Any]]: Subset of :func:`get_pipeline_runs`
            whose context contains all *required_steps*, preserving
            the newest-first ordering.
    """
    all_runs = get_pipeline_runs()
    if not required_steps:
        return all_runs

    eligible: list[dict[str, Any]] = []
    for run in all_runs:
        try:
            context = get_run_context(run["run_id"])
        except FileNotFoundError:
            continue
        if all(step in context for step in required_steps):
            eligible.append(run)
    return eligible


def get_run_context(run_id: str) -> dict[str, Any]:
    """Load the ``context.json`` artifact from a pipeline parent run.

    Args:
        run_id: MLflow run ID of the parent pipeline run.

    Returns:
        dict[str, Any]: The context dictionary containing per-category
            ``run_id`` references from the pipeline execution.

    Raises:
        FileNotFoundError: If ``context.json`` is not found in the
            run's artifacts.
    """
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="context.json")

    with open(artifact_path) as f:
        context: dict[str, Any] = json.load(f)

    return context

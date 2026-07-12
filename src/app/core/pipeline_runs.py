"""Functions for querying past pipeline runs from MLflow."""

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import mlflow
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from app.config.config import EXPERIMENT_NAME, MLFLOW_UI_BASE_URL, PLUGIN_CATEGORIES


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

    def has_required_steps(run: dict[str, Any]) -> bool:
        try:
            context = get_run_context(run["run_id"])
        except FileNotFoundError:
            return False
        return all(step in context for step in required_steps)

    # fetch concurrently, pool.map preserves ordering
    with ThreadPoolExecutor(max_workers=16) as pool:
        keep = list(pool.map(has_required_steps, all_runs))
    return [run for run, ok in zip(all_runs, keep, strict=True) if ok]


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
    # Existence check via the cheap metadata listing first
    try:
        files = MlflowClient().list_artifacts(run_id)
    except MlflowException as exc:
        raise FileNotFoundError(f"context.json not found for run {run_id}") from exc
    if all(f.path != "context.json" for f in files):
        raise FileNotFoundError(f"context.json not found for run {run_id}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="context.json", dst_path=tmp_dir
            )
        except MlflowException as exc:
            # MLflow raises MlflowException (not FileNotFoundError) when the artifact
            # is absent — e.g. an orphaned run with no artifacts. Normalise to the
            # documented FileNotFoundError that every caller already handles.
            raise FileNotFoundError(f"context.json not found for run {run_id}") from exc

        with open(artifact_path) as f:
            context: dict[str, Any] = json.load(f)

    return context


def build_provenance_note(initial_context: dict[str, Any]) -> str:
    """Build a Markdown note linking a derived run to its upstream sources.

    Resolves each inherited step in *initial_context* to its source parent run
    and returns Markdown with clickable links grouped by source, or ``""`` when
    nothing was inherited or resolvable.

    Args:
        initial_context: Inherited context — ``{category: {"run_id": ...}}``.

    Returns:
        str: A Markdown note, or ``""`` if there is no provenance to record.
    """
    if not initial_context:
        return ""

    client = MlflowClient()
    # Group inherited categories by their source parent run, in encounter order.
    sources: dict[str, dict[str, Any]] = {}
    for category, entry in initial_context.items():
        child_run_id = entry.get("run_id") if isinstance(entry, dict) else None
        if not child_run_id:
            continue
        try:
            child = client.get_run(child_run_id)
        except MlflowException:
            continue
        parent_id = child.data.tags.get("mlflow.parentRunId")
        if not parent_id:
            continue
        if parent_id not in sources:
            try:
                parent = client.get_run(parent_id)
            except MlflowException:
                continue
            sources[parent_id] = {"run": parent, "labels": []}
        cat = PLUGIN_CATEGORIES.get(category)
        sources[parent_id]["labels"].append(cat.display_name if cat is not None else category)

    if not sources:
        return ""

    lines = ["**Inherited upstream from:**"]
    for parent_id, info in sources.items():
        parent: Run = info["run"]
        name = parent.data.tags.get("mlflow.runName") or parent.info.run_name or parent_id
        url = f"{MLFLOW_UI_BASE_URL}/#/experiments/{parent.info.experiment_id}/runs/{parent_id}"
        lines.append(f"- [{name}]({url}) — {', '.join(info['labels'])}")
    return "\n".join(lines)

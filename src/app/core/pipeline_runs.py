from typing import Any

import mlflow
from mlflow.entities import ViewType

from utils.mlflow_manager import MLflowRunLoader

EXPERIMENT_NAME = "pipeline_experiments_01"


def get_pipeline_runs() -> list[dict[str, Any]]:
    """Query MLflow for all completed parent pipeline runs.

    Returns a list of runs sorted by start time (newest first), each with:
    - run_id, run_name, timestamp, status
    - context: the pipeline context dict (category -> run_id mapping)
    - steps: list of nested child runs with plugin name and category
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []

    # Fetch all runs and filter to parent-only (those without mlflow.parentRunId tag)
    all_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],
        output_format="list",
    )
    parent_runs = [r for r in all_runs if "mlflow.parentRunId" not in r.data.tags]

    results = []
    for run in parent_runs:
        run_info = run.info
        loader = MLflowRunLoader(run_info.run_id)

        # Load the context.json artifact if it exists
        context = None
        if loader.artifact_exists("context.json"):
            context = loader.get_json_artifact("context.json")

        # Fetch child (nested) runs to list the steps that were executed
        child_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_info.run_id}'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["start_time ASC"],
            output_format="list",
        )

        steps = []
        for child in child_runs:
            plugin_name = child.info.run_name
            category = plugin_name.split(".")[0] if plugin_name else None
            steps.append(
                {
                    "run_id": child.info.run_id,
                    "plugin_name": plugin_name,
                    "category": category,
                    "status": child.info.status,
                }
            )

        results.append(
            {
                "run_id": run_info.run_id,
                "run_name": run_info.run_name,
                "start_time": run_info.start_time,
                "end_time": run_info.end_time,
                "status": run_info.status,
                "context": context,
                "steps": steps,
            }
        )

    return results

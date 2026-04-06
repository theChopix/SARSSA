import datetime
import json
from collections.abc import Generator

import mlflow

from app.core.plugin_manager import PluginManager
from utils.mlflow_manager import MLflowRunLoader

EXPERIMENT_NAME = "pipeline_experiments_01"


class PipelineEngine:
    def __init__(self, steps):
        self.steps = steps

    def run(self, context):
        mlflow.set_experiment(EXPERIMENT_NAME)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with mlflow.start_run(run_name=f"pipeline_run_{timestamp}") as _:
            for step in self.steps:
                plugin_name = step["plugin"]
                params = step.get("params", {})
                plugin = PluginManager.load(plugin_name)

                with mlflow.start_run(run_name=f"{plugin_name}", nested=True) as pipeline_step:
                    plugin.run(context, **params)

                    plugin_category = plugin_name.split(".")[0]
                    context[plugin_category] = {"run_id": pipeline_step.info.run_id}

            mlflow.log_dict(context, "context.json")

        return context

    def run_streaming(self, context) -> Generator[str, None, None]:
        """Run pipeline and yield SSE events for each step."""
        mlflow.set_experiment(EXPERIMENT_NAME)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            with mlflow.start_run(run_name=f"pipeline_run_{timestamp}") as _:
                for i, step in enumerate(self.steps):
                    plugin_name = step["plugin"]
                    plugin_category = plugin_name.split(".")[0]
                    params = step.get("params", {})

                    yield _sse_event(
                        "step_start",
                        {
                            "step_index": i,
                            "plugin": plugin_name,
                            "category": plugin_category,
                        },
                    )

                    try:
                        plugin = PluginManager.load(plugin_name)
                        with mlflow.start_run(
                            run_name=f"{plugin_name}", nested=True
                        ) as pipeline_step:
                            plugin.run(context, **params)
                            context[plugin_category] = {"run_id": pipeline_step.info.run_id}

                        yield _sse_event(
                            "step_done",
                            {
                                "step_index": i,
                                "plugin": plugin_name,
                                "category": plugin_category,
                            },
                        )
                    except Exception as e:
                        yield _sse_event(
                            "step_error",
                            {
                                "step_index": i,
                                "plugin": plugin_name,
                                "category": plugin_category,
                                "error": str(e),
                            },
                        )
                        yield _sse_event("pipeline_error", {"error": str(e)})
                        return

                mlflow.log_dict(context, "context.json")

            yield _sse_event("pipeline_done", {"result": context})
        except Exception as e:
            yield _sse_event("pipeline_error", {"error": str(e)})

    @staticmethod
    def execute_step(parent_run_id: str, step: dict) -> dict:
        """Execute a single plugin step against an existing pipeline run.

        Loads the context from the parent run, executes the plugin as a
        nested child run, updates context, and re-logs context.json.

        Args:
            parent_run_id: MLflow run ID of the parent pipeline run.
            step: Dict with 'plugin' (str) and optional 'params' (dict).

        Returns:
            Updated context dict.
        """
        mlflow.set_experiment(EXPERIMENT_NAME)

        parent_loader = MLflowRunLoader(parent_run_id)
        context = parent_loader.get_json_artifact("context.json")

        plugin_name = step["plugin"]
        params = step.get("params", {})
        plugin = PluginManager.load(plugin_name)

        with mlflow.start_run(run_id=parent_run_id):
            with mlflow.start_run(run_name=f"{plugin_name}", nested=True) as pipeline_step:
                plugin.run(context, **params)

                plugin_category = plugin_name.split(".")[0]
                context[plugin_category] = {"run_id": pipeline_step.info.run_id}

            mlflow.log_dict(context, "context.json")

        return context


def _sse_event(event: str, data: dict) -> str:
    """Format a server-sent event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

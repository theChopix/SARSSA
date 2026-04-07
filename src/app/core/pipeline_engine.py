"""Pipeline execution engine with step-by-step and batch modes."""

import datetime
from typing import Any

import mlflow

from app.config.config import EXPERIMENT_NAME
from app.core.plugin_discovery.plugin_manager import PluginManager


class PipelineEngine:
    """Orchestrates plugin execution within MLflow-tracked pipeline runs.

    Supports two execution modes:

    **Batch mode** (original):
        ``PipelineEngine(steps).run(context)`` — runs all steps in one call.

    **Step-by-step mode** (for frontend-driven SSE streaming):
        1. ``engine.start_run()`` → creates parent MLflow run.
        2. ``engine.execute_step(...)`` → runs a single plugin step.
        3. ``engine.finalize_run(context)`` → logs context and closes run.
    """

    def __init__(self, steps: list[dict[str, Any]] | None = None) -> None:
        self.steps = steps or []
        self._parent_run_id: str | None = None

    # ── Step-by-step mode ─────────────────────────────

    def start_run(self) -> str:
        """Create a new parent pipeline run in MLflow.

        Returns:
            str: The parent run ID.

        Raises:
            RuntimeError: If a run is already in progress.
        """
        if self._parent_run_id is not None:
            raise RuntimeError(
                "A pipeline run is already in progress. "
                "Call finalize_run() before starting a new one."
            )

        mlflow.set_experiment(EXPERIMENT_NAME)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = mlflow.start_run(run_name=f"pipeline_run_{timestamp}")
        self._parent_run_id = run.info.run_id
        mlflow.end_run()
        return self._parent_run_id

    def execute_step(
        self,
        plugin_name: str,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single plugin step as a nested MLflow run.

        Args:
            plugin_name: Dotted module path of the plugin
                (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).
            params: Keyword arguments forwarded to ``Plugin.run()``.
            context: Mutable pipeline context dict.

        Returns:
            dict[str, Any]: The updated context with the new step's
                ``run_id`` added under its category key.

        Raises:
            RuntimeError: If no parent run is active.
        """
        if self._parent_run_id is None:
            raise RuntimeError("No active pipeline run. Call start_run() first.")

        plugin = PluginManager.load(plugin_name)

        with (
            mlflow.start_run(run_id=self._parent_run_id),
            mlflow.start_run(run_name=plugin_name, nested=True) as step_run,
        ):
            plugin.run(context, **params)

            category = plugin_name.split(".")[0]
            context[category] = {"run_id": step_run.info.run_id}

        return context

    def finalize_run(self, context: dict[str, Any]) -> None:
        """Log the final context and close the parent pipeline run.

        Args:
            context: The pipeline context to persist as ``context.json``.

        Raises:
            RuntimeError: If no parent run is active.
        """
        if self._parent_run_id is None:
            raise RuntimeError("No active pipeline run. Call start_run() first.")

        with mlflow.start_run(run_id=self._parent_run_id):
            mlflow.log_dict(context, "context.json")

        self._parent_run_id = None

    # ── Batch mode (original) ─────────────────────────

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute all steps sequentially in a single pipeline run.

        This is the original batch execution mode. For frontend-driven
        step-by-step execution, use ``start_run()`` / ``execute_step()``
        / ``finalize_run()`` instead.

        Args:
            context: Mutable pipeline context dict.

        Returns:
            dict[str, Any]: The final context with all step run IDs.
        """
        self.start_run()

        for step in self.steps:
            plugin_name = step["plugin"]
            params = step.get("params", {})
            self.execute_step(plugin_name, params, context)

        self.finalize_run(context)
        return context

"""Pipeline execution engine with step-by-step and batch modes."""

import datetime
from typing import Any

import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset

from app.config.config import EXPERIMENT_NAME, TIMEZONE
from app.core.plugin_discovery.naming import (
    format_pipeline_run_name,
    format_step_run_name,
    make_dataset_label,
    make_plugin_display_name,
)
from app.core.plugin_discovery.plugin_manager import PluginManager
from utils.cancellation import CancellationToken
from utils.plugin_notifier import PluginNotifier


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
        self._resumed_status: str | None = None

    # ── Step-by-step mode ─────────────────────────────

    _TAG_PREFIX = "sarssa."
    _DATASET_CATEGORY = "dataset_loading"
    _ORDER_OFFSET_TAG = "sarssa.order_offset"

    def start_run(
        self,
        tags: dict[str, str] | None = None,
        description: str = "",
        pipeline_name: str = "",
        order_offset: int = 0,
    ) -> str:
        """Create a new parent pipeline run in MLflow.

        Optionally annotates the run with user-provided key-value tags
        (each key is prefixed with ``sarssa.``) and a description, and
        names the run from an optional user-provided pipeline name.

        Args:
            tags: User-provided key-value tags for the pipeline run.
            description: User-provided free-text description.
            pipeline_name: Optional user-provided label woven into the
                run name (see
                :func:`~app.core.plugin_discovery.naming.format_pipeline_run_name`).
            order_offset: Number of inherited upstream steps. Offsets nested
                step numbering so a derived run continues past them, and when
                > 0 adds an ``( inherited )`` marker to the run name.

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

        mlflow_tags: dict[str, str] = {}
        if tags:
            mlflow_tags.update({f"{self._TAG_PREFIX}{k}": v for k, v in tags.items()})
        if order_offset:
            # Persist the offset so step numbering keeps continuing past the
            # inherited steps even for later execute-step calls on this run.
            mlflow_tags[self._ORDER_OFFSET_TAG] = str(order_offset)

        run = mlflow.start_run(
            run_name=format_pipeline_run_name(
                pipeline_name, datetime.datetime.now(TIMEZONE), derived=order_offset > 0
            ),
            tags=mlflow_tags or None,
            description=description or None,
        )
        self._parent_run_id = run.info.run_id

        mlflow.end_run()
        return self._parent_run_id

    def _next_execution_order(self) -> int:
        """Return the 1-based execution position of the next nested step.

        Counts the nested runs already attached to the active parent run
        (via the ``mlflow.parentRunId`` tag), adds one, and offsets by any
        inherited-step count stored on the parent so a derived run keeps
        numbering past its inherited upstream.  This is the single source of
        truth that works both in batch mode and across separate phase-2
        ``execute-step`` calls, neither of which keeps an in-memory counter.

        Note:
            If two phase-2 steps are executed concurrently on the same
            parent run they may read the same count and receive the same
            number.  MLflow run names need not be unique, so this is
            tolerated rather than locked against.

        Returns:
            int: The next 1-based execution order.

        Raises:
            RuntimeError: If no parent run is active.
        """
        if self._parent_run_id is None:
            raise RuntimeError("No active pipeline run. Call start_run() first.")

        client = mlflow.tracking.MlflowClient()
        parent = client.get_run(self._parent_run_id)
        # Inherited upstream steps aren't children of this run, so their count is
        # carried on the parent as a tag and added to the child-based counter.
        offset = int(parent.data.tags.get(self._ORDER_OFFSET_TAG, "0"))
        children = client.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{self._parent_run_id}'",
        )
        return offset + len(children) + 1

    def execute_step(
        self,
        plugin_name: str,
        params: dict[str, Any],
        context: dict[str, Any],
        notifier: PluginNotifier | None = None,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Execute a single plugin step as a nested MLflow run.

        Args:
            plugin_name: Dotted module path of the plugin
                (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).
            params: Keyword arguments forwarded to ``Plugin.run()``.
            context: Mutable pipeline context dict.
            notifier: Optional notifier to inject into the plugin before
                ``run()`` is called.  When provided, the plugin's
                ``notifier`` attribute is replaced so that any
                ``self.notifier.info(...)`` calls accumulate into the
                shared message list.  When ``None``, the plugin's default
                :class:`~utils.plugin_notifier.NullNotifier` is used.
            cancellation: Optional cancellation token injected into the
                plugin before ``run()``.  Lets a cooperating plugin abort
                mid-step.  When ``None``, the plugin's default
                :class:`~utils.cancellation.NullCancellationToken` is used.

        Returns:
            dict[str, Any]: The updated context with the new step's
                ``run_id`` added under its category key.

        Raises:
            RuntimeError: If no parent run is active.
        """
        if self._parent_run_id is None:
            raise RuntimeError("No active pipeline run. Call start_run() first.")

        plugin = PluginManager.load(plugin_name)

        if notifier is not None:
            plugin.notifier = notifier

        if cancellation is not None:
            plugin.cancellation = cancellation

        order = self._next_execution_order()
        display_name = plugin.name or make_plugin_display_name(plugin_name)
        run_name = format_step_run_name(plugin_name, display_name, order)

        with (
            mlflow.start_run(run_id=self._parent_run_id),
            mlflow.start_run(run_name=run_name, nested=True) as step_run,
        ):
            plugin.load_context(context)
            plugin.run(**params)
            plugin.update_context()

            category = plugin_name.split(".")[0]
            context[category] = {"run_id": step_run.info.run_id}

        if category == self._DATASET_CATEGORY:
            self._log_dataset_input(plugin_name)

        return context

    def _log_dataset_input(self, plugin_name: str) -> None:
        """Record the loaded dataset on the parent run's inputs.

        Populates the MLflow UI ``Dataset`` column for the pipeline run
        by logging a name-only :class:`~mlflow.data.meta_dataset.MetaDataset`
        — the actual interaction data is not duplicated into the tracking
        store.  The label is the dataset loader's implementation name with
        the ``_loader`` suffix stripped (e.g. ``movieLens``).

        Args:
            plugin_name: Dotted module path of the dataset-loading plugin.
        """
        dataset = MetaDataset(
            source=CodeDatasetSource(tags={}),
            name=make_dataset_label(plugin_name),
        )
        with mlflow.start_run(run_id=self._parent_run_id):
            mlflow.log_input(dataset)

    def log_step_param(self, plugin_name: str) -> None:
        """Log a step's plugin as a ``{category: plugin}`` param on the parent.

        Surfaces which plugin produced each of the run's own steps on the parent
        run page. Only steps this run executed are recorded; inherited upstream
        is linked from the run description instead.

        Args:
            plugin_name: Dotted module path of the executed plugin.

        Raises:
            RuntimeError: If no parent run is active.
        """
        if self._parent_run_id is None:
            raise RuntimeError("No active pipeline run. Call start_run() first.")

        category = plugin_name.split(".")[0]
        with mlflow.start_run(run_id=self._parent_run_id):
            mlflow.log_param(category, plugin_name.split(".")[-2])

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

    def fail_run(self, context: dict[str, Any]) -> None:
        """Mark the parent pipeline run as FAILED in MLflow.

        Logs the partial context (steps completed so far) and sets
        the run status to FAILED. Used when the pipeline is cancelled
        or encounters a fatal error.

        Args:
            context: The partial pipeline context to persist.

        Raises:
            RuntimeError: If no parent run is active.
        """
        if self._parent_run_id is None:
            raise RuntimeError("No active pipeline run. Call start_run() first.")

        with mlflow.start_run(run_id=self._parent_run_id):
            mlflow.log_dict(context, "context.json")
            mlflow.set_tag("cancellation", "cancelled_by_user")

        client = mlflow.tracking.MlflowClient()
        client.set_terminated(self._parent_run_id, status="FAILED")

        self._parent_run_id = None

    def resume_run(self, run_id: str) -> None:
        """Attach to an existing parent pipeline run.

        Use this to execute additional steps on a previously
        completed run (e.g. phase-2 multi-run plugins).

        Args:
            run_id: MLflow run ID of the parent run to resume.

        Raises:
            RuntimeError: If a run is already in progress.
        """
        if self._parent_run_id is not None:
            raise RuntimeError(
                "A pipeline run is already in progress. "
                "Call finalize_run() before resuming another."
            )

        mlflow.set_experiment(EXPERIMENT_NAME)
        self._parent_run_id = run_id
        # Remember the run's terminal status so restore_resumed_status() can
        # put it back if a failing step flips the reopened parent to FAILED.
        client = mlflow.tracking.MlflowClient()
        self._resumed_status = client.get_run(run_id).info.status

    def restore_resumed_status(self) -> None:
        """Restore a resumed parent run's original terminal status and detach.

        A failing step marks the reopened parent FAILED on exit; this puts
        the status captured by :meth:`resume_run` back. No-op unless a
        resumed run is active.
        """
        if self._parent_run_id is None or self._resumed_status is None:
            return

        client = mlflow.tracking.MlflowClient()
        if client.get_run(self._parent_run_id).info.status != self._resumed_status:
            client.set_terminated(self._parent_run_id, status=self._resumed_status)

        self._parent_run_id = None
        self._resumed_status = None

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

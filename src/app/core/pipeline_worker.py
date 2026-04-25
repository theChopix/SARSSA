"""Background worker that executes a pipeline and updates a TaskState.

The single public function :func:`run_pipeline_worker` is designed to
be the ``target`` of a :class:`threading.Thread`.  It receives a shared
:class:`~app.models.pipeline.TaskState` object and mutates it in-place
as it progresses through the requested steps.
"""

from typing import Any

from app.core.pipeline_engine import PipelineEngine
from app.core.pipeline_runs import get_run_context
from app.models.pipeline import TaskState
from app.utils.logger import logger
from utils.plugin_notifier import PluginNotifier


def run_pipeline_worker(task: TaskState) -> None:
    """Execute all pipeline steps and update *task* in-place.

    Called from a daemon thread spawned by the ``POST /run-async``
    endpoint.  The ``GET /tasks/{id}`` endpoint reads the same
    *task* object to report progress.

    On success, ``task.status`` is set to ``"completed"`` and
    ``task.context`` contains the final pipeline context.

    On failure, ``task.status`` is set to ``"error"`` and
    ``task.error`` contains the exception message.

    Args:
        task: Shared mutable task state to update during execution.
    """
    engine = PipelineEngine()
    notifier = PluginNotifier()
    task.messages = notifier.messages
    try:
        run_id = engine.start_run(tags=task.tags, description=task.description)
        task.run_id = run_id
        logger.info("[WORKER] Pipeline run started: %s", run_id)

        context: dict[str, Any] = dict(task.initial_context)

        for i, step in enumerate(task.steps_requested):
            # ── Cancellation check ────────────────────
            if task.cancel_event.is_set():
                logger.info("[WORKER] Cancellation requested before step %d", i)
                task.status = "cancelled"
                task.error = "Pipeline cancelled by user."
                engine.fail_run(context)
                return
            # ──────────────────────────────────────────

            plugin = step["plugin"]
            params = step.get("params") or {}
            category = plugin.split(".")[0]

            task.current_step = category
            task.current_step_index = i
            logger.info("[WORKER] Step %d/%d: %s", i + 1, len(task.steps_requested), plugin)

            engine.execute_step(plugin, params, context, notifier=notifier)

            task.completed_steps.append(
                {"category": category, "run_id": context[category]["run_id"]}
            )
            logger.info("[WORKER] Step completed: %s", category)

        engine.finalize_run(context)
        task.context = context
        task.current_step = None
        task.status = "completed"
        logger.info("[WORKER] Pipeline completed: %s", run_id)

    except Exception as exc:
        logger.exception("[WORKER] Pipeline failed: %s", exc)
        task.status = "error"
        task.error = str(exc)


def run_step_worker(task: TaskState) -> None:
    """Execute a single plugin step on an existing pipeline run.

    Called from a daemon thread spawned by the
    ``POST /runs/{run_id}/execute-step-async`` endpoint.  The worker
    resumes the MLflow parent run identified by ``task.run_id``,
    executes the one step in ``task.steps_requested``, and
    re-persists ``context.json``.

    On success, ``task.status`` is set to ``"completed"`` and
    ``task.context`` contains the updated pipeline context.

    On failure, ``task.status`` is set to ``"error"`` and
    ``task.error`` contains the exception message.

    Args:
        task: Shared mutable task state.  ``task.run_id`` and
            ``task.steps_requested`` must be set before calling.
    """
    engine = PipelineEngine()
    notifier = PluginNotifier()
    task.messages = notifier.messages
    try:
        if task.run_id is None:
            raise ValueError("task.run_id must be set before calling run_step_worker.")
        run_id: str = task.run_id
        logger.info("[STEP WORKER] Resuming run %s", run_id)

        context = get_run_context(run_id)
        engine.resume_run(run_id)

        step = task.steps_requested[0]
        plugin = step["plugin"]
        params = step.get("params") or {}
        category = plugin.split(".")[0]

        task.current_step = category
        task.current_step_index = 0
        logger.info("[STEP WORKER] Executing step: %s", plugin)

        engine.execute_step(plugin, params, context, notifier=notifier)

        task.completed_steps.append({"category": category, "run_id": context[category]["run_id"]})
        logger.info("[STEP WORKER] Step completed: %s", category)

        engine.finalize_run(context)
        task.context = context
        task.current_step = None
        task.status = "completed"
        logger.info("[STEP WORKER] Done (run %s)", run_id)

    except Exception as exc:
        logger.exception("[STEP WORKER] Step failed: %s", exc)
        task.status = "error"
        task.error = str(exc)

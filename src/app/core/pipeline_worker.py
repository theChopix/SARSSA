"""Background worker that executes a pipeline and updates a TaskState.

The single public function :func:`run_pipeline_worker` is designed to
be the ``target`` of a :class:`threading.Thread`.  It receives a shared
:class:`~app.models.pipeline.TaskState` object and mutates it in-place
as it progresses through the requested steps.
"""

from typing import Any

from app.core.pipeline_engine import PipelineEngine
from app.models.pipeline import TaskState
from app.utils.logger import logger


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
    try:
        run_id = engine.start_run()
        task.run_id = run_id
        logger.info("[WORKER] Pipeline run started: %s", run_id)

        context: dict[str, Any] = {}

        for i, step in enumerate(task.steps_requested):
            plugin = step["plugin"]
            params = step.get("params") or {}
            category = plugin.split(".")[0]

            task.current_step = category
            task.current_step_index = i
            logger.info("[WORKER] Step %d/%d: %s", i + 1, len(task.steps_requested), plugin)

            engine.execute_step(plugin, params, context)

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

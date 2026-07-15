"""Reconcile MLflow runs left ``RUNNING`` by a force-terminated backend.

A parent pipeline run (and its in-flight child step) is ``RUNNING`` only while
a step executes.  Force-killing the process mid-step — SIGKILL, OOM-killer,
``docker kill`` — skips the MLflow context manager's ``__exit__``, stranding
both runs as ``RUNNING`` forever.  This module sweeps such runs to ``FAILED``.

The sweep marks *every* ``RUNNING`` run in the app's experiments (all except
MLflow's catch-all ``Default``), which is safe only because a single backend
instance owns them (running two against one tracking server would let one
stomp the other's in-flight runs).
"""

import mlflow
from mlflow.entities import ViewType

from app.utils.logger import logger

#: Tag key recorded on each recovered run so the MLflow UI distinguishes a
#: force-terminated zombie from a genuine in-pipeline failure.
RECOVERY_TAG = "sarssa.recovery"


def fail_orphaned_runs(reason: str) -> list[str]:
    """Terminate every ``RUNNING`` run in the app's experiments as ``FAILED``.

    Args:
        reason: Short marker recorded under the :data:`RECOVERY_TAG` tag on
            each terminated run (e.g. ``"terminated_at_shutdown"``).

    Returns:
        list[str]: The run IDs that were terminated.
    """
    experiment_ids = [
        experiment.experiment_id
        for experiment in mlflow.search_experiments()
        if experiment.name != "Default"
    ]
    if not experiment_ids:
        return []

    client = mlflow.tracking.MlflowClient()
    terminated: list[str] = []
    page_token: str | None = None
    while True:
        page = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string="attributes.status = 'RUNNING'",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000,
            page_token=page_token,
        )
        for run in page:
            run_id = run.info.run_id
            try:
                client.set_tag(run_id, RECOVERY_TAG, reason)
                client.set_terminated(run_id, status="FAILED")
                terminated.append(run_id)
            except Exception:
                logger.exception("[RECOVERY] Could not terminate orphaned run %s", run_id)
        page_token = page.token
        if not page_token:
            break

    if terminated:
        logger.warning(
            "[RECOVERY] Marked %d orphaned RUNNING run(s) as FAILED (%s): %s",
            len(terminated),
            reason,
            ", ".join(terminated),
        )
    return terminated

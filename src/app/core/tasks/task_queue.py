"""FIFO queue that serialises compute tasks to one at a time.

The async endpoints submit their tasks here instead of spawning a thread
per request. A single dispatcher thread executes the queued workers
strictly one after another, so at most one computation runs at any
moment.
"""

import queue
import threading
import time
from collections.abc import Callable

from app.models.pipeline import TaskState
from app.utils.logger import logger

_queue: "queue.Queue[tuple[TaskState, Callable[[TaskState], None]]]" = queue.Queue()
_dispatcher_lock = threading.Lock()
_dispatcher: threading.Thread | None = None


def submit(task: TaskState, worker: Callable[[TaskState], None]) -> None:
    """Enqueue *task*; *worker* runs once all earlier tasks have finished.

    The task waits with status ``"queued"`` and is flipped to
    ``"running"`` by the dispatcher right before *worker* starts.

    Args:
        task: Task state created by the task store (status ``"queued"``).
        worker: Worker function that executes the task and sets its
            terminal status (e.g. ``run_pipeline_worker``).
    """
    _ensure_dispatcher()
    _queue.put((task, worker))


def _ensure_dispatcher() -> None:
    """Start the singleton dispatcher thread if it isn't running yet."""
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None or not _dispatcher.is_alive():
            _dispatcher = threading.Thread(
                target=_dispatch_loop,
                daemon=True,
                name="compute-task-dispatcher",
            )
            _dispatcher.start()


def _dispatch_loop() -> None:
    """Run queued tasks one at a time, skipping those cancelled while queued."""
    while True:
        task, worker = _queue.get()

        if task.cancel_event.is_set():
            if task.status == "queued":
                task.status = "cancelled"
                task.error = "Cancelled while queued."
            continue

        task.status = "running"
        task.started_at = time.time()
        try:
            worker(task)
        except Exception:
            # workers set their own terminal status-  this is a backstop so
            # a crashing worker can never kill the dispatcher.
            logger.exception("[QUEUE] Worker crashed for task %s", task.task_id)
            if task.status == "running":
                task.status = "error"
                task.error = "Worker crashed unexpectedly."

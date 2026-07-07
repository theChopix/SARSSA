"""Unit tests for app.core.tasks.task_queue (FIFO compute-task serialisation)."""

import threading
import time
from collections.abc import Callable

from app.core.tasks.task_queue import submit
from app.models.pipeline import TaskState


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Poll *predicate* until it is true or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_task(task_id: str) -> TaskState:
    return TaskState(task_id=task_id, status="queued")


class TestTaskQueue:
    """Tests for submit() and the dispatcher loop."""

    def test_runs_submitted_task(self) -> None:
        """Verify a submitted task's worker runs and can set a terminal state."""
        task = _make_task("t1")

        def worker(t: TaskState) -> None:
            t.status = "completed"

        submit(task, worker)

        assert _wait_until(lambda: task.status == "completed")

    def test_dispatcher_sets_running_before_worker(self) -> None:
        """Verify the task is flipped to 'running' when its worker starts."""
        task = _make_task("t1")
        seen: list[str] = []

        def worker(t: TaskState) -> None:
            seen.append(t.status)

        submit(task, worker)

        assert _wait_until(lambda: bool(seen))
        assert seen == ["running"]

    def test_tasks_run_one_at_a_time_in_fifo_order(self) -> None:
        """Verify the second task stays queued until the first finishes."""
        first, second = _make_task("t1"), _make_task("t2")
        release_first = threading.Event()
        first_started = threading.Event()

        def blocking_worker(t: TaskState) -> None:
            first_started.set()
            release_first.wait(timeout=2.0)
            t.status = "completed"

        def quick_worker(t: TaskState) -> None:
            t.status = "completed"

        submit(first, blocking_worker)
        submit(second, quick_worker)

        assert first_started.wait(timeout=2.0)
        assert second.status == "queued"

        release_first.set()
        assert _wait_until(lambda: second.status == "completed")
        assert first.status == "completed"

    def test_skips_task_cancelled_while_queued(self) -> None:
        """Verify a task cancelled in the queue is never executed."""
        first, second = _make_task("t1"), _make_task("t2")
        release_first = threading.Event()
        executed: list[str] = []

        def blocking_worker(t: TaskState) -> None:
            release_first.wait(timeout=2.0)
            t.status = "completed"

        def victim_worker(t: TaskState) -> None:
            executed.append(t.task_id)
            t.status = "completed"

        submit(first, blocking_worker)
        submit(second, victim_worker)
        second.cancel_event.set()
        release_first.set()

        assert _wait_until(lambda: second.status == "cancelled")
        assert executed == []
        assert second.error == "Cancelled while queued."

    def test_dispatcher_survives_worker_exception(self) -> None:
        """Verify a crashing worker doesn't kill the dispatcher."""
        crashing, follower = _make_task("t1"), _make_task("t2")

        def crashing_worker(_t: TaskState) -> None:
            raise RuntimeError("boom")

        def quick_worker(t: TaskState) -> None:
            t.status = "completed"

        submit(crashing, crashing_worker)
        submit(follower, quick_worker)

        assert _wait_until(lambda: follower.status == "completed")
        assert crashing.status == "error"

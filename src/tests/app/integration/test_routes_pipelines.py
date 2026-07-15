"""Integration tests for pipeline routes.

GET /runs and GET /runs/{run_id}/context hit the real MLflow tracking
store. POST endpoints mock the engine to avoid running actual plugins.
"""

import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Poll *predicate* until true or *timeout* elapses.

    The async endpoints hand tasks to the queue dispatcher thread, so
    assertions about worker calls / task status need a short wait.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestListRuns:
    """Tests for GET /pipelines/runs."""

    def test_returns_200(self, client: TestClient) -> None:
        """Verify the endpoint returns HTTP 200."""
        with patch(
            "app.api.routes_pipelines.get_pipeline_runs",
            return_value=[],
        ):
            response = client.get("/pipelines/runs")
        assert response.status_code == 200

    def test_returns_list(self, client: TestClient) -> None:
        """Verify the response is a JSON list."""
        with patch(
            "app.api.routes_pipelines.get_pipeline_runs",
            return_value=[
                {
                    "run_id": "r1",
                    "run_name": "run_a",
                    "status": "FINISHED",
                    "start_time": 1700000000000,
                }
            ],
        ):
            response = client.get("/pipelines/runs")
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["run_id"] == "r1"

    def test_required_steps_query_routes_to_eligible_filter(
        self,
        client: TestClient,
    ) -> None:
        """Verify ?required_steps= delegates to get_eligible_pipeline_runs."""
        eligible_payload: list[dict[str, Any]] = [
            {
                "run_id": "r2",
                "run_name": "eligible",
                "status": "FINISHED",
                "start_time": 1700000000001,
            },
        ]
        with (
            patch(
                "app.api.routes_pipelines.get_eligible_pipeline_runs",
                return_value=eligible_payload,
            ) as mock_eligible,
            patch(
                "app.api.routes_pipelines.get_pipeline_runs",
                return_value=[],
            ) as mock_all,
        ):
            response = client.get(
                "/pipelines/runs",
                params=[
                    ("required_steps", "dataset_loading"),
                    ("required_steps", "neuron_labeling"),
                ],
            )

        assert response.status_code == 200
        assert response.json() == eligible_payload
        mock_eligible.assert_called_once_with(
            ["dataset_loading", "neuron_labeling"],
            "",
        )
        mock_all.assert_not_called()

    def test_no_required_steps_uses_unfiltered_query(
        self,
        client: TestClient,
    ) -> None:
        """Verify the unfiltered path is used when no query param is sent."""
        with (
            patch(
                "app.api.routes_pipelines.get_eligible_pipeline_runs",
            ) as mock_eligible,
            patch(
                "app.api.routes_pipelines.get_pipeline_runs",
                return_value=[],
            ) as mock_all,
        ):
            response = client.get("/pipelines/runs")

        assert response.status_code == 200
        mock_all.assert_called_once_with("")
        mock_eligible.assert_not_called()


class TestGetContext:
    """Tests for GET /pipelines/runs/{run_id}/context."""

    def test_returns_context(self, client: TestClient) -> None:
        """Verify context.json contents are returned."""
        fake_ctx = {"dataset_loading": {"run_id": "abc"}}
        with patch(
            "app.api.routes_pipelines.get_run_context",
            return_value=fake_ctx,
        ):
            response = client.get("/pipelines/runs/some_id/context")

        assert response.status_code == 200
        assert response.json() == fake_ctx

    def test_returns_404_on_missing_artifact(self, client: TestClient) -> None:
        """Verify 404 when context.json is missing."""
        with patch(
            "app.api.routes_pipelines.get_run_context",
            side_effect=FileNotFoundError("context.json not found"),
        ):
            response = client.get("/pipelines/runs/bad_id/context")

        assert response.status_code == 404


class TestRunAsync:
    """Tests for POST /pipelines/run-async."""

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_returns_200_with_task_id(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify the endpoint returns 200 and a task_id."""
        response = client.post(
            "/pipelines/run-async",
            json={
                "steps": [
                    {"plugin": "cat_a.impl.impl", "params": {}},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert isinstance(data["task_id"], str)

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_task_exists_in_store(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify the task is retrievable from the store after creation."""
        response = client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )
        task_id = response.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        # Queued at creation; the dispatcher flips it to running shortly.
        assert _wait_until(lambda: task.status == "running")

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_spawns_worker(self, mock_worker: MagicMock, client: TestClient) -> None:
        """Verify run_pipeline_worker is called with the task."""
        client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )

        assert _wait_until(lambda: mock_worker.called)
        mock_worker.assert_called_once()

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_forwards_tags_to_task(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify tags from the request body are stored on the task."""
        response = client.post(
            "/pipelines/run-async",
            json={
                "steps": [{"plugin": "cat.p.p", "params": {}}],
                "tags": {"dataset": "MovieLens", "model": "ELSA"},
            },
        )
        task_id = response.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.tags == {"dataset": "MovieLens", "model": "ELSA"}

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_forwards_description_to_task(
        self, _mock_worker: MagicMock, client: TestClient
    ) -> None:
        """Verify description from the request body is stored on the task."""
        response = client.post(
            "/pipelines/run-async",
            json={
                "steps": [{"plugin": "cat.p.p", "params": {}}],
                "description": "Baseline run",
            },
        )
        task_id = response.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.description == "Baseline run"

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_forwards_pipeline_name_to_task(
        self, _mock_worker: MagicMock, client: TestClient
    ) -> None:
        """Verify pipeline_name from the request body is stored on the task."""
        response = client.post(
            "/pipelines/run-async",
            json={
                "steps": [{"plugin": "cat.p.p", "params": {}}],
                "pipeline_name": "Baseline ELSA",
            },
        )
        task_id = response.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.pipeline_name == "Baseline ELSA"

    def test_rejects_overlong_pipeline_name(self, client: TestClient) -> None:
        """Verify a pipeline_name over 60 characters is rejected with 422."""
        response = client.post(
            "/pipelines/run-async",
            json={
                "steps": [{"plugin": "cat.p.p", "params": {}}],
                "pipeline_name": "x" * 61,
            },
        )
        assert response.status_code == 422


class TestListRunningTasks:
    """Tests for GET /pipelines/tasks."""

    def _start(self, client: TestClient, **body: Any) -> str:
        """Start a run-async task and return its task_id."""
        return client.post("/pipelines/run-async", json=body).json()["task_id"]

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_lists_only_running_tasks(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify completed tasks are excluded from the listing."""
        from app.core.tasks.task_store import _tasks, get_task

        _tasks.clear()
        r1 = self._start(client, steps=[{"plugin": "cat_a.p.p", "params": {}}])
        r2 = self._start(client, steps=[{"plugin": "cat_b.p.p", "params": {}}])
        done = self._start(client, steps=[{"plugin": "cat_c.p.p", "params": {}}])
        done_task = get_task(done)
        assert done_task is not None
        done_task.status = "completed"

        response = client.get("/pipelines/tasks")

        assert response.status_code == 200
        ids = [t["task_id"] for t in response.json()]
        assert r1 in ids
        assert r2 in ids
        assert done not in ids

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_summary_includes_steps_and_progress(
        self, _mock_worker: MagicMock, client: TestClient
    ) -> None:
        """Verify the summary carries progress and the full steps_requested."""
        from app.core.tasks.task_store import _tasks, get_task

        _tasks.clear()
        task_id = self._start(
            client,
            steps=[
                {"plugin": "cat_a.p.p", "params": {"x": 1}},
                {"plugin": "cat_b.p.p", "params": {}},
            ],
            pipeline_name="Baseline",
        )
        task = get_task(task_id)
        assert task is not None
        task.current_step = "cat_a"

        entry = next(t for t in client.get("/pipelines/tasks").json() if t["task_id"] == task_id)

        assert entry["pipeline_name"] == "Baseline"
        assert entry["total_steps"] == 2
        assert entry["current_step"] == "cat_a"
        assert len(entry["steps_requested"]) == 2
        assert entry["steps_requested"][0]["plugin"] == "cat_a.p.p"

    def test_empty_when_none_running(self, client: TestClient) -> None:
        """Verify an empty list when the store has no running tasks."""
        from app.core.tasks.task_store import _tasks

        _tasks.clear()
        response = client.get("/pipelines/tasks")

        assert response.status_code == 200
        assert response.json() == []


class TestGetTaskStatus:
    """Tests for GET /pipelines/tasks/{task_id}."""

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_returns_200_with_status(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify the endpoint returns 200 with all expected fields."""
        create_resp = client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )
        task_id = create_resp.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert _wait_until(lambda: task.status == "running")

        response = client.get(f"/pipelines/tasks/{task_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "running"
        assert data["total_steps"] == 1
        assert data["completed_steps"] == []
        assert data["current_step"] is None
        assert data["error"] is None

    def test_returns_404_for_unknown_task(self, client: TestClient) -> None:
        """Verify 404 when the task ID does not exist."""
        response = client.get("/pipelines/tasks/nonexistent-id")
        assert response.status_code == 404

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_reflects_task_mutations(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify the response reflects in-place mutations to the task."""
        create_resp = client.post(
            "/pipelines/run-async",
            json={
                "steps": [
                    {"plugin": "cat_a.p.p", "params": {}},
                    {"plugin": "cat_b.p.p", "params": {}},
                ]
            },
        )
        task_id = create_resp.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        task.run_id = "mlflow_123"
        task.current_step = "cat_a"
        task.current_step_index = 0
        task.completed_steps.append({"category": "cat_a", "run_id": "r1"})

        response = client.get(f"/pipelines/tasks/{task_id}")
        data = response.json()

        assert data["run_id"] == "mlflow_123"
        assert data["current_step"] == "cat_a"
        assert data["current_step_index"] == 0
        assert data["total_steps"] == 2
        assert len(data["completed_steps"]) == 1


class TestCancelTask:
    """Tests for POST /pipelines/tasks/{task_id}/cancel."""

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_returns_200_on_success(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify 200 and confirmation message for a running task."""
        create_resp = client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )
        task_id = create_resp.json()["task_id"]

        response = client.post(f"/pipelines/tasks/{task_id}/cancel")

        assert response.status_code == 200
        assert "message" in response.json()

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_sets_cancel_event(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify cancel_event is set on the task after cancellation."""
        create_resp = client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )
        task_id = create_resp.json()["task_id"]

        client.post(f"/pipelines/tasks/{task_id}/cancel")

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.cancel_event.is_set()

    def test_returns_404_for_unknown_task(self, client: TestClient) -> None:
        """Verify 404 when the task ID does not exist."""
        response = client.post("/pipelines/tasks/nonexistent-id/cancel")
        assert response.status_code == 404

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_graceful_mode_leaves_abort_unset(
        self, _mock_worker: MagicMock, client: TestClient
    ) -> None:
        """Verify the default mode sets only cancel_event, not abort_event."""
        create_resp = client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )
        task_id = create_resp.json()["task_id"]

        client.post(f"/pipelines/tasks/{task_id}/cancel")

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.cancel_event.is_set()
        assert not task.abort_event.is_set()

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_now_mode_sets_abort_event(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify mode=now sets both cancel_event and abort_event."""
        create_resp = client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )
        task_id = create_resp.json()["task_id"]

        response = client.post(f"/pipelines/tasks/{task_id}/cancel?mode=now")

        assert response.status_code == 200
        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.cancel_event.is_set()
        assert task.abort_event.is_set()

    def test_invalid_mode_rejected(self, client: TestClient) -> None:
        """Verify an unknown mode is rejected with 422 (query validation)."""
        response = client.post("/pipelines/tasks/whatever/cancel?mode=bogus")
        assert response.status_code == 422

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_returns_409_for_completed_task(
        self, _mock_worker: MagicMock, client: TestClient
    ) -> None:
        """Verify 409 when the task is already completed."""
        create_resp = client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )
        task_id = create_resp.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        task.status = "completed"

        response = client.post(f"/pipelines/tasks/{task_id}/cancel")
        assert response.status_code == 409


class TestExecuteStep:
    """Tests for POST /pipelines/runs/{run_id}/execute-step."""

    @patch("app.api.routes_pipelines.PipelineEngine")
    @patch("app.api.routes_pipelines.get_run_context")
    def test_returns_step_run_id(
        self,
        mock_get_ctx: MagicMock,
        mock_engine_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify the response contains category and step_run_id."""
        mock_get_ctx.return_value = {"dataset_loading": {"run_id": "old"}}

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any]
        ) -> dict[str, Any]:
            category = plugin.split(".")[0]
            ctx[category] = {"run_id": "new_step_run"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        response = client.post(
            "/pipelines/runs/parent_123/execute-step",
            json={
                "plugin": "steering.single.sae_steering.sae_steering",
                "params": {"alpha": 0.5},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "steering"
        assert data["step_run_id"] == "new_step_run"

    @patch("app.api.routes_pipelines.PipelineEngine")
    @patch("app.api.routes_pipelines.get_run_context")
    def test_resumes_correct_run(
        self,
        mock_get_ctx: MagicMock,
        mock_engine_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify resume_run is called with the path run_id."""
        mock_get_ctx.return_value = {}

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any]
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "x"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        client.post(
            "/pipelines/runs/my_run_42/execute-step",
            json={"plugin": "inspection.insp.insp", "params": {}},
        )

        mock_engine.resume_run.assert_called_once_with("my_run_42")

    @patch("app.api.routes_pipelines.PipelineEngine")
    @patch("app.api.routes_pipelines.get_run_context")
    def test_finalizes_context(
        self,
        mock_get_ctx: MagicMock,
        mock_engine_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify the sync endpoint re-persists context.json.

        The sync path must call ``finalize_run`` with the context
        mutated by ``execute_step`` so the new step is visible to
        downstream steps, matching the async variant.
        """
        mock_get_ctx.return_value = {"dataset_loading": {"run_id": "old"}}

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any]
        ) -> dict[str, Any]:
            ctx[plugin.split(".")[0]] = {"run_id": "new_step_run"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        response = client.post(
            "/pipelines/runs/parent_123/execute-step",
            json={"plugin": "steering.single.sae_steering.sae_steering", "params": {}},
        )

        assert response.status_code == 200
        mock_engine.finalize_run.assert_called_once_with(
            {
                "dataset_loading": {"run_id": "old"},
                "steering": {"run_id": "new_step_run"},
            }
        )


class TestExecuteStepAsync:
    """Tests for POST /pipelines/runs/{run_id}/execute-step-async."""

    @patch("app.api.routes_pipelines.run_step_worker")
    def test_returns_200_with_task_id(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify the endpoint returns 200 and a task_id immediately."""
        response = client.post(
            "/pipelines/runs/parent_run_1/execute-step-async",
            json={"plugin": "labeling_evaluation.impl.impl", "params": {}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert isinstance(data["task_id"], str)

    @patch("app.api.routes_pipelines.run_step_worker")
    def test_task_has_run_id_pre_set(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify the task's run_id matches the path parameter."""
        response = client.post(
            "/pipelines/runs/my_parent_run/execute-step-async",
            json={"plugin": "inspection.insp.insp", "params": {}},
        )
        task_id = response.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.run_id == "my_parent_run"

    @patch("app.api.routes_pipelines.run_step_worker")
    def test_task_has_correct_step(self, _mock_worker: MagicMock, client: TestClient) -> None:
        """Verify steps_requested contains exactly the one submitted step."""
        response = client.post(
            "/pipelines/runs/run_abc/execute-step-async",
            json={
                "plugin": "steering.single.sae_steering.sae_steering",
                "params": {"alpha": 0.3},
            },
        )
        task_id = response.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert len(task.steps_requested) == 1
        assert task.steps_requested[0]["plugin"] == "steering.single.sae_steering.sae_steering"

    @patch("app.api.routes_pipelines.run_step_worker")
    def test_spawns_step_worker(self, mock_worker: MagicMock, client: TestClient) -> None:
        """Verify run_step_worker is called with the task."""
        client.post(
            "/pipelines/runs/run_x/execute-step-async",
            json={"plugin": "inspection.insp.insp", "params": {}},
        )

        assert _wait_until(lambda: mock_worker.called)
        mock_worker.assert_called_once()

    @patch("app.api.routes_pipelines.run_step_worker")
    def test_task_status_polled_via_get_tasks(
        self, _mock_worker: MagicMock, client: TestClient
    ) -> None:
        """Verify the returned task_id is immediately queryable via GET /tasks/{id}."""
        create_resp = client.post(
            "/pipelines/runs/run_y/execute-step-async",
            json={"plugin": "labeling_evaluation.impl.impl", "params": {}},
        )
        task_id = create_resp.json()["task_id"]

        from app.core.tasks.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert _wait_until(lambda: task.status == "running")

        poll_resp = client.get(f"/pipelines/tasks/{task_id}")

        assert poll_resp.status_code == 200
        data = poll_resp.json()
        assert data["task_id"] == task_id
        assert data["status"] == "running"
        assert data["total_steps"] == 1

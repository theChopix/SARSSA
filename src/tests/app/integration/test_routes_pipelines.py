"""Integration tests for pipeline routes.

GET /runs and GET /runs/{run_id}/context hit the real MLflow tracking
store. POST endpoints mock the engine to avoid running actual plugins.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


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

        from app.core.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.status == "running"

    @patch("app.api.routes_pipelines.run_pipeline_worker")
    def test_spawns_worker(self, mock_worker: MagicMock, client: TestClient) -> None:
        """Verify run_pipeline_worker is called with the task."""
        client.post(
            "/pipelines/run-async",
            json={"steps": [{"plugin": "cat.p.p", "params": {}}]},
        )

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

        from app.core.task_store import get_task

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

        from app.core.task_store import get_task

        task = get_task(task_id)
        assert task is not None
        assert task.description == "Baseline run"


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

        from app.core.task_store import get_task

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
                "plugin": "steering.sae_steering.sae_steering",
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

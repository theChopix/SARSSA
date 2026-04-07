"""Integration tests for pipeline routes.

GET /runs and GET /runs/{run_id}/context hit the real MLflow tracking
store. POST endpoints mock the engine to avoid running actual plugins.
"""

import json
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


class TestRunStream:
    """Tests for POST /pipelines/run-stream (SSE)."""

    @patch("app.api.routes_pipelines.PipelineEngine")
    def test_emits_sse_events(self, mock_engine_cls: MagicMock, client: TestClient) -> None:
        """Verify the SSE stream emits expected event types."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "parent_id"

        def fake_execute(
            plugin: str, _params: dict[str, Any], ctx: dict[str, Any]
        ) -> dict[str, Any]:
            category = plugin.split(".")[0]
            ctx[category] = {"run_id": f"{category}_run"}
            return ctx

        mock_engine.execute_step.side_effect = fake_execute

        response = client.post(
            "/pipelines/run-stream",
            json={
                "steps": [
                    {"plugin": "cat_a.impl.impl", "params": {}},
                    {"plugin": "cat_b.impl.impl", "params": {}},
                ]
            },
        )

        assert response.status_code == 200

        lines = [line for line in response.text.split("\n") if line.startswith("data:")]
        events_data = [json.loads(line[len("data:") :]) for line in lines]

        assert any("run_id" in e and e["run_id"] == "parent_id" for e in events_data)
        assert any("category" in e and e["category"] == "cat_a" for e in events_data)
        assert any("category" in e and e["category"] == "cat_b" for e in events_data)
        assert any("context" in e for e in events_data)

    @patch("app.api.routes_pipelines.PipelineEngine")
    def test_calls_finalize(self, mock_engine_cls: MagicMock, client: TestClient) -> None:
        """Verify finalize_run is called after all steps."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.start_run.return_value = "parent_id"

        client.post(
            "/pipelines/run-stream",
            json={"steps": []},
        )

        mock_engine.finalize_run.assert_called_once()


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

"""Integration tests for plugin API routes.

Tests hit the real endpoints via FastAPI TestClient, exercising the
full stack from route → registry → filesystem plugin discovery.
Where downstream MLflow/plugin-loader interactions would otherwise
require fixture data, those collaborators are mocked while the
FastAPI router and handler logic continue to run for real.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.config.config import PLUGIN_CATEGORIES
from plugins.plugin_interface import DynamicDropdownHint, PluginIOSpec


class TestGetPluginRegistry:
    """Tests for GET /plugins/registry."""

    def test_returns_200(self, client: TestClient) -> None:
        """Verify the endpoint returns HTTP 200."""
        response = client.get("/plugins/registry")
        assert response.status_code == 200

    def test_contains_all_categories(self, client: TestClient) -> None:
        """Verify every category from PLUGIN_CATEGORIES is in the response."""
        response = client.get("/plugins/registry")
        data = response.json()
        assert set(data.keys()) == set(PLUGIN_CATEGORIES.keys())

    def test_each_category_has_expected_fields(self, client: TestClient) -> None:
        """Verify each category entry contains category_info and implementations."""
        response = client.get("/plugins/registry")
        data = response.json()

        for key, entry in data.items():
            assert "category_info" in entry, f"Missing category_info for {key}"
            assert "implementations" in entry, f"Missing implementations for {key}"

    def test_category_info_has_expected_fields(self, client: TestClient) -> None:
        """Verify category_info contains order, type, and display_name."""
        response = client.get("/plugins/registry")
        data = response.json()

        for key, entry in data.items():
            info = entry["category_info"]
            assert "order" in info, f"Missing order for {key}"
            assert "type" in info, f"Missing type for {key}"
            assert "display_name" in info, f"Missing display_name for {key}"

    def test_every_category_has_at_least_one_implementation(self, client: TestClient) -> None:
        """Verify each category discovered at least one plugin."""
        response = client.get("/plugins/registry")
        data = response.json()

        for key, entry in data.items():
            assert len(entry["implementations"]) >= 1, f"No implementations for {key}"

    def test_implementations_have_expected_fields(self, client: TestClient) -> None:
        """Verify each implementation has plugin_name, display_name, params, display."""
        response = client.get("/plugins/registry")
        data = response.json()

        for key, entry in data.items():
            for impl in entry["implementations"]:
                assert "plugin_name" in impl, f"Missing plugin_name in {key}"
                assert "display_name" in impl, f"Missing display_name in {key}"
                assert "params" in impl, f"Missing params in {key}"
                assert "display" in impl, f"Missing display in {key}"

    def test_steering_has_display_spec(self, client: TestClient) -> None:
        """Verify steering implementations include a non-null display spec."""
        response = client.get("/plugins/registry")
        data = response.json()

        for impl in data["steering"]["implementations"]:
            assert impl["display"] is not None
            assert impl["display"]["type"] == "item_rows"
            assert len(impl["display"]["rows"]) > 0

    def test_inspection_has_display_spec(self, client: TestClient) -> None:
        """Verify inspection implementations include a non-null display spec."""
        response = client.get("/plugins/registry")
        data = response.json()

        for impl in data["inspection"]["implementations"]:
            assert impl["display"] is not None
            assert impl["display"]["type"] == "item_rows"
            assert len(impl["display"]["rows"]) > 0

    def test_dataset_loading_has_no_display_spec(self, client: TestClient) -> None:
        """Verify dataset_loading implementations have display as null."""
        response = client.get("/plugins/registry")
        data = response.json()

        for impl in data["dataset_loading"]["implementations"]:
            assert impl["display"] is None

    def test_params_include_widget_fields(self, client: TestClient) -> None:
        """Verify every param has widget and widget_config fields."""
        response = client.get("/plugins/registry")
        data = response.json()

        for _key, entry in data.items():
            for impl in entry["implementations"]:
                for param in impl["params"]:
                    assert "widget" in param, (
                        f"Missing widget for {impl['plugin_name']}.{param['name']}"
                    )
                    assert "widget_config" in param, (
                        f"Missing widget_config for {impl['plugin_name']}.{param['name']}"
                    )


class TestGetParamChoicesEndpoint:
    """Tests for /plugins/param-choices.

    The FastAPI router and handler run for real throughout.  Cases
    that need fabricated plugin instances or artifact payloads stub
    ``PluginManager`` / ``MLflowRunLoader``; cases that exercise
    failure paths through real plugin discovery hit the live
    registry instead.
    """

    def test_returns_422_when_run_id_missing(
        self,
        client: TestClient,
    ) -> None:
        """Verify 422 when run_id query param is not provided."""
        response = client.get(
            "/plugins/param-choices/steering/steering.sae_steering.sae_steering/neuron_id",
        )
        assert response.status_code == 422

    def test_returns_404_for_nonexistent_plugin(
        self,
        client: TestClient,
    ) -> None:
        """Verify 404 when plugin module path does not exist."""
        response = client.get(
            "/plugins/param-choices/steering/steering.nonexistent.nonexistent/x",
            params={"run_id": "fake_run_id"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_returns_404_for_param_without_hint(
        self,
        client: TestClient,
    ) -> None:
        """Verify 404 for a real plugin param that has no dropdown hint."""
        response = client.get(
            "/plugins/param-choices/dataset_loading"
            "/dataset_loading.movieLens_loader.movieLens_loader"
            "/no_such_param",
            params={"run_id": "fake_run_id"},
        )
        assert response.status_code == 404
        assert "No DynamicDropdownHint" in response.json()["detail"]

    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_returns_formatted_choices(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify the endpoint returns formatted dropdown options."""

        def _fmt(data: dict) -> list[dict[str, str]]:
            return [{"label": f"{v} [id {k}]", "value": k} for k, v in data.items()]

        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            artifact_loader="json",
            formatter="_fmt",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        plugin.__class__._fmt = staticmethod(_fmt)
        mock_pm.load.return_value = plugin

        mock_loader = MagicMock()
        mock_loader.get_json_artifact.return_value = {
            "0": "sci-fi",
            "42": "comedy",
        }
        mock_loader_cls.return_value = mock_loader

        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/neuron_id",
            params={"run_id": "abc123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert {"label": "sci-fi [id 0]", "value": "0"} in data
        assert {"label": "comedy [id 42]", "value": "42"} in data

    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_returns_404_for_missing_formatter(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify 404 when the formatter method doesn't exist."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_file="neuron_labels.json",
            artifact_loader="json",
            formatter="_nonexistent_formatter",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        plugin.__class__ = type(
            "FakePlugin",
            (),
            {"io_spec": plugin.io_spec},
        )
        mock_pm.load.return_value = plugin

        mock_loader = MagicMock()
        mock_loader.get_json_artifact.return_value = {}
        mock_loader_cls.return_value = mock_loader

        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/neuron_id",
            params={"run_id": "abc123"},
        )

        assert response.status_code == 404
        assert "Formatter" in response.json()["detail"]

    @patch("app.api.routes_plugins.get_run_context")
    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_cascading_hint_resolves_via_parent_context(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
        mock_get_context: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify cascading dropdown loads artifact from parent run's step."""

        def _fmt(data: dict) -> list[dict[str, str]]:
            return [{"label": label, "value": nid} for nid, label in data.items()]

        hint = DynamicDropdownHint(
            param_name="past_neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            artifact_loader="json",
            formatter="_fmt",
            source_run_param="past_run_id",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        plugin.__class__._fmt = staticmethod(_fmt)
        mock_pm.load.return_value = plugin

        mock_get_context.return_value = {
            "neuron_labeling": {"run_id": "past_step_run"},
        }

        mock_loader = MagicMock()
        mock_loader.get_json_artifact.return_value = {"7": "drama"}
        mock_loader_cls.return_value = mock_loader

        response = client.get(
            "/plugins/param-choices/inspection/inspection.compare.x.x/past_neuron_id",
            params={"run_id": "past_parent_run"},
        )

        assert response.status_code == 200
        assert response.json() == [{"label": "drama", "value": "7"}]
        mock_get_context.assert_called_once_with("past_parent_run")
        mock_loader_cls.assert_called_once_with("past_step_run")

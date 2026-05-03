"""Integration tests for plugin API routes.

Tests hit the real endpoints via FastAPI TestClient, exercising the
full stack from route → registry → filesystem plugin discovery.
"""

from fastapi.testclient import TestClient

from app.config.config import PLUGIN_CATEGORIES


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


class TestGetParamChoicesIntegration:
    """Tests for GET /plugins/param-choices."""

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

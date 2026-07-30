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
from plugins.plugin_interface import (
    DependentDropdownHint,
    DropdownArtifactSpec,
    DynamicDropdownHint,
    PluginIOSpec,
)


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
        """Verify category_info contains order, type, display_name, and description."""
        response = client.get("/plugins/registry")
        data = response.json()

        for key, entry in data.items():
            info = entry["category_info"]
            assert "order" in info, f"Missing order for {key}"
            assert "type" in info, f"Missing type for {key}"
            assert "display_name" in info, f"Missing display_name for {key}"
            assert "description" in info, f"Missing description for {key}"
            assert info["description"], f"Empty description for {key}"

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
                assert "description" in impl, f"Missing description in {key}"
                assert "params" in impl, f"Missing params in {key}"
                assert "display" in impl, f"Missing display in {key}"

    def test_every_implementation_has_a_description(self, client: TestClient) -> None:
        """Verify every implementation entry carries a non-empty description."""
        response = client.get("/plugins/registry")
        data = response.json()

        for key, entry in data.items():
            for impl in entry["implementations"]:
                assert impl["description"], f"Empty description for {impl['plugin_name']} in {key}"

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
            "/plugins/param-choices/steering/steering.single.sae_steering.sae_steering/neuron_id",
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

    @patch("app.api.routes_plugins.PluginManager")
    def test_dependent_hint_resolves_choices_from_value(
        self,
        mock_pm: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify a dependent dropdown returns choices for the controlling value."""
        hint = DependentDropdownHint(
            param_name="embedding_model",
            depends_on_param="embedding_provider",
            resolver="embedder_models",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        mock_pm.load.return_value = plugin

        response = client.get(
            "/plugins/param-choices/labeling_evaluation/x.x/embedding_model",
            params={"value": "openai"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data
        assert {"label": "text-embedding-3-small", "value": "text-embedding-3-small"} in data

    @patch("app.api.routes_plugins.PluginManager")
    def test_dependent_hint_empty_value_returns_empty(
        self,
        mock_pm: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify a dependent dropdown with no controlling value returns no options."""
        hint = DependentDropdownHint(
            param_name="embedding_model",
            depends_on_param="embedding_provider",
            resolver="embedder_models",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        mock_pm.load.return_value = plugin

        response = client.get(
            "/plugins/param-choices/labeling_evaluation/x.x/embedding_model",
        )

        assert response.status_code == 200
        assert response.json() == []

    @patch("app.api.routes_plugins.PluginManager")
    def test_dependent_hint_unknown_value_returns_empty(
        self,
        mock_pm: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify an unrecognised controlling value yields no options (not an error)."""
        hint = DependentDropdownHint(
            param_name="embedding_model",
            depends_on_param="embedding_provider",
            resolver="embedder_models",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        mock_pm.load.return_value = plugin

        response = client.get(
            "/plugins/param-choices/labeling_evaluation/x.x/embedding_model",
            params={"value": "no_such_provider"},
        )

        assert response.status_code == 200
        assert response.json() == []

    @patch("app.api.routes_plugins.PluginManager")
    def test_dependent_hint_unknown_resolver_returns_404(
        self,
        mock_pm: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify a hint naming an unregistered resolver surfaces as 404."""
        hint = DependentDropdownHint(
            param_name="embedding_model",
            depends_on_param="embedding_provider",
            resolver="no_such_resolver",
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        mock_pm.load.return_value = plugin

        response = client.get(
            "/plugins/param-choices/labeling_evaluation/x.x/embedding_model",
            params={"value": "openai"},
        )

        assert response.status_code == 404
        assert "no_such_resolver" in response.json()["detail"]

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


class TestServerSearchParamChoices:
    """Tests for server_search dropdowns on the param-choices endpoint."""

    def setup_method(self) -> None:
        """Clear the formatted-options cache between tests."""
        from app.api.routes_plugins import _OPTIONS_CACHE

        _OPTIONS_CACHE.clear()

    def _mock_plugin(self) -> MagicMock:
        """Build a mock plugin with a server-searched multi-artifact hint.

        Returns:
            MagicMock: Plugin whose formatter joins users and counts.
        """

        def _fmt(data: tuple) -> list[dict[str, object]]:
            users, counts = data
            return [
                {
                    "label": f"user {u} · {c} interactions [row {i}]",
                    "value": str(i),
                    "tint": c / 9,
                }
                for i, (u, c) in enumerate(zip(users, counts, strict=True))
            ]

        hint = DynamicDropdownHint(
            param_name="user_id",
            artifact_step="dataset_loading",
            artifact_files=[
                DropdownArtifactSpec(file="users.npy", loader="npy", kwargs={"allow_pickle": True}),
                DropdownArtifactSpec(file="counts.npy", loader="npy"),
            ],
            formatter="_fmt",
            server_search=True,
        )
        plugin = MagicMock()
        plugin.io_spec = PluginIOSpec(param_ui_hints=[hint])
        plugin.__class__._fmt = staticmethod(_fmt)
        return plugin

    def _mock_loader(self) -> MagicMock:
        """Loader stub serving three users and their counts."""
        loader = MagicMock()
        loader.get_npy_artifact.side_effect = lambda f, **_kw: (
            ["alpha", "beta", "gamma"] if f == "users.npy" else [5, 2, 9]
        )
        return loader

    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_returns_envelope_with_limit(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify the envelope shape and that limit truncates options."""
        mock_pm.load.return_value = self._mock_plugin()
        mock_loader_cls.return_value = self._mock_loader()

        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/user_id",
            params={"run_id": "r1", "limit": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["options"]) == 2
        assert data["options"][0]["value"] == "0"

    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_search_filters_labels(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify search filters options and total counts the matches."""
        mock_pm.load.return_value = self._mock_plugin()
        mock_loader_cls.return_value = self._mock_loader()

        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/user_id",
            params={"run_id": "r1", "search": "BETA"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["options"][0]["value"] == "1"

    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_options_cached_per_run(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify the second request reuses formatted options without loading."""
        mock_pm.load.return_value = self._mock_plugin()
        mock_loader = self._mock_loader()
        mock_loader_cls.return_value = mock_loader

        for _ in range(2):
            client.get(
                "/plugins/param-choices/steering/steering.sae.sae/user_id",
                params={"run_id": "r1"},
            )

        # One call per artifact file, first request only.
        assert mock_loader.get_npy_artifact.call_count == 2
        mock_loader.get_npy_artifact.assert_any_call("users.npy", allow_pickle=True)

    def test_registry_marks_steering_user_id_as_server_searched(self, client: TestClient) -> None:
        """Verify the real registry exposes user_id as a server-search dropdown."""
        registry = client.get("/plugins/registry").json()
        for impl in registry["steering"]["implementations"]:
            param = next(p for p in impl["params"] if p["name"] == "user_id")
            assert param["widget"] == "dropdown"
            assert param["widget_config"]["server_search"] is True
            assert param["widget_config"]["run_id_source"] == "dataset_loading"
            assert param["widget_config"]["choices_endpoint"].endswith("/user_id")

    @patch("app.api.routes_plugins.MLflowRunLoader")
    @patch("app.api.routes_plugins.PluginManager")
    def test_sort_by_tint_is_global_before_limit(
        self,
        mock_pm: MagicMock,
        mock_loader_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify sorting happens before limiting, so the page is the global top."""
        mock_pm.load.return_value = self._mock_plugin()
        mock_loader_cls.return_value = self._mock_loader()

        response = client.get(
            "/plugins/param-choices/steering/steering.sae.sae/user_id",
            params={"run_id": "r1", "limit": 2, "sort_by_tint": "true"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        # Counts are [5, 2, 9] → global top-2 by tint are rows 2 and 0.
        assert [o["value"] for o in data["options"]] == ["2", "0"]

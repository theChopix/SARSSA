"""Integration tests for /items/ endpoints.

Tests hit the real endpoints via FastAPI TestClient.  MLflow
artifact loading is mocked so tests don't require a live MLflow
server.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestGetEnrichedItems:
    """Tests for GET /items/enrich."""

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_returns_200(
        self,
        mock_load: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify the endpoint returns HTTP 200."""
        mock_load.return_value = {}
        response = client.get("/items/enrich", params={"run_id": "r1", "ids": "1"})
        assert response.status_code == 200

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_enriched_items_with_metadata(
        self,
        mock_load: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify items are enriched when metadata exists."""
        mock_load.return_value = {
            "10": {"title": "Movie A", "year": 2020},
            "20": {"title": "Movie B", "year": 2021},
        }
        response = client.get(
            "/items/enrich",
            params={"run_id": "r1", "ids": "10,20"},
        )
        data = response.json()

        assert data["metadata_available"] is True
        assert len(data["items"]) == 2
        assert data["items"][0]["id"] == "10"
        assert data["items"][0]["title"] == "Movie A"

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_fallback_when_metadata_missing(
        self,
        mock_load: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify graceful fallback with invalid/missing run."""
        mock_load.return_value = {}
        response = client.get(
            "/items/enrich",
            params={"run_id": "invalid_run", "ids": "42,99"},
        )
        data = response.json()

        assert data["metadata_available"] is False
        assert data["items"][0] == {"id": "42", "title": "42"}
        assert data["items"][1] == {"id": "99", "title": "99"}

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_empty_ids_returns_empty_list(
        self,
        mock_load: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify empty ids produces empty items list."""
        mock_load.return_value = {"1": {"title": "X"}}
        response = client.get(
            "/items/enrich",
            params={"run_id": "r1", "ids": ""},
        )
        data = response.json()

        assert data["items"] == []

    @patch("app.core.item_enrichment.item_enrichment.load_item_metadata")
    def test_response_has_expected_keys(
        self,
        mock_load: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify response contains items and metadata_available."""
        mock_load.return_value = {}
        response = client.get(
            "/items/enrich",
            params={"run_id": "r1", "ids": "1"},
        )
        data = response.json()

        assert "items" in data
        assert "metadata_available" in data


class TestGetStepArtifact:
    """Tests for GET /items/artifact."""

    @patch("app.core.item_enrichment.item_enrichment.MLflowRunLoader")
    def test_returns_artifact_json(
        self,
        mock_loader_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify artifact content is returned as JSON."""
        mock_loader = MagicMock()
        mock_loader.artifact_exists.return_value = True
        mock_loader.get_json_artifact.return_value = ["42", "107"]
        mock_loader_cls.return_value = mock_loader

        response = client.get(
            "/items/artifact",
            params={"run_id": "r1", "filename": "recs.json"},
        )

        assert response.status_code == 200
        assert response.json() == ["42", "107"]

    @patch("app.core.item_enrichment.item_enrichment.MLflowRunLoader")
    def test_returns_404_when_artifact_missing(
        self,
        mock_loader_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """Verify 404 when the artifact does not exist."""
        mock_loader = MagicMock()
        mock_loader.artifact_exists.return_value = False
        mock_loader_cls.return_value = mock_loader

        response = client.get(
            "/items/artifact",
            params={"run_id": "r1", "filename": "missing.json"},
        )

        assert response.status_code == 404

    def test_returns_422_without_required_params(
        self,
        client: TestClient,
    ) -> None:
        """Verify 422 when required query params are missing."""
        response = client.get("/items/artifact")
        assert response.status_code == 422

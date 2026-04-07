"""Integration tests for CORS middleware configuration."""

from fastapi.testclient import TestClient


class TestCorsMiddleware:
    """Tests for CORS header behaviour."""

    def test_allows_vite_origin(self, client: TestClient) -> None:
        """Verify CORS headers are set for the Vite dev server origin."""
        response = client.options(
            "/plugins/registry",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_rejects_unknown_origin(self, client: TestClient) -> None:
        """Verify CORS headers are absent for an unknown origin."""
        response = client.options(
            "/plugins/registry",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert "access-control-allow-origin" not in response.headers

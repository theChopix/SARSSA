"""Shared fixtures for app integration tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    """Provide a FastAPI TestClient for integration tests.

    Returns:
        TestClient: A test client bound to the application.
    """
    return TestClient(app)

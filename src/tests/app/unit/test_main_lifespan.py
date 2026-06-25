"""Unit tests for the FastAPI lifespan run-recovery wiring."""

from unittest.mock import MagicMock, call, patch

import pytest

from app.main import lifespan


@pytest.mark.asyncio
async def test_lifespan_reconciles_on_startup_and_shutdown() -> None:
    """Orphaned runs are swept on startup, then again on graceful shutdown."""
    with patch("app.main.fail_orphaned_runs") as mock_fail:
        async with lifespan(MagicMock()):
            # Inside the context: startup has run, shutdown has not yet.
            mock_fail.assert_called_once_with("orphaned_at_startup")

        assert mock_fail.call_args_list == [
            call("orphaned_at_startup"),
            call("terminated_at_shutdown"),
        ]

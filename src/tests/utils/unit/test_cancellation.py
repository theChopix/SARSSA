"""Unit tests for utils.cancellation."""

import threading

import pytest

from utils.cancellation import CancellationToken, NullCancellationToken, StepAborted


class TestCancellationToken:
    """Tests for CancellationToken."""

    def test_not_cancelled_when_event_clear(self) -> None:
        """Verify cancelled() is False while the event is unset."""
        token = CancellationToken(threading.Event())
        assert token.cancelled() is False

    def test_cancelled_when_event_set(self) -> None:
        """Verify cancelled() is True once the event is set."""
        event = threading.Event()
        event.set()
        assert CancellationToken(event).cancelled() is True

    def test_reflects_event_changes(self) -> None:
        """Verify the token tracks the underlying event live."""
        event = threading.Event()
        token = CancellationToken(event)
        assert token.cancelled() is False
        event.set()
        assert token.cancelled() is True

    def test_raise_if_cancelled_noop_when_clear(self) -> None:
        """Verify raise_if_cancelled() does nothing while unset."""
        CancellationToken(threading.Event()).raise_if_cancelled()

    def test_raise_if_cancelled_raises_when_set(self) -> None:
        """Verify raise_if_cancelled() raises StepAborted once set."""
        event = threading.Event()
        event.set()
        with pytest.raises(StepAborted):
            CancellationToken(event).raise_if_cancelled()


class TestNullCancellationToken:
    """Tests for NullCancellationToken."""

    def test_never_cancelled(self) -> None:
        """Verify the null token is always uncancelled."""
        assert NullCancellationToken().cancelled() is False

    def test_raise_if_cancelled_is_noop(self) -> None:
        """Verify the null token never raises."""
        NullCancellationToken().raise_if_cancelled()

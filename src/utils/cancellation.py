"""Cooperative cancellation token that plugins use to abort mid-step."""

import threading


class StepAborted(Exception):
    """Raised by a plugin when it observes an abort request.

    Distinct from ordinary errors so the pipeline worker can tell a
    user-requested abort apart from a genuine step failure.
    """


class CancellationToken:
    """A read-only view over an abort flag, injected into plugins.

    Wraps a :class:`threading.Event` set by the cancel endpoint. Plugins
    that opt in call :meth:`raise_if_cancelled` at a safe boundary (e.g.
    the top of a training epoch/batch) to stop promptly. The check is a
    lock-free flag read — the same GIL-atomic property the notifier relies
    on — so it is effectively free even in a hot loop.
    """

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def cancelled(self) -> bool:
        """Return whether an abort has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`StepAborted` if an abort has been requested.

        Raises:
            StepAborted: If the abort flag is set.
        """
        if self.cancelled():
            raise StepAborted()


class NullCancellationToken(CancellationToken):
    """No-op token used when no abort signal is available.

    Plugins that consult ``self.cancellation`` behave identically whether
    executed inside the pipeline (a real :class:`CancellationToken`) or
    standalone in tests/scripts — the flag is never set.
    """

    def __init__(self) -> None:
        super().__init__(threading.Event())

    def cancelled(self) -> bool:
        """Always ``False`` — nothing can request an abort here."""
        return False

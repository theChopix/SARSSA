"""Lightweight notifier that plugins use to push messages to the UI."""

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class NotificationMessage:
    """Single notification payload.

    Attributes:
        timestamp: Unix epoch seconds when the message was created.
        level: Severity / type — ``"info"``, ``"warning"``, ``"error"``,
            ``"success"``.
        text: Human-readable message body.
    """

    timestamp: float
    level: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict.

        Returns:
            dict[str, Any]: Dict with keys ``timestamp``, ``level``, ``text``.
        """
        return {"timestamp": self.timestamp, "level": self.level, "text": self.text}


class PluginNotifier:
    """Accumulates notification messages for the frontend.

    Plugins call :meth:`info`, :meth:`warning`, :meth:`success`, or
    :meth:`error` to record a message.  The pipeline worker shares
    ``messages`` with ``TaskState.messages`` (same list object) so the
    polling endpoint sees new entries immediately — no copying needed.

    Thread safety: Python's GIL guarantees that ``list.append()`` is
    atomic.  Since the worker thread only appends and the polling
    endpoint only reads (via a ``list()`` snapshot), no explicit lock
    is required.

    Attributes:
        messages: Ordered list of notification dicts emitted so far.
    """

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def _append(self, level: str, text: str) -> None:
        """Create and store a notification message.

        Args:
            level: Message severity.
            text: Human-readable message body.
        """
        msg = NotificationMessage(timestamp=time.time(), level=level, text=text)
        self.messages.append(msg.to_dict())

    def info(self, text: str) -> None:
        """Record an informational message.

        Args:
            text: Message body.
        """
        self._append("info", text)

    def warning(self, text: str) -> None:
        """Record a warning message.

        Args:
            text: Message body.
        """
        self._append("warning", text)

    def success(self, text: str) -> None:
        """Record a success message.

        Args:
            text: Message body.
        """
        self._append("success", text)

    def error(self, text: str) -> None:
        """Record an error message.

        Args:
            text: Message body.
        """
        self._append("error", text)


class NullNotifier(PluginNotifier):
    """No-op notifier used when no UI polling is available.

    Plugins that call ``self.notifier.info(...)`` work identically
    whether executed inside the pipeline (real ``PluginNotifier``) or
    standalone in tests / scripts (``NullNotifier``).  All messages
    are silently discarded.
    """

    def _append(self, level: str, text: str) -> None:
        """Discard the message silently.

        Args:
            level: Message severity (ignored).
            text: Message body (ignored).
        """

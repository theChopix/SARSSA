"""Unit tests for utils.plugin_notifier."""

from unittest.mock import patch

from utils.plugin_notifier import NotificationMessage, NullNotifier, PluginNotifier

# ── NotificationMessage ───────────────────────────────────────────────


class TestNotificationMessage:
    """Tests for NotificationMessage."""

    def test_to_dict_contains_all_keys(self) -> None:
        """Verify to_dict() returns timestamp, level, and text."""
        msg = NotificationMessage(timestamp=1000.0, level="info", text="hello")

        result = msg.to_dict()

        assert result == {"timestamp": 1000.0, "level": "info", "text": "hello"}

    def test_to_dict_preserves_values(self) -> None:
        """Verify values are not mutated by to_dict()."""
        msg = NotificationMessage(timestamp=42.5, level="error", text="boom")

        result = msg.to_dict()

        assert result["timestamp"] == 42.5
        assert result["level"] == "error"
        assert result["text"] == "boom"


# ── PluginNotifier ────────────────────────────────────────────────────


class TestPluginNotifier:
    """Tests for PluginNotifier."""

    def test_starts_empty(self) -> None:
        """Verify messages list is empty on construction."""
        notifier = PluginNotifier()

        assert notifier.messages == []

    def test_info_appends_one_message(self) -> None:
        """Verify info() appends exactly one dict to messages."""
        notifier = PluginNotifier()

        notifier.info("test message")

        assert len(notifier.messages) == 1

    def test_info_sets_correct_level(self) -> None:
        """Verify info() produces level='info'."""
        notifier = PluginNotifier()

        notifier.info("msg")

        assert notifier.messages[0]["level"] == "info"

    def test_warning_sets_correct_level(self) -> None:
        """Verify warning() produces level='warning'."""
        notifier = PluginNotifier()

        notifier.warning("msg")

        assert notifier.messages[0]["level"] == "warning"

    def test_success_sets_correct_level(self) -> None:
        """Verify success() produces level='success'."""
        notifier = PluginNotifier()

        notifier.success("msg")

        assert notifier.messages[0]["level"] == "success"

    def test_error_sets_correct_level(self) -> None:
        """Verify error() produces level='error'."""
        notifier = PluginNotifier()

        notifier.error("msg")

        assert notifier.messages[0]["level"] == "error"

    def test_message_text_is_preserved(self) -> None:
        """Verify the text field matches what was passed."""
        notifier = PluginNotifier()

        notifier.info("Epoch 1/100 — loss 0.5")

        assert notifier.messages[0]["text"] == "Epoch 1/100 — loss 0.5"

    def test_message_has_timestamp(self) -> None:
        """Verify each message has a numeric timestamp."""
        notifier = PluginNotifier()

        notifier.info("msg")

        assert isinstance(notifier.messages[0]["timestamp"], float)

    def test_timestamp_uses_time_time(self) -> None:
        """Verify timestamp is sourced from time.time()."""
        notifier = PluginNotifier()

        with patch("utils.plugin_notifier.time.time", return_value=9999.0):
            notifier.info("msg")

        assert notifier.messages[0]["timestamp"] == 9999.0

    def test_multiple_calls_append_in_order(self) -> None:
        """Verify messages accumulate in call order."""
        notifier = PluginNotifier()

        notifier.info("first")
        notifier.warning("second")
        notifier.error("third")

        assert len(notifier.messages) == 3
        assert notifier.messages[0]["text"] == "first"
        assert notifier.messages[1]["text"] == "second"
        assert notifier.messages[2]["text"] == "third"

    def test_messages_list_is_shared_by_reference(self) -> None:
        """Verify an external reference to messages sees new appends."""
        notifier = PluginNotifier()
        shared = notifier.messages  # same list object

        notifier.info("live update")

        assert shared[0]["text"] == "live update"

    def test_each_message_dict_has_exactly_three_keys(self) -> None:
        """Verify message dicts contain only timestamp, level, text."""
        notifier = PluginNotifier()

        notifier.success("done")

        assert set(notifier.messages[0].keys()) == {"timestamp", "level", "text"}


# ── NullNotifier ──────────────────────────────────────────────────────


class TestNullNotifier:
    """Tests for NullNotifier."""

    def test_messages_stays_empty_after_info(self) -> None:
        """Verify info() is a no-op — messages list remains empty."""
        notifier = NullNotifier()

        notifier.info("ignored")

        assert notifier.messages == []

    def test_messages_stays_empty_after_warning(self) -> None:
        """Verify warning() is a no-op."""
        notifier = NullNotifier()

        notifier.warning("ignored")

        assert notifier.messages == []

    def test_messages_stays_empty_after_success(self) -> None:
        """Verify success() is a no-op."""
        notifier = NullNotifier()

        notifier.success("ignored")

        assert notifier.messages == []

    def test_messages_stays_empty_after_error(self) -> None:
        """Verify error() is a no-op."""
        notifier = NullNotifier()

        notifier.error("ignored")

        assert notifier.messages == []

    def test_null_notifier_is_plugin_notifier_subclass(self) -> None:
        """Verify NullNotifier is a subclass of PluginNotifier."""
        assert issubclass(NullNotifier, PluginNotifier)

    def test_null_notifier_satisfies_same_interface(self) -> None:
        """Verify NullNotifier has all four public methods."""
        notifier = NullNotifier()

        notifier.info("x")
        notifier.warning("x")
        notifier.success("x")
        notifier.error("x")

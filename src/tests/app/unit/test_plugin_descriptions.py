"""Unit tests for per-plugin (implementation) descriptions.

Mirrors test_plugin_categories.py: asserts against the real shipped
plugins rather than mocks, so a contributor who adds a plugin without
a description gets a clear failure naming the offending module.
"""

from app.core.plugin_discovery.plugin_registry import get_plugin_registry


class TestPluginDescriptions:
    """Tests for the per-implementation description field."""

    def test_every_implementation_has_a_description(self) -> None:
        """Verify every shipped plugin declares a non-empty description."""
        registry = get_plugin_registry()
        for category_name, entry in registry.items():
            for impl in entry.implementations:
                assert impl.description, (
                    f"Plugin '{impl.plugin_name}' in category "
                    f"'{category_name}' is missing a description"
                )

"""Unit tests for app.core.plugin_categories."""

from pathlib import Path

from app.config.config import PLUGIN_CATEGORIES

PLUGINS_DIR = Path(__file__).resolve().parents[3] / "plugins"


class TestPluginCategoryOrdering:
    """Tests for category execution order values."""

    def test_orders_start_at_zero(self) -> None:
        """Verify the lowest order value is 0."""
        orders = [cat.order for cat in PLUGIN_CATEGORIES.values()]
        assert min(orders) == 0

    def test_orders_are_sequential(self) -> None:
        """Verify orders form a contiguous 0..N-1 range."""
        orders = sorted(cat.order for cat in PLUGIN_CATEGORIES.values())
        assert orders == list(range(len(PLUGIN_CATEGORIES)))


class TestPluginCategoryKeys:
    """Tests that category keys match plugin folders on disk."""

    def test_all_keys_have_matching_plugin_folder(self) -> None:
        """Verify every category key maps to a folder in src/plugins/."""
        for category_name in PLUGIN_CATEGORIES:
            folder = PLUGINS_DIR / category_name
            assert folder.is_dir(), (
                f"No plugin folder found for category '{category_name}' at {folder}"
            )

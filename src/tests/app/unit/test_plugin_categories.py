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


class TestHasVisualResults:
    """Tests for the has_visual_results category flag."""

    def test_steering_has_visual_results(self) -> None:
        """Verify steering category has visual results enabled."""
        assert PLUGIN_CATEGORIES["steering"].has_visual_results is True

    def test_inspection_has_visual_results(self) -> None:
        """Verify inspection category has visual results enabled."""
        assert PLUGIN_CATEGORIES["inspection"].has_visual_results is True

    def test_labeling_evaluation_has_visual_results(self) -> None:
        """Verify labeling_evaluation category has visual results enabled."""
        assert PLUGIN_CATEGORIES["labeling_evaluation"].has_visual_results is True

    def test_other_categories_default_to_false(self) -> None:
        """Verify categories without explicit flag default to False."""
        visual_categories = {"steering", "inspection", "labeling_evaluation"}
        for name, info in PLUGIN_CATEGORIES.items():
            if name not in visual_categories:
                assert info.has_visual_results is False, (
                    f"Category '{name}' should not have visual results"
                )

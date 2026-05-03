"""Integration tests for app.core.plugin_registry.

These tests run against the real src/plugins/ directory to verify
that all plugin implementations are correctly discovered and their
run() signatures are properly introspected.

Tests are written to be resilient to plugin changes — they discover
plugins dynamically rather than hard-coding specific names or params.
"""

from app.config.config import PLUGIN_CATEGORIES
from app.core.plugin_discovery.plugin_registry import (
    PLUGINS_DIR,
    _find_plugin_modules,
    get_plugin_registry,
)
from app.models.plugin import (
    CategoryRegistryEntry,
    ImplementationInfo,
    ParameterInfo,
)


class TestGetPluginRegistryIntegration:
    """Tests for get_plugin_registry against the real plugin folder."""

    def test_returns_entry_for_every_category(self) -> None:
        """Verify registry contains all categories from PLUGIN_CATEGORIES."""
        registry = get_plugin_registry()
        assert set(registry.keys()) == set(PLUGIN_CATEGORIES.keys())

    def test_every_category_has_at_least_one_implementation(self) -> None:
        """Verify no category has an empty implementations list."""
        registry = get_plugin_registry()
        for name, entry in registry.items():
            assert len(entry.implementations) > 0, f"Category '{name}' has no implementations"

    def test_all_entries_are_category_registry_entry(self) -> None:
        """Verify each registry value is a CategoryRegistryEntry."""
        registry = get_plugin_registry()
        for entry in registry.values():
            assert isinstance(entry, CategoryRegistryEntry)

    def test_discovered_plugin_names_match_filesystem(self) -> None:
        """Verify discovered plugin_names match _find_plugin_modules output."""
        registry = get_plugin_registry()
        for category_name, entry in registry.items():
            expected_modules = set(_find_plugin_modules(PLUGINS_DIR / category_name, category_name))
            actual_modules = {impl.plugin_name for impl in entry.implementations}
            assert actual_modules == expected_modules, (
                f"Mismatch in '{category_name}': expected {expected_modules}, got {actual_modules}"
            )


class TestImplementationInfoIntegration:
    """Tests for discovered implementation metadata."""

    def test_all_implementations_are_implementation_info(self) -> None:
        """Verify all discovered items are typed ImplementationInfo."""
        registry = get_plugin_registry()
        for entry in registry.values():
            for impl in entry.implementations:
                assert isinstance(impl, ImplementationInfo)

    def test_all_implementations_have_non_empty_display_name(self) -> None:
        """Verify every implementation has a non-empty display_name."""
        registry = get_plugin_registry()
        for entry in registry.values():
            for impl in entry.implementations:
                assert impl.display_name, f"{impl.plugin_name} has empty display_name"


class TestKindIntegration:
    """Tests for kind derivation against the real plugin folder."""

    def test_labeling_evaluation_single_plugins_have_single_kind(self) -> None:
        """Verify every labeling_evaluation/single/* plugin gets kind="single"."""
        registry = get_plugin_registry()
        entry = registry["labeling_evaluation"]
        single_impls = [impl for impl in entry.implementations if ".single." in impl.plugin_name]
        assert single_impls, "expected at least one single plugin under labeling_evaluation"
        for impl in single_impls:
            assert impl.kind == "single", (
                f"{impl.plugin_name} expected kind='single' but got {impl.kind}"
            )

    def test_inspection_flat_plugin_has_no_kind(self) -> None:
        """Verify the un-migrated inspection.sae_inspection has kind=None."""
        registry = get_plugin_registry()
        entry = registry["inspection"]
        flat_impls = [
            impl
            for impl in entry.implementations
            if not (".single." in impl.plugin_name or ".compare." in impl.plugin_name)
        ]
        for impl in flat_impls:
            assert impl.kind is None, f"{impl.plugin_name} unexpectedly has kind={impl.kind}"

    def test_kind_only_set_for_known_subfolders(self) -> None:
        """Verify no plugin reports a kind outside the allowed literals."""
        registry = get_plugin_registry()
        for entry in registry.values():
            for impl in entry.implementations:
                assert impl.kind in (None, "single", "compare"), (
                    f"{impl.plugin_name} has invalid kind={impl.kind}"
                )


class TestExtractParametersIntegration:
    """Tests for parameter extraction across all real plugins."""

    def test_all_params_are_parameter_info(self) -> None:
        """Verify all extracted params are ParameterInfo instances."""
        registry = get_plugin_registry()
        for entry in registry.values():
            for impl in entry.implementations:
                for param in impl.params:
                    assert isinstance(param, ParameterInfo), (
                        f"{impl.plugin_name} param is not ParameterInfo"
                    )

    def test_no_plugin_exposes_self_or_context(self) -> None:
        """Verify self and context are excluded from all plugins."""
        registry = get_plugin_registry()
        for entry in registry.values():
            for impl in entry.implementations:
                names = {p.name for p in impl.params}
                assert "self" not in names, f"{impl.plugin_name} leaks 'self'"
                assert "context" not in names, f"{impl.plugin_name} leaks 'context'"

    def test_every_param_has_a_type(self) -> None:
        """Verify every parameter has a non-empty type string."""
        registry = get_plugin_registry()
        for entry in registry.values():
            for impl in entry.implementations:
                for param in impl.params:
                    assert param.type, f"{impl.plugin_name}.{param.name} has empty type"

    def test_required_params_have_no_default(self) -> None:
        """Verify required params have default set to None."""
        registry = get_plugin_registry()
        for entry in registry.values():
            for impl in entry.implementations:
                for param in impl.params:
                    if param.required:
                        assert param.default is None, (
                            f"{impl.plugin_name}.{param.name} is required "
                            f"but has default={param.default}"
                        )

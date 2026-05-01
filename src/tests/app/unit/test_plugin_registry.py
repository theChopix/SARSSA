"""Unit tests for app.core.plugin_registry.

All tests use mock filesystem structures and mock plugin instances
to avoid coupling to the real src/plugins/ directory.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.core.plugin_discovery.plugin_registry import (
    _convert_display_spec,
    _extract_parameters_from_instance,
    _find_plugin_modules,
    _make_display_name,
    get_plugin_registry,
)
from app.models.plugin import (
    ArtifactDisplayModel,
    CategoryInfo,
    CategoryRegistryEntry,
    CategoryType,
    ItemRowsDisplayModel,
    ParameterInfo,
)
from plugins.plugin_interface import (
    ArtifactDisplaySpec,
    ArtifactFileSpec,
    DisplayRowSpec,
    ItemRowsDisplaySpec,
    PluginIOSpec,
)

# ── Helpers for building mock plugin directory trees ──────────────


def _build_flat_category(tmp_path: Path, category: str, impls: list[str]) -> Path:
    """Create a flat category tree: ``<category>/<impl>/<impl>.py``.

    Args:
        tmp_path: Pytest temporary directory.
        category: Category folder name.
        impls: List of implementation folder/file names.

    Returns:
        Path: The category directory.
    """
    cat_dir = tmp_path / category
    cat_dir.mkdir()
    for impl in impls:
        impl_dir = cat_dir / impl
        impl_dir.mkdir()
        (impl_dir / f"{impl}.py").touch()
    return cat_dir


def _build_nested_category(
    tmp_path: Path,
    category: str,
    subtypes: dict[str, list[str]],
) -> Path:
    """Create a nested category tree: ``<category>/<sub>/<impl>/<impl>.py``.

    Args:
        tmp_path: Pytest temporary directory.
        category: Category folder name.
        subtypes: Mapping of subtype dirs to implementation names.

    Returns:
        Path: The category directory.
    """
    cat_dir = tmp_path / category
    cat_dir.mkdir()
    for subtype, impls in subtypes.items():
        sub_dir = cat_dir / subtype
        sub_dir.mkdir()
        for impl in impls:
            impl_dir = sub_dir / impl
            impl_dir.mkdir()
            (impl_dir / f"{impl}.py").touch()
    return cat_dir


def _make_mock_plugin(
    params: dict[str, tuple[type, Any]],
    plugin_name: str | None = None,
    display: ItemRowsDisplaySpec | ArtifactDisplaySpec | None = None,
) -> MagicMock:
    """Create a mock plugin whose run() has the given signature params.

    Args:
        params: Mapping of param name to (type, default).
            Use ``inspect.Parameter.empty`` for required params.
        plugin_name: Optional custom display name for the plugin.
            Mirrors ``BasePlugin.name``.
        display: Optional display spec to attach to
            ``io_spec.display``.

    Returns:
        MagicMock: A mock with a ``run`` method with proper signature.
    """
    import inspect

    parameters = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
    for name, (ann, default) in params.items():
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=ann,
                default=default,
            )
        )

    sig = inspect.Signature(parameters)
    mock_plugin = MagicMock()
    mock_plugin.name = plugin_name
    mock_plugin.run = MagicMock()
    mock_plugin.run.__signature__ = sig
    mock_plugin.io_spec = PluginIOSpec(display=display)
    return mock_plugin


# ── Tests ─────────────────────────────────────────────────────────


class TestFindPluginModules:
    """Tests for _find_plugin_modules with mock directory trees."""

    def test_flat_structure(self, tmp_path: Path) -> None:
        """Verify discovery of plugins in a flat category layout."""
        cat_dir = _build_flat_category(tmp_path, "my_cat", ["alpha", "beta"])
        modules = _find_plugin_modules(cat_dir, "my_cat")
        assert modules == ["my_cat.alpha.alpha", "my_cat.beta.beta"]

    def test_nested_structure(self, tmp_path: Path) -> None:
        """Verify discovery of plugins in a nested category layout."""
        cat_dir = _build_nested_category(
            tmp_path, "my_cat", {"sub_a": ["impl_x"], "sub_b": ["impl_y"]}
        )
        modules = _find_plugin_modules(cat_dir, "my_cat")
        assert "my_cat.sub_a.impl_x.impl_x" in modules
        assert "my_cat.sub_b.impl_y.impl_y" in modules

    def test_skips_dunder_directories(self, tmp_path: Path) -> None:
        """Verify __pycache__ and similar dirs are ignored."""
        cat_dir = _build_flat_category(tmp_path, "my_cat", ["good_impl"])
        (cat_dir / "__pycache__").mkdir()
        (cat_dir / ".hidden").mkdir()
        modules = _find_plugin_modules(cat_dir, "my_cat")
        assert modules == ["my_cat.good_impl.good_impl"]

    def test_empty_category_returns_empty(self, tmp_path: Path) -> None:
        """Verify an empty category directory yields no modules."""
        cat_dir = tmp_path / "empty_cat"
        cat_dir.mkdir()
        modules = _find_plugin_modules(cat_dir, "empty_cat")
        assert modules == []

    def test_dir_without_matching_py_is_traversed(self, tmp_path: Path) -> None:
        """Verify intermediate dirs without a matching .py are traversed."""
        cat_dir = _build_nested_category(tmp_path, "my_cat", {"intermediate": ["leaf"]})
        modules = _find_plugin_modules(cat_dir, "my_cat")
        assert modules == ["my_cat.intermediate.leaf.leaf"]


class TestExtractParametersFromInstance:
    """Tests for _extract_parameters_from_instance with mock plugins."""

    def test_extracts_optional_params(self) -> None:
        """Verify params with defaults are marked as not required."""
        mock_plugin = _make_mock_plugin({"batch_size": (int, 64), "lr": (float, 0.01)})
        params = _extract_parameters_from_instance(mock_plugin)

        assert len(params) == 2
        for p in params:
            assert not p.required
        assert params[0].name == "batch_size"
        assert params[0].default == 64
        assert params[0].type == "int"

    def test_extracts_required_params(self) -> None:
        """Verify params without defaults are marked as required."""
        import inspect

        mock_plugin = _make_mock_plugin({"tag": (str, inspect.Parameter.empty)})
        params = _extract_parameters_from_instance(mock_plugin)

        assert len(params) == 1
        assert params[0].required is True
        assert params[0].default is None

    def test_excludes_self(self) -> None:
        """Verify self is filtered out."""
        mock_plugin = _make_mock_plugin({"epochs": (int, 10)})
        params = _extract_parameters_from_instance(mock_plugin)
        names = {p.name for p in params}
        assert "self" not in names

    def test_unannotated_param_defaults_to_str(self) -> None:
        """Verify params without type annotations default to 'str'."""
        import inspect

        mock_plugin = _make_mock_plugin({})
        # Manually add unannotated param
        sig = inspect.signature(mock_plugin.run)
        new_params = list(sig.parameters.values()) + [
            inspect.Parameter(
                "mystery",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default="hello",
            )
        ]
        mock_plugin.run.__signature__ = sig.replace(parameters=new_params)

        params = _extract_parameters_from_instance(mock_plugin)
        assert params[0].type == "str"

    def test_returns_parameter_info_instances(self) -> None:
        """Verify returned items are ParameterInfo models."""
        mock_plugin = _make_mock_plugin({"x": (int, 1)})
        params = _extract_parameters_from_instance(mock_plugin)
        for p in params:
            assert isinstance(p, ParameterInfo)


class TestMakeDisplayName:
    """Tests for _make_display_name (pure function, no mocking needed)."""

    @pytest.mark.parametrize(
        ("module_path", "expected"),
        [
            ("cat.my_impl.my_impl", "My Impl"),
            ("cat.sub.deep_impl.deep_impl", "Deep Impl"),
            ("cat.single_word.single_word", "Single Word"),
        ],
    )
    def test_produces_expected_display_name(self, module_path: str, expected: str) -> None:
        """Verify display name derivation from module path.

        Args:
            module_path: Dotted plugin module path.
            expected: Expected human-readable display name.
        """
        assert _make_display_name(module_path) == expected


class TestDiscoverImplementationsDisplayName:
    """Tests for custom name preference in _discover_implementations."""

    @patch("app.core.plugin_discovery.plugin_registry._find_plugin_modules")
    @patch("app.core.plugin_discovery.plugin_registry.PluginManager")
    def test_uses_custom_name_when_set(
        self,
        mock_pm: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Verify plugin.name is used when the plugin defines it."""
        from app.core.plugin_discovery.plugin_registry import (
            _discover_implementations,
        )

        mock_find.return_value = ["cat.sae_trainer.sae_trainer"]
        mock_pm.load.return_value = _make_mock_plugin(
            {"epochs": (int, 10)},
            plugin_name="SAE Trainer",
        )

        impls = _discover_implementations("cat")

        assert len(impls) == 1
        assert impls[0].display_name == "SAE Trainer"

    @patch("app.core.plugin_discovery.plugin_registry._find_plugin_modules")
    @patch("app.core.plugin_discovery.plugin_registry.PluginManager")
    def test_falls_back_to_auto_derived_name(
        self,
        mock_pm: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Verify auto-derived name is used when plugin.name is None."""
        from app.core.plugin_discovery.plugin_registry import (
            _discover_implementations,
        )

        mock_find.return_value = ["cat.my_impl.my_impl"]
        mock_pm.load.return_value = _make_mock_plugin(
            {"x": (int, 1)},
            plugin_name=None,
        )

        impls = _discover_implementations("cat")

        assert len(impls) == 1
        assert impls[0].display_name == "My Impl"

    @patch("app.core.plugin_discovery.plugin_registry._find_plugin_modules")
    @patch("app.core.plugin_discovery.plugin_registry.PluginManager")
    def test_empty_string_name_falls_back(
        self,
        mock_pm: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Verify empty string name falls back to auto-derived name."""
        from app.core.plugin_discovery.plugin_registry import (
            _discover_implementations,
        )

        mock_find.return_value = ["cat.my_impl.my_impl"]
        mock_pm.load.return_value = _make_mock_plugin(
            {"x": (int, 1)},
            plugin_name="",
        )

        impls = _discover_implementations("cat")

        assert impls[0].display_name == "My Impl"


class TestConvertDisplaySpec:
    """Tests for _convert_display_spec."""

    def test_returns_none_for_none(self) -> None:
        """Verify None input produces None output."""
        assert _convert_display_spec(None) is None

    def test_converts_item_rows_display_spec(self) -> None:
        """Verify ItemRowsDisplaySpec is converted to Pydantic model."""
        spec = ItemRowsDisplaySpec(
            rows=[
                DisplayRowSpec("top_k", "Top Items"),
                DisplayRowSpec("recs", "Recommendations"),
            ],
        )
        result = _convert_display_spec(spec)
        assert isinstance(result, ItemRowsDisplayModel)
        assert result.type == "item_rows"
        assert len(result.rows) == 2
        assert result.rows[0].key == "top_k"
        assert result.rows[0].label == "Top Items"
        assert result.rows[1].key == "recs"

    def test_converts_empty_rows(self) -> None:
        """Verify ItemRowsDisplaySpec with no rows converts correctly."""
        spec = ItemRowsDisplaySpec(rows=[])
        result = _convert_display_spec(spec)
        assert result is not None
        assert isinstance(result, ItemRowsDisplayModel)
        assert result.rows == []

    def test_converts_artifact_display_spec(self) -> None:
        """Verify ArtifactDisplaySpec is converted to Pydantic model."""
        spec = ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec("map.html", "Map", "text/html"),
            ],
        )
        result = _convert_display_spec(spec)
        assert isinstance(result, ArtifactDisplayModel)
        assert result.type == "artifact"
        assert len(result.files) == 1
        assert result.files[0].filename == "map.html"
        assert result.files[0].label == "Map"
        assert result.files[0].content_type == "text/html"


class TestDiscoverImplementationsDisplay:
    """Tests for display field propagation in _discover_implementations."""

    @patch("app.core.plugin_discovery.plugin_registry._find_plugin_modules")
    @patch("app.core.plugin_discovery.plugin_registry.PluginManager")
    def test_plugin_with_display_spec(
        self,
        mock_pm: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Verify display spec is included in ImplementationInfo."""
        from app.core.plugin_discovery.plugin_registry import (
            _discover_implementations,
        )

        display = ItemRowsDisplaySpec(
            rows=[DisplayRowSpec("items", "Items")],
        )
        mock_find.return_value = ["cat.my_impl.my_impl"]
        mock_pm.load.return_value = _make_mock_plugin(
            {"k": (int, 10)},
            display=display,
        )

        impls = _discover_implementations("cat")
        assert impls[0].display is not None
        assert isinstance(impls[0].display, ItemRowsDisplayModel)
        assert impls[0].display.type == "item_rows"
        assert len(impls[0].display.rows) == 1
        assert impls[0].display.rows[0].key == "items"

    @patch("app.core.plugin_discovery.plugin_registry._find_plugin_modules")
    @patch("app.core.plugin_discovery.plugin_registry.PluginManager")
    def test_plugin_without_display_spec(
        self,
        mock_pm: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Verify display is None when plugin has no display spec."""
        from app.core.plugin_discovery.plugin_registry import (
            _discover_implementations,
        )

        mock_find.return_value = ["cat.my_impl.my_impl"]
        mock_pm.load.return_value = _make_mock_plugin(
            {"k": (int, 10)},
            display=None,
        )

        impls = _discover_implementations("cat")
        assert impls[0].display is None


class TestGetPluginRegistry:
    """Tests for get_plugin_registry with mocked internals."""

    @patch("app.core.plugin_discovery.plugin_registry._discover_implementations")
    @patch(
        "app.core.plugin_discovery.plugin_registry.PLUGIN_CATEGORIES",
        {
            "cat_a": CategoryInfo(order=0, type=CategoryType.ONE_TIME, display_name="Cat A"),
            "cat_b": CategoryInfo(order=1, type=CategoryType.MULTI_RUN, display_name="Cat B"),
        },
    )
    def test_returns_entry_for_every_category(self, mock_discover: MagicMock) -> None:
        """Verify registry contains all mocked categories."""
        mock_discover.return_value = []
        registry = get_plugin_registry()
        assert set(registry.keys()) == {"cat_a", "cat_b"}

    @patch("app.core.plugin_discovery.plugin_registry._discover_implementations")
    @patch(
        "app.core.plugin_discovery.plugin_registry.PLUGIN_CATEGORIES",
        {
            "cat_a": CategoryInfo(order=0, type=CategoryType.ONE_TIME, display_name="Cat A"),
        },
    )
    def test_entries_are_category_registry_entry(self, mock_discover: MagicMock) -> None:
        """Verify each value is a CategoryRegistryEntry."""
        mock_discover.return_value = []
        registry = get_plugin_registry()
        for entry in registry.values():
            assert isinstance(entry, CategoryRegistryEntry)

    @patch("app.core.plugin_discovery.plugin_registry._discover_implementations")
    @patch(
        "app.core.plugin_discovery.plugin_registry.PLUGIN_CATEGORIES",
        {
            "cat_a": CategoryInfo(order=0, type=CategoryType.ONE_TIME, display_name="Cat A"),
        },
    )
    def test_passes_category_info_through(self, mock_discover: MagicMock) -> None:
        """Verify the category_info field matches the source."""
        mock_discover.return_value = []
        registry = get_plugin_registry()
        assert registry["cat_a"].category_info.display_name == "Cat A"

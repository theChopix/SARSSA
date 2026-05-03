"""Unit tests for app.core.plugin_registry.

All tests use mock filesystem structures and mock plugin instances
to avoid coupling to the real src/plugins/ directory.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.core.plugin_discovery.plugin_registry import (
    _build_ui_hint_map,
    _convert_display_spec,
    _derive_kind,
    _extract_parameters_from_instance,
    _find_plugin_modules,
    _make_display_name,
    _resolve_widget,
    get_plugin_registry,
)
from app.models.plugin import (
    ArtifactDisplayModel,
    CategoryInfo,
    CategoryRegistryEntry,
    CategoryType,
    ItemRowsDisplayModel,
    ParameterInfo,
    WidgetConfig,
)
from plugins.plugin_interface import (
    ArtifactDisplaySpec,
    ArtifactFileSpec,
    DisplayRowSpec,
    DynamicDropdownHint,
    ItemRowsDisplaySpec,
    ParamUIHint,
    PluginIOSpec,
    SliderHint,
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
    param_ui_hints: list[ParamUIHint] | None = None,
) -> MagicMock:
    """Create a mock plugin whose run() has the given signature params.

    Args:
        params: Mapping of param name to (type, default).
            Use ``inspect.Parameter.empty`` for required params.
        plugin_name: Optional custom display name for the plugin.
            Mirrors ``BasePlugin.name``.
        display: Optional display spec to attach to
            ``io_spec.display``.
        param_ui_hints: Optional list of UI hints to attach to
            ``io_spec.param_ui_hints``.

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
    mock_plugin.io_spec = PluginIOSpec(
        display=display,
        param_ui_hints=param_ui_hints or [],
    )
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


class TestDeriveKind:
    """Tests for _derive_kind (pure function, no mocking needed)."""

    @pytest.mark.parametrize(
        ("module_path", "category", "expected"),
        [
            (
                "labeling_evaluation.single.dendrogram.dendrogram",
                "labeling_evaluation",
                "single",
            ),
            (
                "labeling_evaluation.compare.diff.diff",
                "labeling_evaluation",
                "compare",
            ),
            (
                "inspection.single.sae_inspection.sae_inspection",
                "inspection",
                "single",
            ),
            (
                "inspection.compare.sae_inspection.sae_inspection",
                "inspection",
                "compare",
            ),
        ],
    )
    def test_returns_kind_when_subfolder_matches(
        self,
        module_path: str,
        category: str,
        expected: str,
    ) -> None:
        """Verify single/compare subfolders are recognised.

        Args:
            module_path: Dotted plugin module path.
            category: Category key the module belongs to.
            expected: Expected kind value.
        """
        assert _derive_kind(module_path, category) == expected

    @pytest.mark.parametrize(
        ("module_path", "category"),
        [
            (
                "dataset_loading.movieLens_loader.movieLens_loader",
                "dataset_loading",
            ),
            ("training_sae.sae_trainer.sae_trainer", "training_sae"),
            ("neuron_labeling.tf_idf.tf_idf", "neuron_labeling"),
        ],
    )
    def test_returns_none_for_flat_layout(
        self,
        module_path: str,
        category: str,
    ) -> None:
        """Verify flat-layout plugins yield no kind.

        Args:
            module_path: Dotted plugin module path.
            category: Category key the module belongs to.
        """
        assert _derive_kind(module_path, category) is None

    def test_returns_none_for_unknown_subfolder(self) -> None:
        """Verify subfolders other than single/compare yield None."""
        result = _derive_kind(
            "my_cat.other_subdir.impl.impl",
            "my_cat",
        )
        assert result is None

    def test_returns_none_when_first_segment_not_category(self) -> None:
        """Verify defensive guard against malformed module paths."""
        result = _derive_kind(
            "wrong_root.single.impl.impl",
            "expected_category",
        )
        assert result is None

    def test_returns_none_for_too_short_path(self) -> None:
        """Verify a one-segment path does not raise and yields None."""
        assert _derive_kind("just_one", "just_one") is None


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


class TestDiscoverImplementationsKind:
    """Tests for kind propagation in _discover_implementations."""

    @patch("app.core.plugin_discovery.plugin_registry._find_plugin_modules")
    @patch("app.core.plugin_discovery.plugin_registry.PluginManager")
    def test_single_kind_is_set(
        self,
        mock_pm: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Verify plugins under single/ get kind="single"."""
        from app.core.plugin_discovery.plugin_registry import (
            _discover_implementations,
        )

        mock_find.return_value = ["my_cat.single.impl.impl"]
        mock_pm.load.return_value = _make_mock_plugin({"x": (int, 1)})

        impls = _discover_implementations("my_cat")

        assert len(impls) == 1
        assert impls[0].kind == "single"

    @patch("app.core.plugin_discovery.plugin_registry._find_plugin_modules")
    @patch("app.core.plugin_discovery.plugin_registry.PluginManager")
    def test_compare_kind_is_set(
        self,
        mock_pm: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Verify plugins under compare/ get kind="compare"."""
        from app.core.plugin_discovery.plugin_registry import (
            _discover_implementations,
        )

        mock_find.return_value = ["my_cat.compare.impl.impl"]
        mock_pm.load.return_value = _make_mock_plugin({"x": (int, 1)})

        impls = _discover_implementations("my_cat")

        assert len(impls) == 1
        assert impls[0].kind == "compare"

    @patch("app.core.plugin_discovery.plugin_registry._find_plugin_modules")
    @patch("app.core.plugin_discovery.plugin_registry.PluginManager")
    def test_flat_layout_kind_is_none(
        self,
        mock_pm: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Verify plugins not under a kind subfolder have kind=None."""
        from app.core.plugin_discovery.plugin_registry import (
            _discover_implementations,
        )

        mock_find.return_value = ["my_cat.impl.impl"]
        mock_pm.load.return_value = _make_mock_plugin({"x": (int, 1)})

        impls = _discover_implementations("my_cat")

        assert len(impls) == 1
        assert impls[0].kind is None


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


class TestBuildUIHintMap:
    """Tests for _build_ui_hint_map."""

    def test_empty_when_no_hints(self) -> None:
        """Verify empty dict when plugin declares no hints."""
        plugin = _make_mock_plugin({"x": (int, 1)})
        assert _build_ui_hint_map(plugin) == {}

    def test_maps_param_name_to_hint(self) -> None:
        """Verify hints are keyed by param_name."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_fmt",
        )
        plugin = _make_mock_plugin(
            {"neuron_id": (str, "0")},
            param_ui_hints=[hint],
        )
        result = _build_ui_hint_map(plugin)
        assert "neuron_id" in result
        assert result["neuron_id"] is hint

    def test_multiple_hints(self) -> None:
        """Verify multiple hints are all present in the map."""
        hint_a = DynamicDropdownHint(param_name="a", formatter="_f")
        hint_b = ParamUIHint(param_name="b")
        plugin = _make_mock_plugin(
            {"a": (str, "x"), "b": (int, 1)},
            param_ui_hints=[hint_a, hint_b],
        )
        result = _build_ui_hint_map(plugin)
        assert len(result) == 2
        assert "a" in result
        assert "b" in result


class TestResolveWidget:
    """Tests for _resolve_widget."""

    def test_dynamic_dropdown_hint_returns_dropdown(self) -> None:
        """Verify DynamicDropdownHint produces dropdown widget."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_format_choices",
        )
        widget, config = _resolve_widget(hint, "steering", "sae.sae")
        assert widget == "dropdown"
        assert config is not None
        assert isinstance(config, WidgetConfig)

    def test_dynamic_dropdown_hint_endpoint_format(self) -> None:
        """Verify choices_endpoint URL contains category, plugin, param."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_fmt",
        )
        _, config = _resolve_widget(
            hint,
            "inspection",
            "inspection.single.sae_inspection.sae_inspection",
        )
        assert config is not None
        assert config.choices_endpoint == (
            "/plugins/param-choices/inspection/inspection.single.sae_inspection.sae_inspection/neuron_id"
        )
        assert config.run_id_source == "neuron_labeling"

    def test_slider_hint_returns_slider(self) -> None:
        """Verify SliderHint produces slider widget with config."""
        hint = SliderHint(
            param_name="alpha",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )
        widget, config = _resolve_widget(hint, "steering", "sae.sae")
        assert widget == "slider"
        assert config is not None
        assert isinstance(config, WidgetConfig)
        assert config.slider_min == 0.0
        assert config.slider_max == 1.0
        assert config.slider_step == 0.01

    def test_base_hint_returns_text(self) -> None:
        """Verify base ParamUIHint falls back to text widget."""
        hint = ParamUIHint(param_name="x")
        widget, config = _resolve_widget(hint, "cat", "cat.impl.impl")
        assert widget == "text"
        assert config is None


class TestExtractParametersWithUIHints:
    """Tests for widget metadata in _extract_parameters_from_instance."""

    def test_param_without_hint_defaults_to_text(self) -> None:
        """Verify params without hints get widget='text'."""
        plugin = _make_mock_plugin({"k": (int, 10)})
        params = _extract_parameters_from_instance(plugin)
        assert params[0].widget == "text"
        assert params[0].widget_config is None

    def test_param_with_dropdown_hint(self) -> None:
        """Verify param with DynamicDropdownHint gets dropdown widget."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_fmt",
        )
        plugin = _make_mock_plugin(
            {"neuron_id": (str, "0")},
            param_ui_hints=[hint],
        )
        params = _extract_parameters_from_instance(
            plugin,
            "steering",
            "steering.sae.sae",
        )
        assert params[0].widget == "dropdown"
        assert params[0].widget_config is not None
        endpoint = params[0].widget_config.choices_endpoint
        assert endpoint is not None
        assert "param-choices" in endpoint

    def test_mixed_params_only_hinted_gets_widget(self) -> None:
        """Verify only the hinted param gets a widget override."""
        hint = DynamicDropdownHint(
            param_name="neuron_id",
            artifact_step="neuron_labeling",
            artifact_file="neuron_labels.json",
            formatter="_fmt",
        )
        plugin = _make_mock_plugin(
            {"neuron_id": (str, "0"), "k": (int, 10)},
            param_ui_hints=[hint],
        )
        params = _extract_parameters_from_instance(
            plugin,
            "inspection",
            "inspection.impl.impl",
        )
        neuron_param = next(p for p in params if p.name == "neuron_id")
        k_param = next(p for p in params if p.name == "k")
        assert neuron_param.widget == "dropdown"
        assert k_param.widget == "text"
        assert k_param.widget_config is None

    def test_param_with_slider_hint(self) -> None:
        """Verify param with SliderHint gets slider widget."""
        hint = SliderHint(
            param_name="alpha",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )
        plugin = _make_mock_plugin(
            {"alpha": (float, 0.3)},
            param_ui_hints=[hint],
        )
        params = _extract_parameters_from_instance(
            plugin,
            "steering",
            "steering.sae.sae",
        )
        assert params[0].widget == "slider"
        assert params[0].widget_config is not None
        assert params[0].widget_config.slider_min == 0.0
        assert params[0].widget_config.slider_max == 1.0
        assert params[0].widget_config.slider_step == 0.05

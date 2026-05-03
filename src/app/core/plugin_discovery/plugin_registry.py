"""Plugin discovery and registry logic.

Walks the ``src/plugins/`` directory tree, imports each plugin module,
introspects the ``Plugin.run()`` signature, and builds a structured
registry consumable by the frontend.
"""

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from app.config.config import PLUGIN_CATEGORIES
from app.core.plugin_discovery.plugin_manager import PluginManager
from app.models.plugin import (
    ArtifactDisplayModel,
    ArtifactFileModel,
    CategoryRegistryEntry,
    ImplementationInfo,
    ItemRowsDisplayModel,
    ParameterInfo,
    WidgetConfig,
)
from app.models.plugin import (
    DisplayRowSpec as DisplayRowModel,
)

if TYPE_CHECKING:
    from plugins.plugin_interface import (
        BasePlugin,
        DisplaySpec,
        ParamUIHint,
    )

PLUGINS_DIR = Path(__file__).resolve().parents[2].parent / "plugins"

SKIP_PARAMS = {"self"}


def _derive_kind(
    module_path: str,
    category_name: str,
) -> Literal["single", "compare"] | None:
    """Derive a plugin's kind from its dotted module path.

    A plugin is considered ``"single"`` or ``"compare"`` when the
    path segment immediately following the category name matches
    one of those literals.  Any other layout yields ``None`` so
    categories that do not opt into the single/compare distinction
    are unaffected.

    Args:
        module_path: Dotted module path produced by
            ``_find_plugin_modules`` (e.g.
            ``"inspection.single.sae_inspection.sae_inspection"``).
        category_name: Category key the module belongs to
            (e.g. ``"inspection"``).

    Returns:
        Literal["single", "compare"] | None: The derived kind, or
            ``None`` when the path does not include a kind subfolder.
    """
    parts = module_path.split(".")
    if len(parts) < 2 or parts[0] != category_name:
        return None

    candidate = parts[1]
    if candidate == "single":
        return "single"
    if candidate == "compare":
        return "compare"
    return None


def _build_ui_hint_map(
    plugin_instance: "BasePlugin",
) -> dict[str, "ParamUIHint"]:
    """Build a mapping from parameter name to its UI hint.

    Args:
        plugin_instance: An instantiated plugin object.

    Returns:
        dict[str, ParamUIHint]: Mapping of parameter names to
            their UI hints.  Empty if the plugin declares none.
    """
    return {hint.param_name: hint for hint in plugin_instance.io_spec.param_ui_hints}


def _resolve_widget(
    hint: "ParamUIHint",
    category_name: str,
    plugin_name: str,
) -> tuple[str, WidgetConfig | None]:
    """Convert a ParamUIHint to a widget type and config.

    Args:
        hint: The UI hint dataclass from the plugin's io_spec.
        category_name: Plugin category key (e.g. ``"steering"``).
        plugin_name: Dotted plugin module path.

    Returns:
        tuple[str, WidgetConfig | None]: The widget type string
            and optional widget configuration.
    """
    from plugins.plugin_interface import (
        DynamicDropdownHint,
        PastRunsDropdownHint,
        SliderHint,
    )

    if isinstance(hint, DynamicDropdownHint):
        endpoint = f"/plugins/param-choices/{category_name}/{plugin_name}/{hint.param_name}"
        return "dropdown", WidgetConfig(
            choices_endpoint=endpoint,
            run_id_source=hint.artifact_step,
        )

    if isinstance(hint, PastRunsDropdownHint):
        return "past_runs_dropdown", WidgetConfig(
            required_steps=list(hint.required_steps),
        )

    if isinstance(hint, SliderHint):
        return "slider", WidgetConfig(
            slider_min=hint.min_value,
            slider_max=hint.max_value,
            slider_step=hint.step,
        )

    return "text", None


def _extract_parameters_from_instance(
    plugin_instance: "BasePlugin",
    category_name: str = "",
    plugin_name: str = "",
) -> list[ParameterInfo]:
    """Extract configurable parameters from a plugin's run() method.

    Inspects the ``run()`` signature of an already-loaded plugin
    instance and returns metadata for every parameter except
    ``self``.  If the plugin declares ``param_ui_hints``, matching
    parameters receive ``widget`` and ``widget_config`` overrides.

    Args:
        plugin_instance: An instantiated plugin object.
        category_name: Plugin category key (e.g. ``"steering"``).
        plugin_name: Dotted plugin module path.

    Returns:
        list[ParameterInfo]: Ordered list of parameter descriptors.
    """
    sig = inspect.signature(plugin_instance.run)
    hint_map = _build_ui_hint_map(plugin_instance)

    params: list[ParameterInfo] = []
    for name, param in sig.parameters.items():
        if name in SKIP_PARAMS:
            continue

        has_default = param.default is not inspect.Parameter.empty
        type_name = (
            param.annotation.__name__ if param.annotation is not inspect.Parameter.empty else "str"
        )

        widget = "text"
        widget_config = None
        if name in hint_map:
            widget, widget_config = _resolve_widget(
                hint_map[name],
                category_name,
                plugin_name,
            )

        params.append(
            ParameterInfo(
                name=name,
                type=type_name,
                default=param.default if has_default else None,
                required=not has_default,
                widget=widget,
                widget_config=widget_config,
            )
        )

    return params


def _find_plugin_modules(category_dir: Path, prefix: str) -> list[str]:
    """Recursively find plugin module paths under a category directory.

    A valid plugin module is a ``.py`` file whose stem matches its
    parent directory name (e.g. ``elsa_trainer/elsa_trainer.py``).
    Intermediate directories like ``single/`` or ``compare/`` are
    traversed transparently.

    Args:
        category_dir: Absolute path to the category folder
            (e.g. ``src/plugins/dataset_loading/``).
        prefix: Dotted path prefix accumulated so far
            (e.g. ``dataset_loading``).

    Returns:
        list[str]: Dotted module paths suitable for ``PluginManager.load()``.
    """
    modules: list[str] = []

    for child in sorted(category_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue

        plugin_file = child / f"{child.name}.py"
        if plugin_file.is_file():
            modules.append(f"{prefix}.{child.name}.{child.name}")
        else:
            modules.extend(_find_plugin_modules(child, f"{prefix}.{child.name}"))

    return modules


def _make_display_name(plugin_module_path: str) -> str:
    """Derive a human-readable display name from a dotted module path.

    Takes the second-to-last segment (the implementation directory name)
    and converts it from snake_case to Title Case.

    Args:
        plugin_module_path: Dotted module path
            (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).

    Returns:
        str: Human-readable name (e.g. ``Movieles Loader``).
    """
    parts = plugin_module_path.split(".")
    impl_name = parts[-2]
    return impl_name.replace("_", " ").title()


def _convert_display_spec(
    spec: "DisplaySpec | None",
) -> ItemRowsDisplayModel | ArtifactDisplayModel | None:
    """Convert a dataclass DisplaySpec to a Pydantic model.

    Handles both ``ItemRowsDisplaySpec`` and
    ``ArtifactDisplaySpec`` variants.

    Args:
        spec: Dataclass display spec from the plugin's
            ``io_spec``, or ``None``.

    Returns:
        ItemRowsDisplayModel | ArtifactDisplayModel | None:
            Pydantic model for JSON serialisation, or ``None``
            if *spec* is ``None``.
    """
    if spec is None:
        return None

    from plugins.plugin_interface import (
        ArtifactDisplaySpec,
        ItemRowsDisplaySpec,
    )

    if isinstance(spec, ItemRowsDisplaySpec):
        return ItemRowsDisplayModel(
            rows=[DisplayRowModel(key=r.key, label=r.label) for r in spec.rows],
        )
    if isinstance(spec, ArtifactDisplaySpec):
        return ArtifactDisplayModel(
            files=[
                ArtifactFileModel(
                    filename=f.filename,
                    label=f.label,
                    content_type=f.content_type,
                )
                for f in spec.files
            ],
        )
    return None


def _discover_implementations(
    category_name: str,
) -> list[ImplementationInfo]:
    """Discover all plugin implementations for a given category.

    Args:
        category_name: Key from ``PLUGIN_CATEGORIES``
            (e.g. ``dataset_loading``).

    Returns:
        list[ImplementationInfo]: All discovered implementations
            with their parameters.
    """
    category_dir = PLUGINS_DIR / category_name
    module_paths = _find_plugin_modules(category_dir, category_name)

    implementations: list[ImplementationInfo] = []
    for module_path in module_paths:
        plugin_instance = PluginManager.load(module_path)
        params = _extract_parameters_from_instance(
            plugin_instance,
            category_name,
            module_path,
        )
        display_name = plugin_instance.name or _make_display_name(module_path)
        display = _convert_display_spec(plugin_instance.io_spec.display)
        kind = _derive_kind(module_path, category_name)
        implementations.append(
            ImplementationInfo(
                plugin_name=module_path,
                display_name=display_name,
                params=params,
                display=display,
                kind=kind,
            )
        )

    return implementations


def get_plugin_registry() -> dict[str, CategoryRegistryEntry]:
    """Build the complete plugin registry for all categories.

    Iterates over ``PLUGIN_CATEGORIES``, discovers implementations
    for each, and assembles a registry keyed by category name.

    Returns:
        dict[str, CategoryRegistryEntry]: Full registry mapping
            category names to their metadata and implementations.
    """
    registry: dict[str, CategoryRegistryEntry] = {}

    for category_name, category_info in PLUGIN_CATEGORIES.items():
        implementations = _discover_implementations(category_name)
        registry[category_name] = CategoryRegistryEntry(
            category_info=category_info,
            implementations=implementations,
        )

    return registry

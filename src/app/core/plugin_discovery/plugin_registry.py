"""Plugin discovery and registry logic.

Walks the ``src/plugins/`` directory tree, imports each plugin module,
introspects the ``Plugin.run()`` signature, and builds a structured
registry consumable by the frontend.
"""

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from app.config.config import PLUGIN_CATEGORIES
from app.core.plugin_discovery.naming import make_plugin_display_name
from app.core.plugin_discovery.plugin_manager import PluginManager
from app.models.plugin import (
    ArtifactDisplayModel,
    ArtifactFileModel,
    CategoryRegistryEntry,
    ImplementationInfo,
    ItemRowsDisplayModel,
    ParameterInfo,
    ParamGroup,
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
    from plugins.plugin_interface import ParamGroup as AuthoringParamGroup

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
        DependentDropdownHint,
        DynamicDropdownHint,
        PastRunsDropdownHint,
        SliderHint,
        StaticDropdownHint,
        ToggleHint,
    )

    if isinstance(hint, StaticDropdownHint):
        return "dropdown", WidgetConfig(
            choices=[{"label": v, "value": v} for v in hint.choices],
        )

    if isinstance(hint, DependentDropdownHint):
        endpoint = f"/plugins/param-choices/{category_name}/{plugin_name}/{hint.param_name}"
        return "dropdown", WidgetConfig(
            choices_endpoint=endpoint,
            source_param=hint.depends_on_param,
        )

    if isinstance(hint, DynamicDropdownHint):
        endpoint = f"/plugins/param-choices/{category_name}/{plugin_name}/{hint.param_name}"
        return "dropdown", WidgetConfig(
            choices_endpoint=endpoint,
            run_id_source=hint.artifact_step,
            source_run_param=hint.source_run_param,
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

    if isinstance(hint, ToggleHint):
        return "toggle", None

    return "text", None


def _first_description(metadata: tuple[Any, ...]) -> str | None:
    """Return the first usable string in annotation metadata.

    Args:
        metadata: The ``__metadata__`` tuple of a ``typing.Annotated``
            alias.

    Returns:
        str | None: The first stripped, non-empty string entry, or
            ``None`` when no usable string is present.
    """
    for item in metadata:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                return stripped
    return None


def _parse_annotation(annotation: Any) -> tuple[str, str | None]:
    """Resolve a parameter annotation to a type name and description.

    Handles three shapes of ``inspect.Parameter.annotation``:

    - ``inspect.Parameter.empty`` (no annotation) → ``("str", None)``.
    - A plain type (e.g. ``int``) → its name with no description.
    - A ``typing.Annotated`` alias → the wrapped type's name plus the
      first non-empty string found in its metadata, if any.

    A defensive ``getattr(..., "str")`` fallback is used for annotations
    that expose no ``__name__`` (e.g. ``int | None``) so introspection
    never raises.

    Args:
        annotation: The raw ``inspect.Parameter.annotation`` value.

    Returns:
        tuple[str, str | None]: The display type name and the first
            non-empty, stripped string description from the
            annotation's metadata, or ``None`` when absent.
    """
    if annotation is inspect.Parameter.empty:
        return "str", None

    metadata = getattr(annotation, "__metadata__", None)
    if metadata is not None:
        wrapped = annotation.__origin__
        return getattr(wrapped, "__name__", "str"), _first_description(metadata)

    return getattr(annotation, "__name__", "str"), None


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
        type_name, description = _parse_annotation(param.annotation)

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
                description=description,
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


def _convert_param_group(
    group: "AuthoringParamGroup",
    param_names: list[str],
    grouped: set[str],
) -> ParamGroup:
    """Validate and convert one authoring ``ParamGroup`` to its API model.

    Recurses into ``subgroups``.  Records every claimed parameter in
    ``grouped`` so the caller can detect leftovers, and enforces that a
    parameter is claimed by at most one (sub)group across the whole tree.

    Args:
        group: A plugin's declared ``ParamGroup`` (authoring dataclass).
        param_names: All parameter names the plugin's ``run()`` declares.
        grouped: Mutable set of parameter names already claimed; updated
            in place as this group and its subgroups are walked.

    Returns:
        ParamGroup: The API-model section, with nested subgroups.

    Raises:
        ValueError: If the group names an unknown parameter, or re-uses
            a parameter already claimed by another (sub)group.
    """
    for name in group.params:
        if name not in param_names:
            raise ValueError(
                f"Param group '{group.title}' names unknown parameter "
                f"'{name}'; declared run() params: {param_names}"
            )
        if name in grouped:
            raise ValueError(
                f"Param group '{group.title}' re-uses parameter '{name}'; "
                "each parameter may belong to only one group."
            )
        grouped.add(name)

    return ParamGroup(
        title=group.title,
        params=list(group.params),
        subgroups=[_convert_param_group(sub, param_names, grouped) for sub in group.subgroups],
    )


def _build_param_groups(
    plugin_instance: "BasePlugin",
    params: list[ParameterInfo],
) -> list[ParamGroup]:
    """Build the parameter-form sections for a plugin.

    Returns the plugin's declared ``param_groups`` in order — nesting
    preserved — with any parameters left out of every group (at any
    level) appended as a trailing top-level "Other" section.  Empty
    when the plugin declares no groups (the frontend then renders a
    flat list).

    Args:
        plugin_instance: An instantiated plugin object.
        params: The plugin's extracted parameters, in signature order.

    Returns:
        list[ParamGroup]: Ordered sections covering every parameter.

    Raises:
        ValueError: If a group names a parameter the plugin's ``run()``
            does not declare, or re-uses one across (sub)groups.
    """
    declared = plugin_instance.io_spec.param_groups
    if not declared:
        return []

    param_names = [p.name for p in params]
    grouped: set[str] = set()
    groups = [_convert_param_group(g, param_names, grouped) for g in declared]

    leftovers = [name for name in param_names if name not in grouped]
    if leftovers:
        groups.append(ParamGroup(title="Other", params=leftovers))

    return groups


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
        display_name = plugin_instance.name or make_plugin_display_name(module_path)
        # getattr (not direct access) so a third-party plugin subclassing
        # an older BasePlugin snapshot without ``description`` can't break
        # discovery.
        description = getattr(plugin_instance, "description", None)
        display = _convert_display_spec(plugin_instance.io_spec.display)
        kind = _derive_kind(module_path, category_name)
        param_groups = _build_param_groups(plugin_instance, params)
        implementations.append(
            ImplementationInfo(
                plugin_name=module_path,
                display_name=display_name,
                description=description,
                params=params,
                display=display,
                kind=kind,
                param_groups=param_groups,
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
